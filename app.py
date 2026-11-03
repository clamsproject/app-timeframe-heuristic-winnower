"""
TimeFrame Heuristic Winnower CLAMS app.

Treats every view in the input MMIF that carries time-anchored annotations
as one source of candidate segment boundaries and keeps the boundaries
that enough sources agree on. A view with Sentence annotations votes with
the endpoints of its sentences (resolved through Token to TimeFrame
alignments); any other view votes with the endpoints of its TimeFrames,
skipping word-level frames that are aligned to Tokens.

This module is the MMIF layer: it collects events per the ``document`` of
the TimeFrames, and writes segments with that same ``document``. The rules
themselves live in ``winnow.py`` as pure functions over integer
milliseconds, with no knowledge of MMIF or documents.

Output is one ``segment`` TimeFrame per interval between consecutive
selected boundaries. The app attaches no text; downstream consumers can
slice the transcript for any output TimeFrame with the SDK's
``text_document_helper``.
"""

import argparse
import logging
import math
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from clams import ClamsApp, Restifier
from mmif import Annotation, AnnotationTypes, Mmif
from mmif.utils import video_document_helper as vdh
from mmif.utils.timeunit_helper import convert as convert_time

from winnow import winnow

SEGMENT_LABEL = 'segment'
FRAME_UNITS = ('frame', 'frames')
DURATION_PROPERTIES = ('duration', 'durationMs')

# a voting source: (view id, labelsetUri), with '' for no labelsetUri
Source = Tuple[str, str]
# source -> sorted candidate event times in milliseconds
EventsBySource = Dict[Source, List[int]]
# document id -> candidate events of that document, by source
EventsByDocument = Dict[str, EventsBySource]


def find_document(mmif: Mmif, doc_id: str) -> Optional[Annotation]:
    """The document with ``doc_id``, or ``None`` when the MMIF holds none."""
    try:
        return mmif[doc_id]
    except KeyError:
        return None


def framerate_or_none(mmif: Mmif, doc_id: str) -> Optional[float]:
    """Frame rate of the document, or ``None`` when it cannot be determined.
    Only needed when a view declares its time unit in frames."""
    try:
        return vdh.get_framerate(find_document(mmif, doc_id))
    except Exception:
        return None


def document_of(tf: Annotation) -> str:
    """The ``document`` of a TimeFrame, as declared on it or on its view."""
    doc_id = tf.get_property('document')
    if not doc_id:
        raise ValueError(f'TimeFrame {tf.id} has no `document` property; '
                         f'the input MMIF is malformed.')
    return doc_id


def timeframe_unit(mmif: Mmif, tf: Annotation) -> str:
    """Time unit declared by the view that holds ``tf``."""
    contain = mmif[tf.parent].metadata.contains.get(AnnotationTypes.TimeFrame)
    unit = contain.get('timeUnit') if contain is not None else None
    return unit or 'milliseconds'


def to_ms(value, unit: Optional[str], fps: Optional[float]) -> int:
    unit = unit or 'milliseconds'
    if unit.lower() in FRAME_UNITS and fps is None:
        raise ValueError('a view declares its time unit in frames but the '
                         'frame rate of the video cannot be determined')
    return int(round(convert_time(value, unit, 'milliseconds', fps or 1)))


def timeframe_span(mmif: Mmif, tf: Annotation) -> Tuple[int, int]:
    unit = timeframe_unit(mmif, tf)
    fps = (framerate_or_none(mmif, document_of(tf))
           if unit.lower() in FRAME_UNITS else None)
    return (to_ms(mmif.get_start(tf), unit, fps),
            to_ms(mmif.get_end(tf), unit, fps))


def _qualify(view_id: str, ann_id: str) -> str:
    """Prefix a short annotation id with its view id."""
    if Mmif.id_delimiter in ann_id:
        return ann_id
    return f'{view_id}{Mmif.id_delimiter}{ann_id}'


def _event_store() -> Dict[str, Dict[Source, Set[int]]]:
    return defaultdict(lambda: defaultdict(set))


def source_of(tf: Annotation) -> Source:
    """The voting source of a TimeFrame: its view, split by the
    ``labelsetUri`` it declares, so one view can hold several kinds of
    TimeFrames that vote separately."""
    return tf.parent, tf.get_property('labelsetUri') or ''


def agreement_threshold(ratio: float, n_sources: int) -> int:
    """K: the number of sources that must agree, as the share ``ratio`` of
    ``n_sources``, rounded up and at least 1."""
    if not 0 < ratio <= 1:
        raise ValueError(f'minAgreementRatio must be in (0, 1], got {ratio}')
    # the small offset keeps an exact product such as 0.6 * 5 from rounding
    # up past 3 through floating point error
    return max(1, math.ceil(ratio * n_sources - 1e-9))


def sentence_events(mmif: Mmif) -> Dict[str, Dict[Source, Set[int]]]:
    """Start and end of every Sentence, per document and view, resolved
    through the TimeFrames aligned to the Sentence's target Tokens. The
    document is the one of those TimeFrames, since a Sentence itself points
    to its text document."""
    events = _event_store()
    for view in mmif.get_all_views_contain(AnnotationTypes.Sentence):
        for sentence in view.get_annotations(AnnotationTypes.Sentence):
            spans = defaultdict(list)
            for target in sentence.get_property('targets') or []:
                token = mmif[_qualify(view.id, str(target))]
                for aligned in token.get_all_aligned():
                    if aligned.is_type(AnnotationTypes.TimeFrame):
                        spans[document_of(aligned)].append(
                            timeframe_span(mmif, aligned))
            for doc_id, doc_spans in spans.items():
                source = (view.id, '')
                events[doc_id][source].add(min(s for s, _ in doc_spans))
                events[doc_id][source].add(max(e for _, e in doc_spans))
    return events


def timeframe_events(mmif: Mmif, labels: Sequence[str],
                     labels_ignored: Sequence[str],
                     skip_views: Set[str]
                     ) -> Dict[str, Dict[Source, Set[int]]]:
    """Start and end of every TimeFrame that passes the label filters, per
    document and source (see :func:`source_of`), skipping views in
    ``skip_views`` and word-level frames that are aligned to a Token."""
    events = _event_store()
    for tf in mmif.get_annotations_of_labels(
            AnnotationTypes.TimeFrame, labels, labels_ignored):
        if tf.parent in skip_views:
            continue
        if any(a.is_type(AnnotationTypes.Token)
               for a in tf.get_all_aligned()):
            continue
        events[document_of(tf)][source_of(tf)].update(
            timeframe_span(mmif, tf))
    return events


def collect_events(mmif: Mmif, labels: Sequence[str],
                   labels_ignored: Sequence[str]) -> EventsByDocument:
    """Candidate events of every voting source, keyed by document. A view
    with Sentence events votes with those, and any other view with its
    TimeFrames."""
    sentences = sentence_events(mmif)
    voted = {view_id for by_source in sentences.values()
             for view_id, _ in by_source}
    frames = timeframe_events(mmif, labels, labels_ignored, voted)
    events: EventsByDocument = defaultdict(dict)
    for source in (sentences, frames):
        for doc_id, by_source in source.items():
            for key, times in by_source.items():
                events[doc_id][key] = sorted(times)
    return dict(events)


def duration_ms(mmif: Mmif, doc_id: str, sources: EventsBySource) -> int:
    """Duration from the document when it carries one, otherwise the latest
    event of the document's sources."""
    doc = find_document(mmif, doc_id)
    if doc is not None:
        for key in DURATION_PROPERTIES:
            value = doc.get_property(key)
            if value is not None:
                return int(float(value))
    return max(max(times) for times in sources.values())


class TimeframeHeuristicWinnower(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # the SDK loads ``metadata.py`` from this directory
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        new_view = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(AnnotationTypes.TimeFrame,
                             timeUnit='milliseconds')

        ratio = parameters['minAgreementRatio']
        events = collect_events(mmif, parameters['tfLabels'],
                                parameters['tfLabelIgnore'])
        if not events:
            warnings.warn('no view produced candidate events; no boundary '
                          'can be selected')
        for doc_id, sources in events.items():
            for (view_id, uri), times in sources.items():
                app_uri = mmif[view_id].metadata.app
                kind = f' [{uri}]' if uri else ''
                self.logger.debug(f'{doc_id}: source {view_id}{kind} '
                                  f'({app_uri}): {len(times)} events')
            min_agreement = agreement_threshold(ratio, len(sources))
            self.logger.debug(f'{doc_id}: {len(sources)} sources, '
                              f'minimum agreement {min_agreement}')
            if min_agreement == 1:
                warnings.warn(f'document {doc_id} has {len(sources)} '
                              f'source(s) and minAgreementRatio={ratio}, so '
                              f'one source is enough to make a boundary; '
                              f'its boundaries are not corroborated')
            duration = duration_ms(mmif, doc_id, sources)
            if duration <= 0:
                warnings.warn(f'could not determine the duration of '
                              f'document {doc_id}')
                continue
            segments = winnow(sources, parameters['windowMs'], min_agreement,
                              parameters['minSeparationMs'], duration)
            self.logger.debug(f'{doc_id}: {len(segments)} segments')
            for start, end in segments:
                new_view.new_annotation(
                    AnnotationTypes.TimeFrame, document=doc_id,
                    start=start, end=end, label=SEGMENT_LABEL)
        return mmif


def get_app():
    return TimeframeHeuristicWinnower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000",
                        help="set port to listen")
    parser.add_argument("--production", action="store_true",
                        help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
