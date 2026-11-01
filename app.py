"""
TimeFrame Winnower CLAMS app.

Treats every view in the input MMIF that carries time-anchored annotations
as one source of candidate segment boundaries and keeps the boundaries
that enough sources agree on. A view with Sentence annotations votes with
the endpoints of its sentences (resolved through Token to TimeFrame
alignments); any other view votes with the endpoints of its TimeFrames,
skipping word-level frames that are aligned to Tokens. The fusion itself
lives in ``winnow.py`` as pure functions over integer milliseconds.

Output is one ``segment`` TimeFrame per interval between consecutive
selected boundaries. The app attaches no text; downstream consumers can
slice the transcript for any output TimeFrame with the SDK's
``text_document_helper``.
"""

import argparse
import logging
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

from clams import ClamsApp, Restifier
from mmif import Annotation, AnnotationTypes, Document, DocumentTypes, Mmif
from mmif.utils import video_document_helper as vdh
from mmif.utils.timeunit_helper import convert as convert_time

from winnow import (
    agreement_signal,
    candidates_from_signal,
    segments_from_boundaries,
    select_boundaries_dp,
)

SEGMENT_LABEL = 'segment'
FRAME_UNITS = ('frame', 'frames')
DURATION_PROPERTIES = ('duration', 'durationMs')

# view id -> sorted candidate event times in milliseconds
EventsBySource = Dict[str, List[int]]


def framerate_or_none(video_doc: Document) -> Optional[float]:
    """Frame rate of the video, or ``None`` when the media file cannot be
    opened. Only needed when a view declares its time unit in frames."""
    try:
        return vdh.get_framerate(video_doc)
    except Exception:
        return None


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


def timeframe_span(mmif: Mmif, tf: Annotation,
                   fps: Optional[float]) -> Tuple[int, int]:
    unit = timeframe_unit(mmif, tf)
    return (to_ms(mmif.get_start(tf), unit, fps),
            to_ms(mmif.get_end(tf), unit, fps))


def _qualify(view_id: str, ann_id: str) -> str:
    """Prefix a short annotation id with its view id."""
    if Mmif.id_delimiter in ann_id:
        return ann_id
    return f'{view_id}{Mmif.id_delimiter}{ann_id}'


def sentence_events(mmif: Mmif, fps: Optional[float]) -> EventsBySource:
    """Start and end of every Sentence, per view, resolved through the
    TimeFrames aligned to the Sentence's target Tokens."""
    events: EventsBySource = {}
    for view in mmif.get_all_views_contain(AnnotationTypes.Sentence):
        view_events: List[int] = []
        for sentence in view.get_annotations(AnnotationTypes.Sentence):
            spans = []
            for target in sentence.get_property('targets') or []:
                token = mmif[_qualify(view.id, str(target))]
                for aligned in token.get_all_aligned():
                    if aligned.is_type(AnnotationTypes.TimeFrame):
                        spans.append(timeframe_span(mmif, aligned, fps))
            if spans:
                view_events.append(min(s for s, _ in spans))
                view_events.append(max(e for _, e in spans))
        if view_events:
            events[view.id] = sorted(set(view_events))
    return events


def timeframe_events(mmif: Mmif, fps: Optional[float],
                     labels: Sequence[str], labels_ignored: Sequence[str],
                     skip_views: Sequence[str]) -> EventsBySource:
    """Start and end of every TimeFrame that passes the label filters, per
    view, skipping views in ``skip_views`` and word-level frames that are
    aligned to a Token."""
    collected: Dict[str, set] = {}
    for tf in mmif.get_annotations_of_labels(
            AnnotationTypes.TimeFrame, labels, labels_ignored):
        if tf.parent in skip_views:
            continue
        if any(a.is_type(AnnotationTypes.Token)
               for a in tf.get_all_aligned()):
            continue
        span = timeframe_span(mmif, tf, fps)
        collected.setdefault(tf.parent, set()).update(span)
    return {view_id: sorted(times) for view_id, times in collected.items()}


def video_duration_ms(mmif: Mmif, video_doc: Document,
                      fps: Optional[float]) -> int:
    """Duration from the VideoDocument when it carries one, otherwise the
    latest TimeFrame end across all views."""
    for key in DURATION_PROPERTIES:
        value = video_doc.get_property(key)
        if value is not None:
            return int(float(value))
    latest = 0
    for tf in mmif.get_annotations_of_labels(AnnotationTypes.TimeFrame):
        latest = max(latest, timeframe_span(mmif, tf, fps)[1])
    return latest


class TimeframeWinnower(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # the SDK loads ``metadata.py`` from this directory
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        video_docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not video_docs:
            raise ValueError('TimeFrame Winnower requires a VideoDocument')
        video_doc = video_docs[0]
        fps = framerate_or_none(video_doc)

        new_view = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(AnnotationTypes.TimeFrame,
                             timeUnit='milliseconds', document=video_doc.id)

        events = sentence_events(mmif, fps)
        events.update(timeframe_events(
            mmif, fps, parameters['tfLabels'], parameters['tfLabelIgnore'],
            skip_views=list(events)))
        for view_id, times in events.items():
            app_uri = mmif[view_id].metadata.app
            self.logger.debug(f'source {view_id} ({app_uri}): '
                              f'{len(times)} events')

        min_agreement = parameters['minAgreement']
        if len(events) < min_agreement:
            warnings.warn(f'{len(events)} source view(s) produced candidate '
                          f'events, fewer than minAgreement={min_agreement}; '
                          f'no boundary can be selected')
            return mmif

        duration = video_duration_ms(mmif, video_doc, fps)
        if duration <= 0:
            warnings.warn('could not determine the duration of the recording')
            return mmif

        pieces = agreement_signal(events, parameters['windowMs'])
        candidates = candidates_from_signal(pieces, min_agreement)
        boundaries = select_boundaries_dp(
            candidates, parameters['minSeparationMs'], duration)
        self.logger.debug(f'{len(candidates)} candidates, '
                          f'{len(boundaries)} boundaries selected')

        for start, end in segments_from_boundaries(boundaries, duration):
            new_view.new_annotation(AnnotationTypes.TimeFrame,
                                    start=start, end=end, label=SEGMENT_LABEL)
        return mmif


def get_app():
    return TimeframeWinnower()


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
