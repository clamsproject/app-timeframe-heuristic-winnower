import sys
import warnings
from pathlib import Path

import pytest
from mmif import Mmif, Document, AnnotationTypes, DocumentTypes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402
import winnow  # noqa: E402

APP_URI_PREFIX = 'http://apps.clams.ai/timeframe-heuristic-winnower'
DURATION = 300_000
# Sources agree on boundaries at 60 s, 120 s, 180 s and 240 s; source "d"
# dissents at 180 s, source "e" votes alone at 150 s.
SOURCE_EVENTS = {
    'a': [60_000, 120_000, 180_000, 240_000],
    'b': [60_500, 119_500, 180_200, 240_300],
    'c': [59_800, 120_300, 179_900, 239_700],
    'd': [60_100, 120_100, 240_100],
    'e': [150_000],
}


def add_video(mmif: Mmif) -> Document:
    doc = Document()
    doc.at_type = DocumentTypes.VideoDocument
    doc.id = 'd1'
    doc.location = 'file:///nonexistent/video.mp4'
    doc.properties['duration'] = DURATION
    mmif.add_document(doc)
    return doc


def add_timeframe_view(mmif: Mmif, app_name: str, spans, label='shot',
                       unit='milliseconds', document='d1', labelset_uri=None):
    view = mmif.new_view()
    view.metadata.app = f'http://apps.clams.ai/{app_name}/v1'
    view.new_contain(AnnotationTypes.TimeFrame, timeUnit=unit,
                     document=document)
    for start, end in spans:
        props = dict(start=start, end=end)
        if label is not None:
            props['label'] = label
        if labelset_uri is not None:
            props['labelsetUri'] = labelset_uri
        view.new_annotation(AnnotationTypes.TimeFrame, **props)
    return view


def add_asr_view(mmif: Mmif, sentences):
    """``sentences``: list of (text, [(word, start_ms, end_ms), ...])."""
    view = mmif.new_view()
    view.metadata.app = 'http://apps.clams.ai/fake-asr/v1'
    text = ' '.join(s[0] for s in sentences)
    td = view.new_textdocument(text)
    view.new_contain(AnnotationTypes.Token, document=td.id)
    view.new_contain(AnnotationTypes.TimeFrame, timeUnit='milliseconds',
                     document='d1')
    view.new_contain(AnnotationTypes.Alignment)
    view.new_contain(AnnotationTypes.Sentence)
    offset = 0
    for sent_text, words in sentences:
        token_ids = []
        for word, start, end in words:
            tok = view.new_annotation(AnnotationTypes.Token, text=word,
                                      start=offset, end=offset + len(word),
                                      document=td.id)
            offset += len(word) + 1
            tf = view.new_annotation(AnnotationTypes.TimeFrame, start=start,
                                     end=end, label='speech')
            view.new_annotation(AnnotationTypes.Alignment,
                                source=tok.id, target=tf.id)
            token_ids.append(tok.id)
        view.new_annotation(AnnotationTypes.Sentence, text=sent_text,
                            targets=token_ids)
    return view


def reload(mmif: Mmif) -> Mmif:
    """Serialize and parse again so alignment caches are rebuilt."""
    return Mmif(mmif.serialize())


def events_to_spans(events):
    """Adjacent pairs of event times as (start, end) TimeFrames."""
    return [(events[i], events[i + 1]) for i in range(len(events) - 1)]


@pytest.fixture
def five_source_mmif():
    mmif = Mmif()
    add_video(mmif)
    for name, events in SOURCE_EVENTS.items():
        spans = events_to_spans([0] + events + [DURATION])
        add_timeframe_view(mmif, f'src-{name}', spans)
    return reload(mmif)


@pytest.fixture
def winnower():
    return app.get_app()


def run(winnower, mmif, **params):
    """``params`` values are query-string style: lists of strings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Mmif(winnower.annotate(mmif.serialize(), **params))


def output_views(out: Mmif):
    return [v for v in out.views
            if v.metadata.app.startswith(APP_URI_PREFIX)
            and len(v.annotations) > 0]


def segments(out: Mmif):
    views = output_views(out)
    assert len(views) == 1
    return [(tf.get_property('start'), tf.get_property('end'))
            for tf in views[0].get_annotations(AnnotationTypes.TimeFrame)]


class TestFusion:

    def test_agreement_signal_counts_distinct_sources(self):
        events = {'a': [10_000], 'b': [10_000]}
        assert winnow.agreement_signal(events, 2_000) == [(9_000, 11_000, 2)]

    def test_burst_from_one_source_counts_once(self):
        pieces = winnow.agreement_signal({'a': [10_000, 10_500, 11_000]},
                                         2_000)
        assert max(level for _, _, level in pieces) == 1

    def test_dp_keeps_stronger_of_two_close_candidates(self):
        chosen = winnow.select_boundaries_dp(
            [(100_000, 3), (110_000, 5)], 20_000, 300_000)
        assert chosen == [110_000]

    def test_dp_excludes_margins(self):
        chosen = winnow.select_boundaries_dp(
            [(5_000, 5), (150_000, 5), (298_000, 5)], 20_000, 300_000)
        assert chosen == [150_000]

    def test_segments_tile_recording(self):
        assert winnow.segments_from_boundaries([60_000, 120_000], 200_000) == [
            (0, 60_000), (60_000, 120_000), (120_000, 200_000)]


class TestLabelFilter:

    @pytest.fixture
    def labeled_and_unlabeled(self):
        mmif = Mmif()
        add_video(mmif)
        labeled = add_timeframe_view(mmif, 'x', [(0, 1000)], label='shot')
        unlabeled = add_timeframe_view(mmif, 'y', [(0, 1000)], label=None)
        return mmif, labeled.id, unlabeled.id

    def test_allow_list_drops_unlabeled(self, labeled_and_unlabeled):
        mmif, labeled, _ = labeled_and_unlabeled
        events = app.timeframe_events(mmif, ['shot'], [], set())
        assert set(events['d1']) == {(labeled, '')}

    def test_ignore_list_keeps_unlabeled(self, labeled_and_unlabeled):
        mmif, _, unlabeled = labeled_and_unlabeled
        events = app.timeframe_events(mmif, [], ['shot'], set())
        assert set(events['d1']) == {(unlabeled, '')}


class TestApp:

    def test_boundaries_where_four_of_five_agree(self, winnower,
                                                 five_source_mmif):
        out = run(winnower, five_source_mmif, minAgreementRatio=['0.8'])
        segs = segments(out)
        starts = [s for s, _ in segs]
        assert starts[0] == 0
        assert segs[-1][1] == DURATION
        # 60 s, 120 s and 240 s are agreed by four sources; 180 s has three
        # votes and the lone vote at 150 s has one, so neither survives
        assert len(starts) == 4
        for expected, actual in zip([60_000, 120_000, 240_000], starts[1:]):
            assert abs(actual - expected) < 2_000

    def test_output_view_declares_unit_and_label(self, winnower,
                                                 five_source_mmif):
        out = run(winnower, five_source_mmif)
        view = output_views(out)[0]
        contain = view.metadata.contains.get(AnnotationTypes.TimeFrame)
        assert contain['timeUnit'] == 'milliseconds'
        labels = {tf.get_property('label')
                  for tf in view.get_annotations(AnnotationTypes.TimeFrame)}
        assert labels == {'segment'}

    def test_threshold_is_a_share_of_sources(self):
        assert [app.agreement_threshold(0.5, n) for n in (1, 2, 3, 5)] == [
            1, 1, 2, 3]
        assert app.agreement_threshold(0.8, 5) == 4
        assert app.agreement_threshold(0.6, 5) == 3
        with pytest.raises(ValueError):
            app.agreement_threshold(0, 5)

    def test_ignore_label_removes_a_voter(self, winnower):
        mmif = Mmif()
        add_video(mmif)
        for name, events in SOURCE_EVENTS.items():
            label = 'speaker' if name == 'd' else 'shot'
            spans = events_to_spans([0] + events + [DURATION])
            add_timeframe_view(mmif, f'src-{name}', spans, label=label)
        mmif = reload(mmif)
        with_d = segments(run(winnower, mmif, minAgreementRatio=['0.8']))
        without_d = segments(run(winnower, mmif, minAgreementRatio=['0.8'],
                                 tfLabelIgnore=['speaker']))
        # with d voting, 60 s, 120 s and 240 s reach four votes; without d
        # no point does
        assert len(with_d) == 4
        assert len(without_d) == 1
        assert without_d == [(0, DURATION)]

    def test_own_output_does_not_vote_by_default(self, winnower,
                                                 five_source_mmif):
        out = run(winnower, five_source_mmif, minAgreementRatio=['0.8'])
        ignored = next(p.default for p in winnower.metadata.parameters
                       if p.name == 'tfLabelIgnore')
        events = app.timeframe_events(out, [], ignored, set())
        assert len(events['d1']) == len(SOURCE_EVENTS)
        assert (output_views(out)[0].id, '') not in events['d1']

    def test_seconds_unit_is_converted(self, winnower):
        mmif = Mmif()
        add_video(mmif)
        for name, events in SOURCE_EVENTS.items():
            spans = [(s / 1000, e / 1000)
                     for s, e in events_to_spans([0] + events + [DURATION])]
            add_timeframe_view(mmif, f'src-{name}', spans, unit='seconds')
        out = run(winnower, reload(mmif), minAgreementRatio=['0.8'])
        starts = [s for s, _ in segments(out)][1:]
        assert len(starts) == 3
        assert all(isinstance(s, int) for s in starts)
        assert abs(starts[0] - 60_000) < 2_000

    def test_sentences_vote_and_token_frames_are_skipped(self, winnower):
        mmif = Mmif()
        add_video(mmif)
        # ASR sentences end exactly at 60 s and 120 s
        add_asr_view(mmif, [
            ('hello there', [('hello', 1_000, 30_000),
                             ('there', 30_000, 60_000)]),
            ('second one', [('second', 61_000, 90_000),
                            ('one', 90_000, 120_000)]),
            ('third', [('third', 121_000, 200_000)]),
        ])
        for name in ('a', 'b', 'c'):
            add_timeframe_view(mmif, f'src-{name}',
                               events_to_spans([0, 60_000, 120_000, DURATION]))
        mmif = reload(mmif)
        sentence_events = app.sentence_events(mmif)
        assert [sorted(ts) for ts in sentence_events['d1'].values()] == [
            [1_000, 60_000, 61_000, 120_000, 121_000, 200_000]]
        tf_events = app.timeframe_events(mmif, [], [], skip_views=set())
        # the ASR view's word-level frames never appear as TimeFrame events
        assert set(tf_events['d1']) == {(f'v_{i}', '') for i in range(1, 4)}
        out = run(winnower, mmif, minAgreementRatio=['1.0'])
        starts = [s for s, _ in segments(out)]
        assert len(starts) == 3

    def test_output_carries_document_of_input(self, winnower,
                                              five_source_mmif):
        out = run(winnower, five_source_mmif)
        view = output_views(out)[0]
        documents = {tf.get_property('document')
                     for tf in view.get_annotations(AnnotationTypes.TimeFrame)}
        assert documents == {'d1'}

    def test_no_document_object_is_needed(self, winnower):
        mmif = Mmif()
        for name, events in SOURCE_EVENTS.items():
            spans = events_to_spans([0] + events + [DURATION])
            add_timeframe_view(mmif, f'src-{name}', spans)
        spans = segments(run(winnower, reload(mmif),
                             minAgreementRatio=['0.8']))
        # the length falls back to the latest event of the voting views
        assert spans[-1][1] == DURATION
        assert len(spans) == 4

    def test_documents_are_winnowed_separately(self, winnower):
        mmif = Mmif()
        add_video(mmif)
        for name, events in SOURCE_EVENTS.items():
            spans = events_to_spans([0] + events + [DURATION])
            add_timeframe_view(mmif, f'src-{name}', spans)
        # three views on another document, at times d1 has no agreement on
        for name in ('x', 'y', 'z'):
            add_timeframe_view(
                mmif, f'other-{name}',
                events_to_spans([0, 150_000, 300_000]), document='d2')
        out = run(winnower, reload(mmif), minAgreementRatio=['0.6'])
        view = output_views(out)[0]
        by_doc = {}
        for tf in view.get_annotations(AnnotationTypes.TimeFrame):
            by_doc.setdefault(tf.get_property('document'), []).append(
                (tf.get_property('start'), tf.get_property('end')))
        assert set(by_doc) == {'d1', 'd2'}
        assert by_doc['d2'] == [(0, 150_000), (150_000, 300_000)]
        assert all(150_000 not in span for span in by_doc['d1'])

    def test_single_source_warns_and_still_segments(self, winnower):
        mmif = Mmif()
        add_video(mmif)
        add_timeframe_view(mmif, 'lonely',
                           events_to_spans([0, 100_000, DURATION]))
        # annotate() records warnings in a separate view of the output; the
        # ``run`` helper silences them, so call annotate() directly
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            out = Mmif(winnower.annotate(reload(mmif).serialize()))
        assert any('not corroborated' in str(v.metadata.warnings)
                   for v in out.views if v.has_warnings())
        assert segments(out) == [(0, 100_000), (100_000, DURATION)]

    def test_labelset_uri_splits_a_view_into_sources(self):
        mmif = Mmif()
        add_video(mmif)
        view = add_timeframe_view(mmif, 'two-kinds', [(0, 60_000)],
                                  labelset_uri='http://example.org/shots')
        view.new_annotation(AnnotationTypes.TimeFrame, start=60_000,
                            end=120_000, label='speech',
                            labelsetUri='http://example.org/speech')
        events = app.collect_events(reload(mmif), [], [])
        assert set(events['d1']) == {
            (view.id, 'http://example.org/shots'),
            (view.id, 'http://example.org/speech')}

    def test_timeframe_without_document_is_an_error(self, winnower):
        mmif = Mmif()
        add_video(mmif)
        add_timeframe_view(mmif, 'x', [(0, 1000)], document=None)
        with pytest.raises(ValueError, match='`document`'):
            app.collect_events(mmif, [], [])
