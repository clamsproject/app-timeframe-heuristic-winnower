# TimeFrame Heuristic Winnower

## Description

The TimeFrame Heuristic Winnower selects segment boundaries for a recording with heuristic rules based on agreement across upstream analysis views. Every view in the input MMIF that carries time-anchored annotations is one source of candidate boundary events. A boundary survives where enough sources agree, and the final boundary set is the best-corroborated one subject to a minimum segment length. The rules are hand-set: the method learns nothing and needs no GPU.

Heuristic rules for how a view votes:

1. A view that contains `Sentence` annotations votes with the start and end of every sentence, resolved through its `Token` to `TimeFrame` alignments. This is the shape ASR apps such as [whisper-wrapper](https://apps.clams.ai/whisper-wrapper/) produce.
1. Any other view votes with the start and end of every `TimeFrame` it contains. Word-level frames that are aligned to a `Token` are skipped so a transcript does not flood the event set.
1. The rule looks only at annotation granularity, never at labels or app names, so any `TimeFrame` producer (shot detection, speaker diarization, audio classification, scene type detection, ...) can be a source without configuration.

Heuristic rules for how boundaries are selected (`winnow.py`):

1. Each event is widened into a window of `windowMs`. Overlapping windows from one source merge, so a burst of events from a single source counts once.
1. The agreement signal counts, at every point of the timeline, how many distinct sources cover it. Each maximal run where the count reaches K becomes one candidate boundary, placed at the midpoint of its peak and weighted by that peak.
1. The candidate subset with the largest total weight such that no two candidates are closer than `minSeparationMs` (and none is closer than that to the start or the end) is found exactly by weighted interval scheduling. Where two corroborated candidates compete for one span, the better-corroborated one survives.
1. Segments are the intervals between consecutive selected boundaries, tiling the whole recording.

K is the share `minAgreementRatio` of the sources voting on a document, rounded up and at least 1. With the default of 0.5, K is 1 of 2 sources, 2 of 3, and 3 of 5. With two sources, any boundary of either source can be selected, and the selection prefers boundaries both agree on. When K is 1, the app records a warning that the boundaries are not corroborated.

The JCDL 2026 paper that introduced the method ([doi:10.1145/3805696.3846037](https://doi.org/10.1145/3805696.3846037)) used W = 5000 ms and M = 20000 ms, the defaults, with K = 4 over five sources, that is `minAgreementRatio=0.8`.

## User instruction

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### Input

A MMIF with the views of the upstream apps whose output should vote: views with `TimeFrame` annotations, or views with `Sentence` annotations whose `Token`s are aligned to `TimeFrame`s. Each `TimeFrame` must have a `document`, on the annotation or at view level, and an input without one is an error. The source media itself is not read, so the input can come from video or audio.

Votes are grouped by that `document`, and the rules run once per document, so several documents in one MMIF do not mix. The rules themselves work on time values alone.

Time values are read in the `timeUnit` each view declares in its `contains` metadata; a view that declares none is read as milliseconds. If a view declares its unit in frames, the video file must be readable so the frame rate can be determined. The length of each recording is taken from a `duration` property on its document when present, otherwise from the latest event among the views that vote on it.

A source is one view. When a view's `TimeFrame`s declare a `labelsetUri`, each value is a separate source, so a view that holds several kinds of `TimeFrame`s votes once per kind. Views with `Sentence`s are one source each.

### Output

One view with `TimeFrame` annotations labeled `segment`, with integer-millisecond `start` and `end`, tiling each document from 0 to its duration. Each one carries the `document` of the input `TimeFrame`s it was derived from. The app attaches no text. To get the transcript of a segment, slice it from the ASR view with the SDK's `mmif.utils.text_document_helper`.

### Configurable runtime parameters

The full list of parameters, with types, defaults, and descriptions, is in the app metadata: see the app's page in the [CLAMS App Directory](https://apps.clams.ai) or run `python metadata.py`. `windowMs` and `minSeparationMs` are W and M above, and `minAgreementRatio` sets K. `tfLabels` and `tfLabelIgnore` are an allow list and an ignore list over `TimeFrame` labels; they select which frames may vote and leave Sentence-derived events untouched. By default `tfLabelIgnore` holds `segment`, so `segment` frames from an earlier winnower run do not vote. Setting the parameter replaces that default, so add `segment` to your own list when the input holds such frames.

Example, excluding the audio classifier's frames from the vote:

```bash
curl -X POST -d@input.mmif "http://localhost:5000?tfLabelIgnore=segment&tfLabelIgnore=speech&tfLabelIgnore=music&tfLabelIgnore=noise"
```

## System requirements

CPU only. Python dependencies are listed in `requirements.txt`; the app has no dependency beyond the CLAMS SDK.

## Development

Run the tests with `pytest` from the repository root. `tests/test_app.py` builds small synthetic MMIFs, so no media is needed.
