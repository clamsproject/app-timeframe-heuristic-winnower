# TimeFrame Winnower

## Description

The TimeFrame Winnower selects segment boundaries for a recording by agreement across upstream analysis views. Every view in the input MMIF that carries time-anchored annotations is one source of candidate boundary events. A boundary survives where enough sources agree, and the final boundary set is the best-corroborated one subject to a minimum segment length. The method learns nothing and needs no GPU.

How a view votes:

1. A view that contains `Sentence` annotations votes with the start and end of every sentence, resolved through its `Token` to `TimeFrame` alignments. This is the shape ASR apps such as [whisper-wrapper](https://apps.clams.ai/whisper-wrapper/) produce.
1. Any other view votes with the start and end of every `TimeFrame` it contains. Word-level frames that are aligned to a `Token` are skipped so a transcript does not flood the event set.
1. The rule looks only at annotation granularity, never at labels or app names, so any `TimeFrame` producer (shot detection, speaker diarization, audio classification, scene type detection, ...) can be a source without configuration.

How boundaries are selected (`winnow.py`):

1. Each event is widened into a window of `windowMs`. Overlapping windows from one source merge, so a burst of events from a single source counts once.
1. The agreement signal counts, at every point of the timeline, how many distinct sources cover it. Each maximal run where the count reaches `minAgreement` becomes one candidate boundary, placed at the midpoint of its peak and weighted by that peak.
1. The candidate subset with the largest total weight such that no two candidates are closer than `minSeparationMs` (and none is closer than that to the start or the end) is found exactly by weighted interval scheduling. Where two corroborated candidates compete for one span, the better-corroborated one survives.
1. Segments are the intervals between consecutive selected boundaries, tiling the whole recording.

The default parameters (W = 5000 ms, K = 4, M = 20000 ms) are the ones used over five sources in the JCDL 2026 paper that introduced the method ([doi:10.1145/3805696.3846037](https://doi.org/10.1145/3805696.3846037)). `minAgreement` is an absolute count of views: lower it when fewer sources are present, or no boundary can be selected.

## User instruction

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### Input

A MMIF with a `VideoDocument` and the views of the upstream apps whose output should vote. When the MMIF holds several `VideoDocument`s, the first one is used. Time values are read in the `timeUnit` each view declares in its `contains` metadata; a view that declares none is read as milliseconds. If a view declares its unit in frames, the video file must be readable so the frame rate can be determined. The recording length is taken from a `duration` property on the `VideoDocument` when present, otherwise from the latest `TimeFrame` end across all views.

Fewer than `minAgreement` voting views leaves nothing to select; the app then records a warning and emits no annotations.

### Output

One view with `TimeFrame` annotations labeled `segment`, with integer-millisecond `start` and `end`, tiling the recording from 0 to its duration. The app attaches no text. To get the transcript of a segment, slice it from the ASR view with the SDK's `mmif.utils.text_document_helper`.

### Configurable runtime parameters

The full list of parameters, with types, defaults, and descriptions, is in the app metadata: see the app's page in the [CLAMS App Directory](https://apps.clams.ai) or run `python metadata.py`. `windowMs`, `minAgreement`, and `minSeparationMs` are W, K, and M above. `tfLabels` and `tfLabelIgnore` are an allow list and an ignore list over `TimeFrame` labels; they select which frames may vote and leave Sentence-derived events untouched.

Example, excluding the audio classifier's frames from the vote:

```bash
curl -X POST -d@input.mmif "http://localhost:5000?tfLabelIgnore=speech&tfLabelIgnore=music&tfLabelIgnore=noise"
```

## System requirements

CPU only. Python dependencies are listed in `requirements.txt`; the app has no dependency beyond the CLAMS SDK.

## Development

Run the tests with `pytest` from the repository root. `tests/test_app.py` builds small synthetic MMIFs, so no media is needed.
