"""
Agreement-based boundary selection over candidate events from several
sources.

Every source (one MMIF view) contributes a sorted list of candidate
boundary times in integer milliseconds. Widening each event into a window
of ``window_ms`` turns source agreement into a piecewise-constant signal
A(t) counting the distinct sources that cover t. Each maximal run with
A(t) >= ``min_agreement`` becomes one weighted candidate boundary, and the
candidate set that maximizes total agreement subject to a minimum
separation is selected exactly by weighted interval scheduling (dynamic
programming). All functions are pure and operate on integer milliseconds.
"""

import bisect
from collections import defaultdict
from typing import Dict, List, Tuple

# (start_ms, end_ms, number of distinct sources covering the interval)
Piece = Tuple[int, int, int]
# (position_ms, agreement strength)
Candidate = Tuple[int, int]


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or touching intervals so a burst of events from
    one source contributes one continuous span of coverage, not many."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def agreement_signal(events_by_source: Dict[str, List[int]],
                     window_ms: int) -> List[Piece]:
    """Sweep-line count of distinct sources covering each point of the
    timeline. Returns piecewise-constant pieces in time order, covering
    every interval where at least one source has coverage."""
    half = max(window_ms, 0) // 2
    delta: Dict[int, int] = defaultdict(int)
    for times in events_by_source.values():
        for s, e in merge_intervals([(t - half, t + half) for t in times]):
            delta[s] += 1
            delta[e] -= 1

    times = sorted(delta)
    pieces: List[Piece] = []
    level = 0
    for i, t in enumerate(times):
        level += delta[t]
        t_next = times[i + 1] if i + 1 < len(times) else t
        if t_next > t:
            pieces.append((t, t_next, level))
    return pieces


def candidates_from_signal(pieces: List[Piece],
                           min_agreement: int) -> List[Candidate]:
    """Take every maximal run where the agreement signal meets
    ``min_agreement`` as one candidate. Its strength is the peak agreement
    in the run and its position is the midpoint of the peak plateau."""
    candidates: List[Candidate] = []
    run: List[Piece] = []
    for piece in pieces:
        if piece[2] >= min_agreement:
            run.append(piece)
        elif run:
            candidates.append(_candidate_from_run(run))
            run = []
    if run:
        candidates.append(_candidate_from_run(run))
    return candidates


def _candidate_from_run(run: List[Piece]) -> Candidate:
    peak = max(level for _, _, level in run)
    plateau = [(s, e) for s, e, level in run if level == peak]
    lo, hi = plateau[0][0], plateau[-1][1]
    return (lo + hi) // 2, peak


def select_boundaries_dp(candidates: List[Candidate],
                         min_separation_ms: int,
                         duration_ms: int) -> List[int]:
    """Select the maximum-total-strength subset of ``candidates`` with
    pairwise separation >= ``min_separation_ms``, excluding any candidate
    within ``min_separation_ms`` of the start or the end of the recording.
    Solved exactly by weighted interval scheduling, so the boundary kept
    in a close pair is the strongest, never merely the earliest."""
    m = max(min_separation_ms, 0)
    cand = sorted((pos, strength) for pos, strength in candidates
                  if m <= pos <= duration_ms - m)
    n = len(cand)
    if n == 0:
        return []

    positions = [c[0] for c in cand]
    strengths = [c[1] for c in cand]

    # prev[i]: index of the last candidate compatible with i (or -1)
    prev = [-1] * n
    for i in range(n):
        j = bisect.bisect_right(positions, positions[i] - m) - 1
        prev[i] = j if j < i else i - 1

    best = [0] * n
    take = [False] * n
    for i in range(n):
        skip_score = best[i - 1] if i >= 1 else 0
        take_score = strengths[i] + (best[prev[i]] if prev[i] >= 0 else 0)
        take[i] = take_score >= skip_score
        best[i] = take_score if take[i] else skip_score

    chosen: List[int] = []
    i = n - 1
    while i >= 0:
        if take[i]:
            chosen.append(positions[i])
            i = prev[i]
        else:
            i -= 1
    chosen.reverse()
    return chosen


def segments_from_boundaries(boundaries: List[int],
                             duration_ms: int) -> List[Tuple[int, int]]:
    """Turn boundary times into ``(start, end)`` segments that tile
    ``[0, duration_ms]``."""
    cuts = sorted({0, duration_ms,
                   *(t for t in boundaries if 0 < t < duration_ms)})
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
