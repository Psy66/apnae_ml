"""Разметка суб-эпох по респираторным событиям и артефактам."""
from __future__ import annotations
from typing import List, Optional, Tuple

Event = Tuple[float, float]


def label_subepoch(epoch_start: float, epoch_end: float, events: List[Event],
                   t_pos: float = 0.5, buffer_sec: float = 30.0,
                   artifacts: Optional[List[Event]] = None,
                   artifact_min_overlap: float = 1.0) -> dict:
    """
    label:
      1  → overlap_fraction >= t_pos
      0  → overlap_fraction == 0 и нет артефактов
      -1 → discard (промежуточное перекрытие или артефакт)
    """
    dur = epoch_end - epoch_start
    if dur <= 0:
        return {"label": -1, "overlap_fraction": 0.0, "n_events": 0,
                "has_event_onset": 0, "has_event_offset": 0,
                "is_tonic": 0, "artifact_overlap_sec": 0.0}

    total_overlap = 0.0
    n_events = 0
    has_onset = 0
    has_offset = 0
    nearest_dist = float("inf")

    for on, off in events:
        ov = max(0.0, min(epoch_end, off) - max(epoch_start, on))
        if ov > 0:
            total_overlap += ov
            n_events += 1
            if epoch_start <= on < epoch_end:
                has_onset = 1
            if epoch_start <= off < epoch_end:
                has_offset = 1
        if off <= epoch_start:
            nearest_dist = min(nearest_dist, epoch_start - off)
        elif on >= epoch_end:
            nearest_dist = min(nearest_dist, on - epoch_end)
        else:
            nearest_dist = 0.0

    frac = min(total_overlap, dur) / dur

    artifact_overlap = 0.0
    if artifacts:
        for a_on, a_off in artifacts:
            ov = max(0.0, min(epoch_end, a_off) - max(epoch_start, a_on))
            if ov > 0:
                artifact_overlap += ov

    if artifact_overlap >= artifact_min_overlap:
        label = -1
        is_tonic = 0
    elif frac >= t_pos:
        label = 1
        is_tonic = 0
    elif frac == 0.0:
        label = 0
        is_tonic = 1 if nearest_dist >= buffer_sec else 0
    else:
        label = -1
        is_tonic = 0

    return {
        "label": label,
        "overlap_fraction": frac,
        "n_events": n_events,
        "has_event_onset": has_onset,
        "has_event_offset": has_offset,
        "is_tonic": is_tonic,
        "artifact_overlap_sec": artifact_overlap,
    }