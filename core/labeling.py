"""Разметка суб-эпох по респираторным событиям (трёхуровневая)."""
from __future__ import annotations
from typing import List, Tuple

Event = Tuple[float, float]


def label_subepoch(epoch_start: float, epoch_end: float, events: List[Event],
                   t_pos: float = 0.5, buffer_sec: float = 30.0) -> dict:
    """
    Классификация суб-эпохи относительно респираторных событий.

    label:
      1  → overlap_fraction >= t_pos (апноэ)
      0  → overlap_fraction == 0 (нет события)
      -1 → промежуточное значение (discard)

    is_tonic: True, если нет событий в пределах buffer_sec от границ эпохи
              (используется для patient-z нормализации).
    """
    dur = epoch_end - epoch_start
    if dur <= 0:
        return {"label": -1, "overlap_fraction": 0.0, "n_events": 0,
                "has_event_onset": 0, "has_event_offset": 0, "is_tonic": 0}

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
    if frac >= t_pos:
        label = 1
    elif frac == 0.0:
        label = 0
    else:
        label = -1

    is_tonic = 1 if (frac == 0.0 and nearest_dist >= buffer_sec) else 0

    return {
        "label": label,
        "overlap_fraction": frac,
        "n_events": n_events,
        "has_event_onset": has_onset,
        "has_event_offset": has_offset,
        "is_tonic": is_tonic,
    }