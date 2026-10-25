"""Парсинг аннотаций из EDF (стадии, респираторные события, артефакты).

Формат аннотаций "Нейрон-Спектр":
  'Sleep stage 2(eventUnknown)'                          → N2
  'Обструктивное апноэ(pointPolySomnographyObstructiveApnea)'
  'Артефакт(blockArtefact)'

Работает по коду в скобках (ASCII-safe) и по префиксу для стадий.
"""
from __future__ import annotations
import logging
import re
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_STAGE_PREFIXES = {
    'Sleep stage W': 'W',
    'Sleep stage 1': 'N1',
    'Sleep stage 2': 'N2',
    'Sleep stage 3': 'N3',
    'Sleep stage R': 'R',
    'Sleep stage Unknown': 'UNKNOWN',
}

_RESPIRATORY_CODES = {
    'pointPolySomnographyObstructiveApnea': 'obstructive_apnea',
    'pointPolySomnographyCentralApnea': 'central_apnea',
    'pointPolySomnographyMixedApnea': 'mixed_apnea',
    'pointPolySomnographyHypopnea': 'obstructive_hypopnea',
    'pointPolySomnographyCentralHypopnea': 'central_hypopnea',
    'pointPolySomnographyMixedHypopnea': 'mixed_hypopnea',
}

_ARTIFACT_CODES = {'blockArtefact'}


def _annotation_code(desc: str) -> Optional[str]:
    """'Обструктивное апноэ(pointPolySomnographyObstructiveApnea)' → 'pointPolySomnographyObstructiveApnea'."""
    m = re.search(r'\(([^)]+)\)', str(desc))
    return m.group(1) if m else None


def _stage_code(desc: str) -> Optional[str]:
    s = str(desc)
    for prefix, code in _STAGE_PREFIXES.items():
        if s.startswith(prefix):
            return code
    return None


def read_edf_with_fallback(edf_path) -> Tuple["mne.io.BaseRaw", Optional[str]]:
    """Читает EDF, перебирая кодировки. Возвращает (raw, used_encoding)."""
    import mne
    last_err = None
    for enc in ['utf-8', 'cp1251', 'latin1', 'iso-8859-1', None]:
        try:
            kwargs = dict(preload=True, verbose='ERROR',
                          infer_types=True, stim_channel=None)
            if enc is not None:
                kwargs['encoding'] = enc
            raw = mne.io.read_raw_edf(str(edf_path), **kwargs)
            return raw, enc
        except Exception as e:
            last_err = e
            continue
    raise IOError(f"EDF read failed for {edf_path}: {last_err}")


def build_hypnogram_from_edf(raw, epoch_dur: float = 30.0) -> Optional[dict]:
    """Строит плотный массив стадий по 30-сек эпохам из аннотаций EDF.

    Формат совместим с тем, что хранится в sleep_statistics.hypnogram_data:
        {'epoch_dur': 30.0, 'stages': ['W', 'N1', 'N2', ...]}
    """
    if not hasattr(raw, 'annotations') or not raw.annotations:
        return None

    duration = raw.n_times / raw.info['sfreq']
    n_epochs = int(np.ceil(duration / epoch_dur))
    stages = ['UNKNOWN'] * n_epochs
    matched = 0

    for on, dur, desc in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        if abs(dur - epoch_dur) > 1.0:
            continue
        code = _stage_code(desc)
        if code is None:
            continue
        idx = int(round(on / epoch_dur))
        if 0 <= idx < n_epochs:
            stages[idx] = code
            matched += 1

    if matched == 0:
        return None
    return {'epoch_dur': epoch_dur, 'stages': stages}


def extract_respiratory_events(raw) -> List[Tuple[float, float]]:
    """Список (onset, offset) всех респираторных событий из аннотаций EDF."""
    if not hasattr(raw, 'annotations') or not raw.annotations:
        return []

    events = []
    for on, dur, desc in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        code = _annotation_code(desc)
        if code in _RESPIRATORY_CODES:
            events.append((float(on), float(on + dur)))
    return events


def extract_artifact_regions(raw) -> List[Tuple[float, float]]:
    """Области артефактов (blockArtefact)."""
    if not hasattr(raw, 'annotations') or not raw.annotations:
        return []

    regions = []
    for on, dur, desc in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        code = _annotation_code(desc)
        if code in _ARTIFACT_CODES:
            regions.append((float(on), float(on + dur)))
    return regions


def describe_annotations(raw) -> dict:
    """Диагностика: сколько аннотаций какого типа."""
    counts = {'stages': 0, 'respiratory': 0, 'artifacts': 0, 'other': 0,
              'total': 0}
    if not hasattr(raw, 'annotations') or not raw.annotations:
        return counts

    for desc in raw.annotations.description:
        counts['total'] += 1
        if _stage_code(desc):
            counts['stages'] += 1
        else:
            code = _annotation_code(desc)
            if code in _RESPIRATORY_CODES:
                counts['respiratory'] += 1
            elif code in _ARTIFACT_CODES:
                counts['artifacts'] += 1
            else:
                counts['other'] += 1
    return counts