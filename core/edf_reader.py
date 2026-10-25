"""Просмотр сырых ЭЭГ-сигналов из EDF (для образовательной вкладки).

- SignalCache: LRU-кэш сигналов в памяти (~240 МБ на исследование).
- compute_psd / compute_avg_psd: Welch для эпохи и средних по классам.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from core.config import STANDARD_CHANNELS, BANDS
from core.edf_annotations import read_edf_with_fallback
from core.feature_extractor import prepare_raw, _resolve_channel

logger = logging.getLogger(__name__)


class SignalCache:
    """LRU-кэш отфильтрованных сигналов 6 каналов.

    Позволяет кликать эпохи без чтения EDF каждый раз. Держит
    только max_studies исследований (~240 МБ на каждое).
    """

    def __init__(self, max_studies: int = 1):
        self._cache: Dict[int, dict] = {}
        self._order: list = []
        self._max = max_studies

    def load(self, study_id: int, edf_path: Path,
             progress_cb=None) -> dict:
        if study_id in self._cache:
            if study_id in self._order:
                self._order.remove(study_id)
            self._order.append(study_id)
            return self._cache[study_id]

        if progress_cb:
            progress_cb(f"Чтение EDF: {edf_path.name}...")

        raw, enc = read_edf_with_fallback(edf_path)
        raw = prepare_raw(raw)
        sfreq = float(raw.info["sfreq"])

        signals = {}
        for ch in STANDARD_CHANNELS:
            actual = _resolve_channel(raw, ch)
            if actual:
                signals[ch] = raw.get_data(picks=actual)[0].astype(np.float32)

        entry = {
            "signals": signals,
            "sfreq": sfreq,
            "duration": float(raw.n_times / sfreq),
            "channels": list(signals.keys()),
            "encoding": enc,
        }

        while len(self._cache) >= self._max and self._order:
            old = self._order.pop(0)
            self._cache.pop(old, None)

        self._cache[study_id] = entry
        self._order.append(study_id)
        return entry

    def get_epoch(self, study_id: int, epoch_start: float,
                  epoch_len: float, channel: str) -> Optional[np.ndarray]:
        entry = self._cache.get(study_id)
        if not entry or channel not in entry["signals"]:
            return None
        sfreq = entry["sfreq"]
        s = max(0, int(epoch_start * sfreq))
        e = min(len(entry["signals"][channel]),
                int((epoch_start + epoch_len) * sfreq))
        return entry["signals"][channel][s:e]

    def sfreq(self, study_id: int) -> Optional[float]:
        return self._cache.get(study_id, {}).get("sfreq")

    def is_loaded(self, study_id: int) -> bool:
        return study_id in self._cache

    def clear(self):
        self._cache.clear()
        self._order.clear()


signal_cache = SignalCache(max_studies=1)


def compute_psd(segment_uv: np.ndarray, sfreq: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD для сегмента (сигнал уже в мкВ)."""
    if len(segment_uv) < 16:
        return np.array([]), np.array([])
    nperseg = min(256, len(segment_uv))
    freqs, psd = signal.welch(
        segment_uv, fs=sfreq, nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    return freqs, psd


def compute_avg_psd(segments_uv: List[np.ndarray], sfreq: float,
                    progress_cb=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Средний PSD по списку сегментов одного класса."""
    if not segments_uv:
        return None
    psds = []
    freqs = None
    total = len(segments_uv)
    for i, seg in enumerate(segments_uv):
        f, p = compute_psd(seg, sfreq)
        if len(f) > 0:
            freqs = f
            psds.append(p)
        if progress_cb and (i + 1) % 100 == 0:
            progress_cb(f"PSD {i+1}/{total}")
    if not psds:
        return None
    n = min(len(p) for p in psds)
    return freqs[:n], np.mean([p[:n] for p in psds], axis=0)