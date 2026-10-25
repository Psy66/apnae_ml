"""
Извлечение ЭЭГ-признаков из EDF для скользящих суб-эпох заданной длины.

Признаки (на канал):
  - Статистика: mean, std, min, max, range, rms
  - Абсолютная мощность диапазонов (Welch): delta, theta, alpha, sigma, beta, gamma
  - Относительная мощность тех же диапазонов
  - TBR, DAR, SE50

Оптимизировано для скорости: никаких SampEn, DFA, когерентности.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)

try:
    import mne
    mne.set_log_level("WARNING")
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    logger.warning("MNE not installed")

from core.config import (
    BANDS, STANDARD_CHANNELS, TARGET_SR, LOWCUT, HIGHCUT,
    USE_AVERAGE_REFERENCE, STAGES_OF_INTEREST, STAGE_NORMALIZE,
    DEFAULT_BUFFER_SEC,
)
from core.labeling import label_subepoch


# -----------------------------------------------------------------------------
# Вспомогательные
# -----------------------------------------------------------------------------
def _parse_hypnogram(raw) -> Optional[dict]:
    """JSON из sleep_statistics.hypnogram_data → {'epoch_dur', 'stages'}."""
    if not raw:
        return None
    try:
        h = json.loads(raw) if isinstance(raw, str) else raw
        stages = [STAGE_NORMALIZE.get(str(s), "UNKNOWN") for s in h["s"]]
        return {"epoch_dur": float(h["d"]), "stages": stages}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Bad hypnogram: {e}")
        return None


def _majority_stage(start: float, end: float, hypnogram: Optional[dict]) -> Optional[str]:
    if not hypnogram:
        return None
    d = hypnogram["epoch_dur"]
    stages = hypnogram["stages"]
    counts: Dict[str, float] = {}
    for i, s in enumerate(stages):
        es, ee = i * d, (i + 1) * d
        ov = max(0.0, min(end, ee) - max(start, es))
        if ov > 0:
            counts[s] = counts.get(s, 0.0) + ov
    return max(counts, key=counts.get) if counts else None


def _band_powers(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        out[name] = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
    return out


def _se50(freqs: np.ndarray, psd: np.ndarray) -> float:
    cumsum = np.cumsum(psd)
    total = cumsum[-1] if len(cumsum) > 0 else 0.0
    if total <= 0:
        return 0.0
    idx = np.searchsorted(cumsum, total / 2.0)
    return float(freqs[min(idx, len(freqs) - 1)])


def _resolve_channel(raw, target: str) -> Optional[str]:
    """'F3' → 'EEG F3-A2' и т.п."""
    if target in raw.ch_names:
        return target
    target_u = target.upper()
    for name in raw.ch_names:
        cleaned = (name.upper()
                   .replace("EEG", "")
                   .replace("-A1", "").replace("-A2", "")
                   .replace("-M1", "").replace("-M2", "")
                   .strip())
        if cleaned == target_u or target_u in cleaned.split("-"):
            return name
    return None


@dataclass
class ExtractedStudy:
    study_id: int
    patient_id: int
    df: pd.DataFrame
    n_epochs_total: int
    n_epochs_n2n3: int
    n_epochs_labeled: int


# -----------------------------------------------------------------------------
# Основной экстрактор
# -----------------------------------------------------------------------------
class FeatureExtractor:
    def __init__(self, epoch_length: float, stride: Optional[float] = None,
                 buffer_sec: float = DEFAULT_BUFFER_SEC, t_pos: float = 0.5,
                 channels: Optional[List[str]] = None):
        self.epoch_length = float(epoch_length)
        self.stride = float(stride) if stride else float(epoch_length)
        self.buffer_sec = float(buffer_sec)
        self.t_pos = float(t_pos)
        self.channels = channels or STANDARD_CHANNELS

    def extract(self, edf_path: Path, study_id: int, patient_id: int,
                events: List[Tuple[float, float]],
                hypnogram_raw) -> ExtractedStudy:
        if not MNE_AVAILABLE:
            raise RuntimeError("MNE not installed")

        raw = self._prepare(edf_path)
        hypnogram = _parse_hypnogram(hypnogram_raw)
        sfreq = float(raw.info["sfreq"])
        duration = raw.n_times / sfreq

        # Кэш numpy-массивов по каналам
        ch_data = {}
        for ch in self.channels:
            actual = _resolve_channel(raw, ch)
            if actual:
                ch_data[ch] = raw.get_data(picks=actual)[0]
            else:
                logger.warning(f"Study {study_id}: channel '{ch}' not found "
                               f"(available: {raw.ch_names[:8]}...)")

        if not ch_data:
            raise ValueError(f"Study {study_id}: no usable EEG channels found")

        rows = []
        n_total = n_n2n3 = 0
        t = 0.0
        while t + self.epoch_length <= duration + 1e-6:
            n_total += 1
            ep_start, ep_end = t, t + self.epoch_length

            stage = _majority_stage(ep_start, ep_end, hypnogram)
            if stage not in STAGES_OF_INTEREST:
                t += self.stride
                continue
            n_n2n3 += 1

            lab = label_subepoch(ep_start, ep_end, events,
                                 t_pos=self.t_pos, buffer_sec=self.buffer_sec)

            start_idx = int(ep_start * sfreq)
            end_idx = int(ep_end * sfreq)

            row = {
                "study_id": study_id,
                "patient_id": patient_id,
                "epoch_start": float(ep_start),
                "epoch_len": self.epoch_length,
                "epoch_stage": stage,
                **lab,
            }
            for ch, data in ch_data.items():
                seg = data[start_idx:end_idx]
                if len(seg) < 10:
                    continue
                row.update(self._channel_features(seg, sfreq, ch))
            rows.append(row)
            t += self.stride

        return ExtractedStudy(
            study_id=study_id, patient_id=patient_id,
            df=pd.DataFrame(rows),
            n_epochs_total=n_total, n_epochs_n2n3=n_n2n3,
            n_epochs_labeled=len(rows),
        )

    def _prepare(self, edf_path: Path):
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="WARNING")
        if abs(raw.info["sfreq"] - TARGET_SR) > 1:
            raw.resample(TARGET_SR, verbose="WARNING")
        raw.filter(LOWCUT, HIGHCUT, method="filtfilt", verbose="WARNING")
        if USE_AVERAGE_REFERENCE:
            eeg_channels = [c for c in (_resolve_channel(raw, ch) for ch in self.channels) if c]
            if len(eeg_channels) > 1:
                raw.set_eeg_reference(eeg_channels, projection=False, verbose="WARNING")
        return raw

    def _channel_features(self, segment: np.ndarray, sfreq: float, ch: str) -> dict:
        seg_uv = segment * 1e6
        out = {
            f"{ch}_mean":  float(np.mean(seg_uv)),
            f"{ch}_std":   float(np.std(seg_uv)),
            f"{ch}_min":   float(np.min(seg_uv)),
            f"{ch}_max":   float(np.max(seg_uv)),
            f"{ch}_rms":   float(np.sqrt(np.mean(seg_uv ** 2))),
        }
        out[f"{ch}_range"] = out[f"{ch}_max"] - out[f"{ch}_min"]

        nperseg = min(256, len(seg_uv))
        freqs, psd = signal.welch(seg_uv, fs=sfreq, nperseg=nperseg,
                                  noverlap=nperseg // 2)
        powers = _band_powers(freqs, psd)
        total = sum(powers.values()) or 1e-12

        for band, p in powers.items():
            out[f"{ch}_abs_{band}"] = p
            out[f"{ch}_rel_{band}"] = p / total

        out[f"{ch}_tbr"] = powers["theta"] / (powers["beta"] or 1e-12)
        out[f"{ch}_dar"] = powers["delta"] / (powers["alpha"] or 1e-12)
        out[f"{ch}_se50"] = _se50(freqs, psd)
        return out