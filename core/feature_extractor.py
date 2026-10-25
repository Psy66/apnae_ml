"""
Извлечение ЭЭГ-признаков из EDF для скользящих суб-эпох.

Стадии и респираторные события читаются ПРЯМО ИЗ EDF (raw.annotations).
fallback: hypnogram_data и phasic_events из API/БД (если в EDF ничего нет).
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)

try:
    import mne
    mne.set_log_level("ERROR")
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    logger.error("MNE not installed")

from core.config import (
    BANDS, STANDARD_CHANNELS, TARGET_SR, LOWCUT, HIGHCUT,
    USE_AVERAGE_REFERENCE, STAGES_OF_INTEREST, STAGE_NORMALIZE,
    DEFAULT_BUFFER_SEC,
)
from core.labeling import label_subepoch
from core.edf_annotations import (
    read_edf_with_fallback,
    build_hypnogram_from_edf,
    extract_respiratory_events,
    extract_artifact_regions,
    describe_annotations,
)


# ---------------------------------------------------------------- helpers
def _parse_hypnogram_db(raw) -> Optional[dict]:
    """Fallback-парсер JSON из sleep_statistics.hypnogram_data."""
    if not raw:
        return None
    try:
        h = json.loads(raw) if isinstance(raw, str) else raw
        stages = [STAGE_NORMALIZE.get(str(s), "UNKNOWN") for s in h["s"]]
        return {"epoch_dur": float(h["d"]), "stages": stages}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Bad hypnogram from DB: {e}")
        return None


def _majority_stage(start: float, end: float,
                    hypnogram: Optional[dict]) -> Optional[str]:
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
    if target in raw.ch_names:
        return target
    target_u = target.upper()
    for name in raw.ch_names:
        cleaned = (name.upper().replace("EEG", "")
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
    n_epochs_total: int = 0
    n_epochs_n2n3: int = 0
    n_epochs_labeled: int = 0
    stages_source: str = ""
    events_source: str = ""
    stage_counts: dict = field(default_factory=dict)
    n_events_edf: int = 0
    n_events_api: int = 0
    n_artifacts: int = 0
    annotation_counts: dict = field(default_factory=dict)
    encoding: Optional[str] = None


# ---------------------------------------------------------------- extractor
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
                events_from_api: Optional[List[Tuple[float, float]]] = None,
                hypnogram_from_db=None) -> ExtractedStudy:
        if not MNE_AVAILABLE:
            raise RuntimeError("MNE not installed")

        # 1. Читаем EDF
        raw, enc = read_edf_with_fallback(edf_path)
        ann_counts = describe_annotations(raw)
        duration = raw.n_times / raw.info["sfreq"]
        logger.info(
            f"Study {study_id}: EDF loaded (enc={enc}, dur={duration:.0f}s, "
            f"ch={len(raw.ch_names)}), annotations={ann_counts}"
        )

        # 2. Стадии из EDF
        hyp_edf = build_hypnogram_from_edf(raw)
        stage_counts: dict = {}
        stages_source = ""
        if hyp_edf:
            for s in hyp_edf["stages"]:
                stage_counts[s] = stage_counts.get(s, 0) + 1
            has_n2n3 = any(s in ("N2", "N3") for s in stage_counts)
            if has_n2n3:
                hypnogram = hyp_edf
                stages_source = "EDF"
            else:
                hypnogram = hyp_edf
                stages_source = "EDF (no N2/N3)"
        else:
            hypnogram = None
            stages_source = ""

        # Fallback: hypnogram_data из БД
        if not hypnogram or "N2" not in stage_counts and "N3" not in stage_counts:
            if hypnogram_from_db:
                hyp_db = _parse_hypnogram_db(hypnogram_from_db)
                if hyp_db:
                    stage_counts = {}
                    for s in hyp_db["stages"]:
                        stage_counts[s] = stage_counts.get(s, 0) + 1
                    hypnogram = hyp_db
                    stages_source = "DB"

        if not hypnogram or not hypnogram["stages"]:
            raise ValueError(
                f"Study {study_id}: нет стадий ни в EDF, ни в БД. "
                f"Annotations: {ann_counts}"
            )

        logger.info(
            f"Study {study_id}: stages={stages_source}, "
            f"dist={stage_counts}"
        )

        # 3. События из EDF (приоритет)
        events_edf = extract_respiratory_events(raw)
        if events_edf:
            events = events_edf
            events_source = "EDF"
        elif events_from_api:
            events = events_from_api
            events_source = "API"
        else:
            events = []
            events_source = "none"
        logger.info(
            f"Study {study_id}: events={events_source}, "
            f"EDF={len(events_edf)}, API={len(events_from_api or [])}"
        )

        # 4. Артефакты
        artifacts = extract_artifact_regions(raw)
        logger.info(f"Study {study_id}: artifacts={len(artifacts)}")

        # 5. Фильтрация и референс (ПОСЛЕ аннотаций)
        raw = self._prepare(raw)
        sfreq = float(raw.info["sfreq"])
        duration = raw.n_times / sfreq

        # 6. Данные по каналам
        ch_data = {}
        for ch in self.channels:
            actual = _resolve_channel(raw, ch)
            if actual:
                ch_data[ch] = raw.get_data(picks=actual)[0]
            else:
                logger.warning(f"Study {study_id}: ch '{ch}' not found "
                               f"(available: {raw.ch_names[:8]})")
        if not ch_data:
            raise ValueError(
                f"Study {study_id}: no EEG channels. "
                f"Available: {raw.ch_names}"
            )

        # 7. Скользящее окно
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

            lab = label_subepoch(
                ep_start, ep_end, events,
                t_pos=self.t_pos, buffer_sec=self.buffer_sec,
                artifacts=artifacts,
            )

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
            study_id=study_id,
            patient_id=patient_id,
            df=pd.DataFrame(rows),
            n_epochs_total=n_total,
            n_epochs_n2n3=n_n2n3,
            n_epochs_labeled=len(rows),
            stages_source=stages_source,
            events_source=events_source,
            stage_counts=stage_counts,
            n_events_edf=len(events_edf),
            n_events_api=len(events_from_api or []),
            n_artifacts=len(artifacts),
            annotation_counts=ann_counts,
            encoding=enc,
        )

    def _prepare(self, raw):
        return prepare_raw(raw)

    def _channel_features(self, segment: np.ndarray, sfreq: float, ch: str) -> dict:
        seg_uv = segment * 1e6
        out = {
            f"{ch}_mean": float(np.mean(seg_uv)),
            f"{ch}_std":  float(np.std(seg_uv)),
            f"{ch}_min":  float(np.min(seg_uv)),
            f"{ch}_max":  float(np.max(seg_uv)),
            f"{ch}_rms":  float(np.sqrt(np.mean(seg_uv ** 2))),
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

def prepare_raw(raw):
    """Публичная функция подготовки raw: resample + фильтр + референс.
    Используется в FeatureExtractor._prepare и в SignalCache.
    """
    if abs(raw.info["sfreq"] - TARGET_SR) > 1:
        raw.resample(TARGET_SR, verbose="ERROR")

    iir_params = dict(order=4, ftype="butter", output="sos")
    raw.filter(LOWCUT, HIGHCUT,
               method="iir", iir_params=iir_params,
               phase="zero", verbose="ERROR")

    if USE_AVERAGE_REFERENCE:
        eeg = [c for c in (_resolve_channel(raw, ch)
                            for ch in STANDARD_CHANNELS) if c]
        if len(eeg) > 1:
            raw.set_eeg_reference(eeg, projection=False, verbose="ERROR")
    return raw