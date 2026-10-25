"""Normalizer: fit on train, transform train/test.

Для patient-z используется трансдуктивный подход: для КАЖДОГО пациента
(включая test) μ/σ считаются по его собственным тоническим эпохам.
Это НЕ утечка меток (используем только сигнал, не label) и корректно
воспроизводит deployment: для нового пациента в клинике тоже есть
только его собственный сигнал.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

META_COLS = {
    "study_id", "patient_id", "epoch_start", "epoch_len", "epoch_stage",
    "label", "overlap_fraction", "n_events", "has_event_onset",
    "has_event_offset", "is_tonic", "artifact_overlap_sec", "ahi",
    "cvd_hypertension", "endocrine_diabetes", "som_insomnia", "cvd_ihd",
}


def identify_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


class Normalizer:
    """'none' | 'global' | 'patient'.

    - none: без изменений.
    - global: z-score по train-тоническим, применяется ко всем одинаково.
    - patient: per-patient z-score. Для КАЖДОГО пациента (train и test)
      μ/σ берутся из его тонических эпох. Если тонических < MIN_TONIC
      — fallback на глобальные μ/σ (для train и test одинаково).
    """

    MIN_TONIC = 20  # минимум тонических эпох для надёжной per-patient нормы

    def __init__(self, method: str = "patient"):
        self.method = method
        self._global_stats = None

    def fit(self, df_train: pd.DataFrame, feature_cols: list) -> "Normalizer":
        if self.method == "none":
            return self

        tonic = (df_train[df_train["is_tonic"] == 1]
                 if "is_tonic" in df_train.columns else df_train)
        if tonic.empty:
            tonic = df_train

        self._global_stats = {
            "mean": tonic[feature_cols].mean().values,
            "std": tonic[feature_cols].std().replace(0, 1).values,
        }
        return self

    def transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        if self.method == "none" or self._global_stats is None:
            return df.copy()
        df = df.copy()

        if self.method == "global":
            df[feature_cols] = (
                    (df[feature_cols].values - self._global_stats["mean"])
                    / self._global_stats["std"]
            )
            return df

        # patient — для КАЖДОГО пациента отдельно
        X = df[feature_cols].values.astype(float)
        g_mean = self._global_stats["mean"]
        g_std = self._global_stats["std"]

        for pid, idx in df.groupby("patient_id").groups.items():
            idx = np.asarray(idx, dtype=int)
            if "is_tonic" in df.columns:
                tonic_mask = df.iloc[idx]["is_tonic"].values == 1
                baseline_idx = idx[tonic_mask]
            else:
                baseline_idx = idx
            if len(baseline_idx) < self.MIN_TONIC:
                baseline_idx = idx

            if len(baseline_idx) >= 2:
                # nanmean / nanstd с "safe" выходом
                m = np.nanmean(X[baseline_idx], axis=0)
                s = np.nanstd(X[baseline_idx], axis=0)
                # Если mean/std получились NaN (все значения были NaN),
                # или std = 0 — подменяем глобальными
                m = np.where(np.isnan(m), g_mean, m)
                s = np.where(np.isnan(s) | (s == 0), 1.0, s)
            else:
                m, s = g_mean, g_std
            X[idx] = (X[idx] - m) / s

        df[feature_cols] = X
        return df