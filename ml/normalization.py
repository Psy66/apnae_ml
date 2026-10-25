"""Normalizer + строгий отбор признаков.

Признаки ЭЭГ определяются ЯВНО по префиксам каналов. Всё остальное
(клинические, метаданные, разметка) — игнорируется. Это защита от
случайного попадания в X_train клинических колонок с NaN.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Явный список префиксов ЭЭГ-признаков (по 6 каналам + когерентность)
CHANNEL_PREFIXES = ("F3_", "C3_", "O1_", "F4_", "C4_", "O2_")
COHERENCE_PREFIXES = ("F3C3_", "F3F4_", "F3C4_", "C3C4_",
                      "C3O1_", "F4C4_", "O1O2_", "C3O2_")


def identify_feature_columns(df: pd.DataFrame) -> list:
    """Возвращает только ЭЭГ-признаки (спектральные + когерентность).

    Клинические/метаданные колонки (bmi, age_at_study, neuro_*,
    meds_*, scale_*, lab_*, gender_code, ...) автоматически исключены,
    потому что не начинаются с префиксов каналов.
    """
    out = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.startswith(CHANNEL_PREFIXES) or c.startswith(COHERENCE_PREFIXES):
            out.append(c)
    return out


class Normalizer:
    """'none' | 'global' | 'patient'.

    - none: без изменений.
    - global: z-score по train-тоническим, применяется ко всем одинаково.
    - patient: per-patient z-score. Для КАЖДОГО пациента (train и test)
      μ/σ берутся из его тонических эпох. Если тонических < MIN_TONIC
      — fallback на глобальные μ/σ (для train и test одинаково).
    """

    MIN_TONIC = 20

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

        # patient — для каждого пациента отдельно
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
                with np.errstate(invalid="ignore", divide="ignore"):
                    m = np.nanmean(X[baseline_idx], axis=0)
                    s = np.nanstd(X[baseline_idx], axis=0)
                # Если nanmean/nanstd вернули NaN (все значения NaN)
                # или std = 0 → подменяем глобальными/единицей
                m = np.where(np.isnan(m), g_mean, m)
                s = np.where(np.isnan(s) | (s == 0), 1.0, s)
            else:
                m, s = g_mean, g_std
            X[idx] = (X[idx] - m) / s

        df[feature_cols] = X
        return df