"""Normalizer: fit on train, transform train/test без утечки."""
from __future__ import annotations
import numpy as np
import pandas as pd

META_COLS = {
    "study_id", "patient_id", "epoch_start", "epoch_len", "epoch_stage",
    "label", "overlap_fraction", "n_events", "has_event_onset",
    "has_event_offset", "is_tonic", "ahi",
    "cvd_hypertension", "endocrine_diabetes", "som_insomnia", "cvd_ihd",
}


def identify_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


class Normalizer:
    """'none' | 'global' | 'patient' — с fit() на train-наборе."""

    def __init__(self, method: str = "patient"):
        self.method = method
        self._stats = None

    def fit(self, df_train: pd.DataFrame, feature_cols: list) -> "Normalizer":
        if self.method == "none":
            return self

        # Baseline = тонические эпохи (без событий)
        tonic = df_train[df_train.get("is_tonic", 1) == 1] if "is_tonic" in df_train else df_train
        if tonic.empty:
            tonic = df_train

        if self.method == "global":
            self._stats = {
                "mean": tonic[feature_cols].mean(),
                "std": tonic[feature_cols].std().replace(0, 1),
            }
        elif self.method == "patient":
            g = tonic.groupby("patient_id")[feature_cols]
            per_patient = g.agg(["mean", "std"])
            per_patient.columns = [f"{c}_{s}" for c, s in per_patient.columns]
            self._stats = {
                "per_patient": per_patient,
                "global_mean": tonic[feature_cols].mean().values,
                "global_std": tonic[feature_cols].std().replace(0, 1).values,
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        if self.method == "none" or self._stats is None:
            return df.copy()
        df = df.copy()

        if self.method == "global":
            df[feature_cols] = (df[feature_cols] - self._stats["mean"]) / self._stats["std"]
            return df

        # per-patient
        per_patient = self._stats["per_patient"]
        g_mean = self._stats["global_mean"]
        g_std = self._stats["global_std"]

        means = per_patient[[f"{c}_mean" for c in feature_cols]].copy()
        means.columns = feature_cols
        stds = per_patient[[f"{c}_std" for c in feature_cols]].replace(0, 1).copy()
        stds.columns = feature_cols

        X = df[feature_cols].values.copy()
        for pid, idx in df.groupby("patient_id").groups.items():
            if pid in means.index:
                m, s = means.loc[pid].values, stds.loc[pid].values
            else:
                m, s = g_mean, g_std
            X[idx] = (X[idx] - m) / s
        df[feature_cols] = X
        return df