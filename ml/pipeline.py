"""Полный ML-пайплайн: split → CV → fit → eval → сохранение."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from core.config import DEFAULT_RANDOM_STATE
from ml.metrics import full_metrics
from ml.models import get_model
from ml.normalization import Normalizer, identify_feature_columns

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str
    epoch_len: float
    models: list
    normalization: str
    target: str
    test_frac: float = 0.2
    n_folds: int = 5
    overlap_t: float = 0.5
    comorbidity_filters: dict = field(default_factory=dict)
    random_state: int = DEFAULT_RANDOM_STATE


def _apply_comorbidity_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for col, val in filters.items():
        if col not in df.columns:
            continue
        if val == "only_positive":
            df = df[df[col] == 1]
        elif val == "only_negative":
            df = df[df[col] == 0]
    return df


def _prepare(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    df = df[df["label"].isin([0, 1])].copy()
    if cfg.comorbidity_filters:
        df = _apply_comorbidity_filters(df, cfg.comorbidity_filters)
    return df


def _get_target(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "has_apnea":
        return df["label"].astype(int).values
    if target == "ahi_ge_15":
        if "ahi" not in df.columns:
            raise ValueError("'ahi' column required for target='ahi_ge_15'")
        return (df["ahi"] >= 15).astype(int).values
    raise ValueError(f"Unknown target: {target}")


def _compute_shap_importance(model, X_sample, feature_names, top_n=20):
    """Грубая важность через SHAP. Возвращает [] если недоступно."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        imp = np.abs(sv).mean(axis=0)
        order = np.argsort(imp)[::-1][:top_n]
        return [{"feature": feature_names[i], "importance": float(imp[i])} for i in order]
    except Exception as e:
        logger.debug(f"SHAP failed: {e}")
        return []


def run_experiment(df: pd.DataFrame, cfg: ExperimentConfig,
                   experiments_dir: Path,
                   progress_cb=None) -> dict:
    def progress(msg):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    df = _prepare(df, cfg)
    if df.empty:
        raise ValueError("После фильтрации данных не осталось")

    feature_cols = identify_feature_columns(df)
    y = _get_target(df, cfg.target)
    if len(np.unique(y)) < 2:
        raise ValueError("Только один класс в target")

    groups = df["patient_id"].values
    patients = np.unique(groups)

    progress(f"Данные: {len(df)} эпох, {len(patients)} пациентов, "
             f"{len(feature_cols)} признаков")
    progress(f"Баланс: {y.sum()} positive / {(1 - y).sum()} negative")

    # Стратификация по пациентам: доля positive эпох у пациента
    pat_label = df.groupby("patient_id")["label"].mean() > 0.1
    pat_label = pat_label.loc[patients].astype(int).values

    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_frac,
                                 random_state=cfg.random_state)
    tr_p_idx, te_p_idx = next(splitter.split(patients, pat_label, patients))
    train_patients = set(patients[tr_p_idx])
    test_patients = set(patients[te_p_idx])

    tr_mask = df["patient_id"].isin(train_patients).values
    te_mask = df["patient_id"].isin(test_patients).values

    df_train = df[tr_mask].reset_index(drop=True)
    df_test = df[te_mask].reset_index(drop=True)
    y_train, y_test = y[tr_mask], y[te_mask]

    progress(f"Train: {len(df_train)} эпох ({len(train_patients)} пац.)")
    progress(f"Test:  {len(df_test)} эпох ({len(test_patients)} пац.)")

    normalizer = Normalizer(cfg.normalization).fit(df_train, feature_cols)
    df_train_n = normalizer.transform(df_train, feature_cols)
    df_test_n = normalizer.transform(df_test, feature_cols)

    X_train = df_train_n[feature_cols].values
    X_test = df_test_n[feature_cols].values
    groups_train = df_train["patient_id"].values

    results = {
        "config": asdict(cfg),
        "n_train": len(df_train), "n_test": len(df_test),
        "n_train_patients": len(train_patients),
        "n_test_patients": len(test_patients),
        "class_balance": {"pos": int(y.sum()), "neg": int((1 - y).sum())},
        "features": feature_cols,
        "models": {},
        "timestamp": datetime.now().isoformat(),
    }

    for model_name in cfg.models:
        progress(f"→ {model_name}")
        try:
            # GroupKFold на train
            gkf = GroupKFold(n_splits=cfg.n_folds)
            cv_aucs = []
            for fold_i, (tr, va) in enumerate(gkf.split(X_train, y_train, groups_train)):
                m = get_model(model_name, cfg.random_state + fold_i)
                m.fit(X_train[tr], y_train[tr])
                if hasattr(m, "predict_proba"):
                    pred = m.predict_proba(X_train[va])[:, 1]
                else:
                    pred = m.decision_function(X_train[va])
                cv_aucs.append(float(roc_auc_score(y_train[va], pred)))

            # Финальная модель на всём train
            model = get_model(model_name, cfg.random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]

            metrics = full_metrics(y_test, y_pred)
            metrics["cv_auc_mean"] = float(np.mean(cv_aucs))
            metrics["cv_auc_std"] = float(np.std(cv_aucs))
            metrics["cv_aucs"] = cv_aucs

            # SHAP для tree-based
            if model_name in ("rf", "xgb"):
                n_sample = min(1000, len(X_train))
                idx = np.random.default_rng(cfg.random_state).choice(
                    len(X_train), n_sample, replace=False)
                shap_imp = _compute_shap_importance(
                    model if not hasattr(model, "named_steps") else model,
                    X_train[idx], feature_cols)
                if shap_imp:
                    metrics["shap_top_features"] = shap_imp

            results["models"][model_name] = metrics
            progress(f"  AUC = {metrics['auc']['auc']:.3f} "
                     f"[{metrics['auc']['ci_low']:.3f}, {metrics['auc']['ci_high']:.3f}], "
                     f"CV = {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
        except Exception as e:
            logger.exception(f"{model_name} failed")
            results["models"][model_name] = {"error": str(e)}
            progress(f"  ERROR: {e}")

    experiments_dir.mkdir(parents=True, exist_ok=True)
    out = experiments_dir / f"{cfg.name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    progress(f"Сохранено: {out}")
    results["_path"] = str(out)
    return results


def load_experiment(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_experiments(experiments_dir: Path) -> list:
    if not Path(experiments_dir).exists():
        return []
    return sorted(Path(experiments_dir).glob("*.json"), reverse=True)