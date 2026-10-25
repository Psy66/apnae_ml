"""Полный ML-пайплайн: диагностика → split → CV → fit → eval → сохранение.

Особенности:
- Адаптивное число CV-фолдов (не падает при <5 пациентах в train).
- Диагностика распределения positive-эпох по пациентам (train/test).
- Per-patient AUC на test (для понимания, не «утягивает» ли результат
  один пациент).
- Сохранение предсказаний (первые 2000) для последующего анализа.
- run_comparison() — сравнение нормализаций на одном split (для ablation).
- Методологически корректно: split по patient_id, нормализация fit только
  на train.
"""
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


# ---------------------------------------------------------------- config
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


# ---------------------------------------------------------------- helpers
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
        return (df["ahi"].fillna(0) >= 15).astype(int).values
    raise ValueError(f"Unknown target: {target}")


def _compute_shap_importance(model, X_sample, feature_names, top_n=20):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        imp = np.abs(sv).mean(axis=0)
        order = np.argsort(imp)[::-1][:top_n]
        return [{"feature": feature_names[i], "importance": float(imp[i])}
                for i in order]
    except Exception as e:
        logger.debug(f"SHAP failed: {e}")
        return []


def _per_patient_auc(df_test: pd.DataFrame, y_test: np.ndarray,
                     y_pred_proba: np.ndarray) -> list:
    """AUC на каждом test-пациенте отдельно. None если только один класс."""
    out = []
    for pid in sorted(df_test["patient_id"].unique()):
        mask = df_test["patient_id"].values == pid
        y_p = y_test[mask]
        p_p = y_pred_proba[mask]
        n_pos = int(y_p.sum())
        n_tot = int(len(y_p))
        auc_val = None
        note = ""
        if n_tot < 5:
            note = "too few epochs"
        elif len(np.unique(y_p)) < 2:
            note = "single class (все positive или все negative)"
        else:
            try:
                auc_val = float(roc_auc_score(y_p, p_p))
            except Exception as e:
                note = f"auc failed: {e}"
        out.append({
            "patient_id": int(pid),
            "n_epochs": n_tot,
            "n_positive": n_pos,
            "prevalence": round(n_pos / n_tot, 3) if n_tot else 0.0,
            "auc": auc_val,
            "note": note,
        })
    return out


def _diagnose_split(df_train: pd.DataFrame, df_test: pd.DataFrame,
                    progress) -> dict:
    """Распределение positive-эпох по пациентам в train/test."""
    def dist(df):
        rows = []
        for pid, g in df.groupby("patient_id"):
            n = len(g)
            pos = int(g["label"].sum())
            rows.append({
                "patient_id": int(pid),
                "n": n,
                "pos": pos,
                "prev": round(pos / n, 3) if n else 0.0,
            })
        return rows

    train_d = dist(df_train)
    test_d = dist(df_test)

    warnings = []
    if any(r["pos"] == 0 for r in test_d):
        zero = [r["patient_id"] for r in test_d if r["pos"] == 0]
        warnings.append(
            f"⚠ В test {len(zero)} пациент(ов) без positive-эпох: {zero}. "
            f"Их вклад в AUC не определён."
        )
    if any(r["pos"] == r["n"] for r in test_d):
        full = [r["patient_id"] for r in test_d if r["pos"] == r["n"]]
        warnings.append(
            f"⚠ В test {len(full)} пациент(ов) со ВСЕМИ positive-эпохами: {full}. "
            f"Возможно, у них нет тонических эпох N2/N3."
        )
    # Разброс prevalence в train
    if train_d:
        prevs = [r["prev"] for r in train_d]
        if max(prevs) - min(prevs) > 0.7:
            warnings.append(
                f"⚠ Большой разброс prevalence в train: "
                f"{min(prevs):.2f}–{max(prevs):.2f}. Модель может выучить "
                f"«пациента» вместо «апноэ»."
            )

    progress("=== Диагностика train ===")
    for r in train_d:
        progress(f"  pid={r['patient_id']:>4} | "
                 f"{r['pos']:>5}/{r['n']:>5} positive "
                 f"({r['prev']*100:>5.1f}%)")
    progress("=== Диагностика test ===")
    for r in test_d:
        progress(f"  pid={r['patient_id']:>4} | "
                 f"{r['pos']:>5}/{r['n']:>5} positive "
                 f"({r['prev']*100:>5.1f}%)")
    for w in warnings:
        progress(w)

    return {"train": train_d, "test": test_d, "warnings": warnings}


# ---------------------------------------------------------------- main
def run_experiment(df: pd.DataFrame, cfg: ExperimentConfig,
                   experiments_dir: Path,
                   progress_cb=None) -> dict:
    """Единичный эксперимент. Возвращает dict с метриками и диагностикой."""
    def progress(msg):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    df = _prepare(df, cfg)
    if df.empty:
        raise ValueError("После фильтрации не осталось данных")

    feature_cols = identify_feature_columns(df)
    y = _get_target(df, cfg.target)
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"В target только один класс. Проверьте фильтры и {cfg.target}."
        )

    groups = df["patient_id"].values
    patients = np.unique(groups)

    progress(f"Данные: {len(df)} эпох, {len(patients)} пациентов, "
             f"{len(feature_cols)} признаков")
    progress(f"Баланс: {int(y.sum())} positive / {int((1 - y).sum())} negative")

    if len(patients) < 10:
        progress(
            f"⚠ ВНИМАНИЕ: всего {len(patients)} пациентов. "
            f"Для осмысленного ML-анализа нужно ≥20 (пилот), "
            f"≥50 (статья). Результаты ниже — только smoke-test пайплайна."
        )

    # --- Стратификация по пациентам по «уровню» prevalence ---
    pat_label = (df.groupby("patient_id")["label"].mean() > 0.1).astype(int)
    pat_label = pat_label.loc[patients].values

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

    progress(f"Train: {len(df_train)} эпох / {len(train_patients)} пац.")
    progress(f"Test:  {len(df_test)} эпох / {len(test_patients)} пац.")

    diag = _diagnose_split(df_train, df_test, progress)

    # --- Нормализация: fit ТОЛЬКО на train ---
    normalizer = Normalizer(cfg.normalization).fit(df_train, feature_cols)
    df_train_n = normalizer.transform(df_train, feature_cols)
    df_test_n = normalizer.transform(df_test, feature_cols)

    X_train = df_train_n[feature_cols].values
    X_test = df_test_n[feature_cols].values
    groups_train = df_train["patient_id"].values

    # --- Адаптивный CV ---
    n_train_patients = len(np.unique(groups_train))
    n_folds_eff = min(cfg.n_folds, n_train_patients)
    if n_folds_eff < 2:
        progress(f"⚠ В train {n_train_patients} пациент(ов) — CV пропущен")
        n_folds_eff = 0

    results = {
        "config": asdict(cfg),
        "n_train": len(df_train), "n_test": len(df_test),
        "n_train_patients": len(train_patients),
        "n_test_patients": len(test_patients),
        "class_balance": {"pos": int(y.sum()), "neg": int((1 - y).sum())},
        "features": feature_cols,
        "diagnostics": diag,
        "models": {},
        "timestamp": datetime.now().isoformat(),
    }

    for model_name in cfg.models:
        progress(f"→ {model_name}")
        try:
            # --- CV с адаптивным числом фолдов ---
            cv_aucs = []
            if n_folds_eff >= 2:
                if n_folds_eff < cfg.n_folds:
                    progress(
                        f"  CV: {n_folds_eff} фолдов (запрошено {cfg.n_folds}, "
                        f"в train только {n_train_patients} пациентов)"
                    )
                gkf = GroupKFold(n_splits=n_folds_eff)
                for fold_i, (tr, va) in enumerate(
                        gkf.split(X_train, y_train, groups_train)):
                    if len(np.unique(y_train[tr])) < 2:
                        continue
                    if len(np.unique(y_train[va])) < 2:
                        continue
                    m = get_model(model_name, cfg.random_state + fold_i)
                    m.fit(X_train[tr], y_train[tr])
                    if hasattr(m, "predict_proba"):
                        pred = m.predict_proba(X_train[va])[:, 1]
                    else:
                        pred = m.decision_function(X_train[va])
                    cv_aucs.append(float(roc_auc_score(y_train[va], pred)))
                if not cv_aucs:
                    progress("  ⚠ CV: ни один фолд не дал валидных классов")

            # --- Финальная модель на всём train ---
            model = get_model(model_name, cfg.random_state)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            metrics = full_metrics(y_test, y_pred_proba)

            # CV-статистика
            if cv_aucs:
                metrics["cv_auc_mean"] = float(np.mean(cv_aucs))
                metrics["cv_auc_std"] = float(np.std(cv_aucs))
                metrics["cv_aucs"] = cv_aucs
                metrics["cv_n_folds"] = len(cv_aucs)
            else:
                metrics["cv_auc_mean"] = None
                metrics["cv_auc_std"] = None
                metrics["cv_aucs"] = []
                metrics["cv_n_folds"] = 0

            # Per-patient AUC на test
            metrics["per_patient_test"] = _per_patient_auc(
                df_test, y_test, y_pred_proba)

            # SHAP для tree-based
            if model_name in ("rf", "xgb"):
                n_sample = min(1000, len(X_train))
                rng = np.random.default_rng(cfg.random_state)
                idx = rng.choice(len(X_train), n_sample, replace=False)
                shap_imp = _compute_shap_importance(
                    model, X_train[idx], feature_cols)
                if shap_imp:
                    metrics["shap_top_features"] = shap_imp

            # Сохраняем предсказания (первые 2000, чтобы JSON не раздувался)
            cap = min(2000, len(y_test))
            metrics["_y_test_sample"] = y_test[:cap].tolist()
            metrics["_y_pred_sample"] = [round(float(p), 4)
                                         for p in y_pred_proba[:cap]]

            results["models"][model_name] = metrics

            cv_str = (f", CV={metrics['cv_auc_mean']:.3f}"
                      f" ± {metrics['cv_auc_std']:.3f}"
                      if metrics["cv_auc_mean"] is not None else "")
            progress(
                f"  AUC = {metrics['auc']['auc']:.3f} "
                f"[{metrics['auc']['ci_low']:.3f}, "
                f"{metrics['auc']['ci_high']:.3f}]{cv_str}"
            )

            # Краткий комментарий по per-patient
            pa = metrics["per_patient_test"]
            if pa:
                single = [p for p in pa if p["note"]]
                if single:
                    progress(f"  ⚠ {len(single)} из {len(pa)} test-пациентов "
                             f"с проблемами (single class / too few)")

        except Exception as e:
            logger.exception(f"{model_name} failed")
            results["models"][model_name] = {"error": str(e)}
            progress(f"  ✗ ERROR: {e}")

    # --- Сохранение ---
    experiments_dir.mkdir(parents=True, exist_ok=True)
    out = experiments_dir / f"{cfg.name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    progress(f"Сохранено: {out}")
    results["_path"] = str(out)
    return results


# ---------------------------------------------------------------- comparison
def run_comparison(df: pd.DataFrame, base_cfg: ExperimentConfig,
                   experiments_dir: Path,
                   norms=("none", "global", "patient"),
                   progress_cb=None) -> dict:
    """Сравнивает нормализации на ОДНОМ и том же split-е.

    Важно: используется одинаковый random_state, поэтому train/test
    одинаковы для всех нормализаций — сравнение честное.
    """
    def progress(msg):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    progress(f"=== Сравнение нормализаций {list(norms)} на одном split ===")
    comparison = {}
    for norm in norms:
        progress(f"\n--- normalizer: {norm} ---")
        cfg = ExperimentConfig(
            name=f"COMPARE_{norm}_{base_cfg.name}",
            epoch_len=base_cfg.epoch_len,
            models=base_cfg.models,
            normalization=norm,
            target=base_cfg.target,
            test_frac=base_cfg.test_frac,
            n_folds=base_cfg.n_folds,
            overlap_t=base_cfg.overlap_t,
            comorbidity_filters=base_cfg.comorbidity_filters,
            random_state=base_cfg.random_state,
        )
        try:
            res = run_experiment(df, cfg, experiments_dir, progress_cb)
            comparison[norm] = {
                "path": res.get("_path"),
                "models": {
                    m: {
                        "auc": r.get("auc", {}).get("auc"),
                        "ci_low": r.get("auc", {}).get("ci_low"),
                        "ci_high": r.get("auc", {}).get("ci_high"),
                        "cv_auc_mean": r.get("cv_auc_mean"),
                        "error": r.get("error"),
                    }
                    for m, r in res["models"].items()
                },
            }
        except Exception as e:
            logger.exception(f"comparison {norm} failed")
            comparison[norm] = {"error": str(e)}

    # Сохраняем сводку
    summary_path = (experiments_dir /
                    f"COMPARISON_{datetime.now():%Y%m%d_%H%M%S}.json")
    summary_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    progress(f"\nСводка сравнения: {summary_path}")
    return {"summary_path": str(summary_path), "comparison": comparison}


# ---------------------------------------------------------------- io
def load_experiment(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_experiments(experiments_dir: Path) -> list:
    if not Path(experiments_dir).exists():
        return []
    # Исключаем COMPARISON_* из общего списка
    return sorted(
        [p for p in Path(experiments_dir).glob("*.json")
         if not p.name.startswith("COMPARISON_")],
        reverse=True,
    )