"""Метрики: bootstrap CI для AUC, sens/spec, DeLong test."""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)


def bootstrap_auc(y_true, y_score, n_iter: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs) if aucs else np.array([np.nan])
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ci_low": float(np.nanpercentile(aucs, 2.5)),
        "ci_high": float(np.nanpercentile(aucs, 97.5)),
        "n_bootstraps": int(len(aucs)),
    }


def threshold_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    f1   = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    acc  = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "sens": sens, "spec": spec, "ppv": ppv, "npv": npv,
        "f1": f1, "balanced_acc": (sens + spec) / 2, "accuracy": acc,
    }


def full_metrics(y_true, y_score) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    result = {
        "auc": bootstrap_auc(y_true, y_score),
        "auc_pr": float(average_precision_score(y_true, y_score)),
    }
    for thr in (0.3, 0.5, 0.7):
        result[f"at_{thr}"] = threshold_metrics(y_true, y_score, thr)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    result["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    result["pr_curve"] = {"precision": prec.tolist(), "recall": rec.tolist()}
    return result


def delong_test(y_true, y_score_a, y_score_b) -> dict:
    """DeLong test для двух коррелированных ROC."""
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N)
        T2[J] = T
        return T2

    def fast_delong(preds, m):
        n = preds.shape[1] - m
        k = preds.shape[0]
        tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
        for r in range(k):
            tx[r] = compute_midrank(preds[r, :m])
            ty[r] = compute_midrank(preds[r, m:])
            tz[r] = compute_midrank(preds[r])
        aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / 2 / n
        v01 = (tz[:, :m] - tx) / n
        v10 = 1 - (tz[:, m:] - ty) / m
        return aucs, np.cov(v01) / m + np.cov(v10) / n

    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)
    preds = np.vstack([np.asarray(y_score_a)[order], np.asarray(y_score_b)[order]])
    aucs, cov = fast_delong(preds, int(y_true.sum()))
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(max(var, 1e-12))
    p = 2 * stats.norm.sf(abs(z))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "diff": float(diff), "z": float(z), "p": float(p)}