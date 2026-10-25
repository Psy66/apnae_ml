"""HTML-отчёт по эксперименту.

Формируется по данным `run_experiment()` (ml/pipeline.py).

Содержит:
- Configuration — все параметры эксперимента
- Dataset — train/test, баланс классов
- Diagnostics — распределение по пациентам, предупреждения
- Results — таблица метрик
- Per-model Details — ROC, confusion matrix, SHAP, per-patient AUC
- Draft Results section — черновик английского текста для статьи
- Draft Methods section — черновик английского текста для статьи

Все условные тексты (нормализация, CV) формируются по конфигу —
чтобы отчёт не врал о том, что реально было сделано.
"""
from __future__ import annotations
import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- plotting
def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64


def _roc_plot(m: dict, name: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    roc = m["roc_curve"]
    a = m["auc"]
    ax.plot(roc["fpr"], roc["tpr"], "b-",
            label=f"AUC = {a['auc']:.3f} "
                  f"[{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    b64 = _fig_b64(fig)
    plt.close(fig)
    return b64


def _cm_plot(m: dict, thr: float = 0.5) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = f"at_{thr}"
    if key not in m:
        return ""
    d = m[key]
    cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticklabels(["True Neg", "True Pos"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_title(f"Confusion matrix @ {thr}")
    b64 = _fig_b64(fig)
    plt.close(fig)
    return b64


def _per_patient_plot(m: dict) -> str:
    """Столбчатая диаграмма AUC по test-пациентам (если ≥2 с AUC)."""
    per = m.get("per_patient_test", [])
    valid = [p for p in per if p.get("auc") is not None]
    if len(valid) < 2:
        return ""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"pid {p['patient_id']}\n"
              f"({p['n_positive']}/{p['n_epochs']})" for p in valid]
    values = [p["auc"] for p in valid]
    colors = ["#2ecc71" if v >= 0.7 else
              "#e67e22" if v < 0.55 else
              "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(valid) * 0.8), 3.5))
    bars = ax.bar(range(len(valid)), values, color=colors)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5,
               label="random (0.5)")
    ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5,
               label="clinical threshold (0.7)")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_title("Per-patient AUC on test set")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    b64 = _fig_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------- text blocks
def _shap_text(m: dict) -> str:
    shap_top = m.get("shap_top_features", [])
    if not shap_top:
        return ""
    rows = "".join(
        f"<tr><td>{i + 1}</td><td><code>{r['feature']}</code></td>"
        f"<td>{r['importance']:.4f}</td></tr>"
        for i, r in enumerate(shap_top[:10])
    )
    return f"""
    <h4>Top-10 features by mean |SHAP|</h4>
    <table>
      <tr><th>#</th><th>Feature</th><th>Importance</th></tr>
      {rows}
    </table>
    """


def _normalization_methods_text(norm: str) -> str:
    """Условный текст про нормализацию для раздела Methods."""
    if norm == "patient":
        return (
            "Patient-specific z-score normalization was applied: for each "
            "patient (train and test), the mean μ and standard deviation σ "
            "of every feature were estimated from that patient's own tonic "
            "epochs (epochs with no respiratory events within ±30 s from "
            "the boundaries). This procedure uses only the EEG signal, not "
            "the labels, and mirrors real-world deployment where only the "
            "new patient's own signal is available."
        )
    if norm == "global":
        return (
            "Global z-score normalization was applied: μ and σ of every "
            "feature were estimated on the tonic epochs of the training "
            "set only, then applied unchanged to the test set."
        )
    # none
    return (
        "No feature normalization was applied: raw spectral power values "
        "in μV²/Hz were used as input to the models."
    )


def _normalization_results_text(norm: str) -> str:
    """Короткая фраза про нормализацию для раздела Results."""
    return {
        "patient": "Patient-specific z-score normalization was used.",
        "global": "Global z-score normalization was used.",
        "none": "No feature normalization was used.",
    }.get(norm, "")


def _cv_results_text(m: dict) -> str:
    """Текст про кросс-валидацию для Results."""
    cv = m.get("cv_auc_mean")
    if cv is None:
        return (
            "Cross-validation could not be performed (fewer than 2 patients "
            "in the training set)."
        )
    std = m.get("cv_auc_std", 0.0)
    n_folds = m.get("cv_n_folds", 0)
    return (
        f"GroupKFold cross-validation ({n_folds}-fold) on the training "
        f"set yielded mean AUC = {cv:.3f} (SD {std:.3f})."
    )


def _results_draft(cfg: dict, exp: dict, best_model: str) -> str:
    """Черновик раздела Results для статьи."""
    m = exp["models"][best_model]
    a, t = m["auc"], m["at_0.5"]
    norm = cfg.get("normalization", "patient")
    return f"""
<p>We evaluated the feasibility of EEG-based apnea detection on N2/N3
sub-epochs of <strong>{cfg['epoch_len']} s</strong> (overlap threshold
t_pos = {cfg.get('overlap_t', 0.5):.2f}). On the test set
(<strong>{exp['n_test']}</strong> epochs from
<strong>{exp['n_test_patients']}</strong> held-out patients), the
<strong>{best_model}</strong> classifier achieved
<strong>AUC = {a['auc']:.3f}</strong>
(95% CI: {a['ci_low']:.3f}–{a['ci_high']:.3f}), with sensitivity
<strong>{t['sens'] * 100:.1f}%</strong> and specificity
<strong>{t['spec'] * 100:.1f}%</strong> at the default decision
threshold of 0.5. Balanced accuracy was {t['balanced_acc']:.3f},
and F1-score was {t['f1']:.3f}. {_cv_results_text(m)}
{_normalization_results_text(norm)}</p>
"""


def _methods_draft(cfg: dict, exp: dict) -> str:
    """Черновик раздела Methods для статьи."""
    norm = cfg.get("normalization", "patient")
    return f"""
<p><strong>Data preprocessing.</strong> EEG signals were band-pass
filtered (0.5–45 Hz, zero-phase Butterworth, 4th order) and re-referenced
to the common average. Records were divided into sub-epochs of
<strong>{cfg['epoch_len']} s</strong> from N2/N3 sleep stages.
A sub-epoch was labeled <em>positive</em> if at least
<strong>{int(cfg.get('overlap_t', 0.5) * 100)}%</strong> of its duration
overlapped a respiratory event, <em>negative</em> if no overlap occurred,
and <em>discarded</em> otherwise.</p>

<p><strong>Features.</strong> For each sub-epoch, absolute and relative
spectral power was computed with Welch's method in six frequency bands
(δ 0.5–4 Hz, θ 4–8 Hz, α 8–13 Hz, σ 12–15 Hz, β 15–30 Hz, γ 30–45 Hz)
for six EEG channels (F3, C3, O1, F4, C4, O2). Derived features included
theta/beta ratio (TBR), delta/alpha ratio (DAR), and spectral edge
frequency (SE50). Total feature count: {len(exp.get('features', []))}.</p>

<p><strong>Normalization.</strong> {_normalization_methods_text(norm)}</p>

<p><strong>Modeling and evaluation.</strong> Splits were performed at the
patient level (<code>GroupShuffleSplit</code>, {int(cfg.get('test_frac', 0.2) * 100)}%
test fraction, random_state = {cfg.get('random_state', 42)}); no patient
contributed to both train and test sets. Cross-validation used
<code>GroupKFold({cfg.get('n_folds', 5)})</code>. AUC confidence intervals
were computed via bootstrap (1000 iterations). All experiments were run
with fixed random seed to ensure reproducibility.</p>
"""


# ---------------------------------------------------------------- main
def generate_report(experiment: dict, output_path: Path) -> Path:
    """Формирует HTML-отчёт по результатам эксперимента.

    Возвращает путь к сохранённому файлу.
    """
    cfg = experiment.get("config", {})
    models = experiment.get("models", {})
    diag = experiment.get("diagnostics", {})

    # --- таблица результатов ---
    rows = []
    for name, m in models.items():
        if "error" in m:
            rows.append(
                f"<tr><td><b>{name}</b></td>"
                f"<td colspan='7' style='color:#c0392b'>"
                f"ERROR: {m['error']}</td></tr>"
            )
            continue

        a = m.get("auc")
        if a is None:
            continue
        t = m.get("at_0.5", {})
        cv_str = "—"
        if m.get("cv_auc_mean") is not None:
            cv_str = (f"{m['cv_auc_mean']:.3f} ± "
                      f"{m.get('cv_auc_std', 0):.3f} "
                      f"({m.get('cv_n_folds', 0)}-fold)")

        color = ("#27ae60" if a["auc"] >= 0.85 else
                 "#2980b9" if a["auc"] >= 0.70 else
                 "#e67e22")

        rows.append(
            f"<tr>"
            f"<td><b>{name}</b></td>"
            f"<td style='color:{color};font-weight:bold'>"
            f"{a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]</td>"
            f"<td>{m.get('auc_pr', float('nan')):.3f}</td>"
            f"<td>{cv_str}</td>"
            f"<td>{t.get('sens', 0):.3f}</td>"
            f"<td>{t.get('spec', 0):.3f}</td>"
            f"<td>{t.get('balanced_acc', 0):.3f}</td>"
            f"<td>{t.get('f1', 0):.3f}</td>"
            f"</tr>"
        )

    valid = [(n, m) for n, m in models.items()
             if "auc" in m and "error" not in m]
    best_name = (max(valid, key=lambda kv: kv[1]["auc"]["auc"])[0]
                 if valid else None)

    # --- предупреждения из диагностики ---
    warnings = diag.get("warnings", [])
    warnings_html = ""
    if warnings:
        warnings_html = "<h2>⚠ Diagnostics &amp; warnings</h2><ul>"
        for w in warnings:
            warnings_html += f"<li style='margin:6px 0'>{w}</li>"
        warnings_html += "</ul>"

    # --- распределение по пациентам ---
    patient_dist_html = ""
    train_d = diag.get("train", [])
    test_d = diag.get("test", [])
    if train_d or test_d:
        patient_dist_html = (
            "<h2>Распределение эпох по пациентам</h2>"
            "<table><tr><th>Set</th><th>patient_id</th>"
            "<th>Эпох</th><th>Positive</th><th>Prevalence</th></tr>"
        )
        for r in train_d:
            patient_dist_html += (
                f"<tr><td>train</td><td>{r['patient_id']}</td>"
                f"<td>{r['n']}</td><td>{r['pos']}</td>"
                f"<td>{r['prev'] * 100:.1f}%</td></tr>"
            )
        for r in test_d:
            patient_dist_html += (
                f"<tr><td>test</td><td>{r['patient_id']}</td>"
                f"<td>{r['n']}</td><td>{r['pos']}</td>"
                f"<td>{r['prev'] * 100:.1f}%</td></tr>"
            )
        patient_dist_html += "</table>"

    # --- графики по каждой модели ---
    plots_html = ""
    for name, m in valid:
        roc = _roc_plot(m, name)
        cm = _cm_plot(m, 0.5)
        per_plot = _per_patient_plot(m)

        per_block = ""
        per = m.get("per_patient_test", [])
        if per:
            per_block = (
                "<details style='margin-top:12px'>"
                "<summary style='cursor:pointer;font-weight:bold'>"
                "Per-patient результаты</summary>"
                "<table><tr><th>patient_id</th><th>n_epochs</th>"
                "<th>n_positive</th><th>prevalence</th><th>AUC</th>"
                "<th>note</th></tr>"
            )
            for p in per:
                auc_v = (f"{p['auc']:.3f}"
                         if p["auc"] is not None else "—")
                per_block += (
                    f"<tr><td>{p['patient_id']}</td>"
                    f"<td>{p['n_epochs']}</td>"
                    f"<td>{p['n_positive']}</td>"
                    f"<td>{p['prevalence'] * 100:.1f}%</td>"
                    f"<td>{auc_v}</td>"
                    f"<td>{p.get('note', '') or ''}</td></tr>"
                )
            per_block += "</table></details>"

        plots_html += f"""
        <div class="plot-block">
          <h3>{name}</h3>
          <div class="plot-row">
            <img src="data:image/png;base64,{roc}" width="440"/>
            {"<img src='data:image/png;base64," + cm + "' width='360'/>"
             if cm else ""}
          </div>
          {f'<div class="plot-row"><img src="data:image/png;base64,{per_plot}" '
           f'width="100%" style="max-width:720px"/></div>' if per_plot else ""}
          {per_block}
          {_shap_text(m)}
        </div>
        """

    # --- черновики текстов ---
    draft_results = (_results_draft(cfg, experiment, best_name)
                     if best_name else "<p>—</p>")
    draft_methods = _methods_draft(cfg, experiment)

    # --- финальный HTML ---
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>APNEA-ML Report — {cfg.get('name', '')}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;max-width:1100px;margin:20px auto;
     padding:0 20px;color:#333;line-height:1.55}}
h1{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px}}
h2{{color:#34495e;margin-top:32px;border-bottom:1px solid #ecf0f1;
   padding-bottom:4px}}
h3{{color:#2c3e50;margin-top:20px}}
h4{{color:#7f8c8d;margin:12px 0 6px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;
      font-size:14px}}
th,td{{border:1px solid #bdc3c7;padding:6px 10px;text-align:left}}
th{{background:#ecf0f1;font-weight:600}}
code{{background:#f4f4f4;padding:2px 5px;border-radius:3px;
     font-family:Consolas,monospace;font-size:13px}}
.config{{background:#e8f4f8;padding:14px;border-radius:6px;margin:14px 0}}
.draft{{background:#fffbea;padding:14px;border-left:4px solid #f39c12;
       margin:14px 0;border-radius:4px}}
.draft p{{margin:8px 0}}
.footnote{{color:#7f8c8d;font-size:13px;margin-top:40px;
          border-top:1px solid #ecf0f1;padding-top:12px}}
.plot-block{{margin:24px 0;padding:16px;background:#f8f9fa;
            border-radius:6px}}
.plot-row{{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;
          margin:10px 0}}
.plot-row img{{border-radius:4px}}
details summary{{user-select:none}}
</style></head><body>

<h1>APNEA-ML Experiment Report</h1>
<p><strong>Experiment:</strong> {cfg.get('name', '—')}<br>
<strong>Generated:</strong> {datetime.now():%Y-%m-%d %H:%M:%S}</p>

<h2>Configuration</h2>
<div class="config"><table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Epoch length</td><td>{cfg.get('epoch_len', '—')} s</td></tr>
<tr><td>Overlap threshold (t_pos)</td>
    <td>{cfg.get('overlap_t', '—')}</td></tr>
<tr><td>Normalization</td>
    <td>{cfg.get('normalization', '—')}</td></tr>
<tr><td>Target variable</td>
    <td>{cfg.get('target', '—')}</td></tr>
<tr><td>Test fraction</td>
    <td>{cfg.get('test_frac', '—')}</td></tr>
<tr><td>CV folds</td><td>{cfg.get('n_folds', '—')}</td></tr>
<tr><td>Random seed</td>
    <td>{cfg.get('random_state', '—')}</td></tr>
<tr><td>Comorbidity filters</td>
    <td>{cfg.get('comorbidity_filters') or '—'}</td></tr>
</table></div>

<h2>Dataset</h2>
<table>
<tr><th>Split</th><th>Patients</th><th>Epochs</th></tr>
<tr><td>Train</td><td>{experiment.get('n_train_patients', '—')}</td>
    <td>{experiment.get('n_train', '—')}</td></tr>
<tr><td>Test</td><td>{experiment.get('n_test_patients', '—')}</td>
    <td>{experiment.get('n_test', '—')}</td></tr>
</table>
<p>Class balance (all data): <b>{experiment.get('class_balance', {}).get('pos', 0)}</b>
positive / <b>{experiment.get('class_balance', {}).get('neg', 0)}</b> negative<br>
Features used: <b>{len(experiment.get('features', []))}</b></p>

{warnings_html}
{patient_dist_html}

<h2>Results</h2>
<table>
<tr>
  <th>Model</th>
  <th>AUC (95% CI)</th>
  <th>AUC-PR</th>
  <th>CV AUC</th>
  <th>Sens @ 0.5</th>
  <th>Spec @ 0.5</th>
  <th>Balanced Acc</th>
  <th>F1</th>
</tr>
{''.join(rows)}
</table>

<h2>Per-model Details</h2>
{plots_html}

<h2>Draft for the Results section</h2>
<div class="draft">{draft_results}</div>

<h2>Draft for the Methods section</h2>
<div class="draft">{draft_methods}</div>

<div class="footnote">
<p><strong>Как читать отчёт.</strong> Основной результат — AUC на test set
с 95% доверительным интервалом. Сравнивайте модели по CV AUC, не по test:
если CV сильно ниже test — результат может быть случайностью одного split.
Если нижняя граница CI &lt; 0.5 — модель статистически не лучше случайной.
Смотрите per-patient результаты: большой разброс AUC (0.4–0.9) означает
нестабильность модели.</p>
<p>Черновики Results и Methods автоматически сформированы на основе
параметров эксперимента — можно копировать в статью, но обязательно
перепроверьте числа.</p>
</div>

</body></html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path