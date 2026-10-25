"""HTML-отчёт с графиками, таблицами и черновиком раздела Methods для статьи."""
from __future__ import annotations
import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _roc_plot(m: dict, name: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    roc = m["roc_curve"]
    ax.plot(roc["fpr"], roc["tpr"], "b-",
            label=f"AUC = {m['auc']['auc']:.3f} "
                  f"[{m['auc']['ci_low']:.3f}, {m['auc']['ci_high']:.3f}]")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {name}"); ax.legend(); ax.grid(alpha=0.3)
    b64 = _fig_b64(fig); plt.close(fig); return b64


def _cm_plot(m: dict, thr: float = 0.5) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = m[f"at_{thr}"]
    cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticklabels(["True Neg", "True Pos"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    ax.set_title(f"CM @ {thr}")
    b64 = _fig_b64(fig); plt.close(fig); return b64


def _shap_text(m: dict) -> str:
    shap_top = m.get("shap_top_features", [])
    if not shap_top:
        return ""
    rows = "".join(
        f"<tr><td>{i+1}</td><td><code>{r['feature']}</code></td>"
        f"<td>{r['importance']:.4f}</td></tr>"
        for i, r in enumerate(shap_top[:10])
    )
    return f"""
    <h3>Top-10 features by mean |SHAP|</h3>
    <table><tr><th>#</th><th>Feature</th><th>Importance</th></tr>{rows}</table>
    """


def _results_draft(cfg: dict, exp: dict, best_model: str) -> str:
    """Авто-черновик Results для статьи."""
    m = exp["models"][best_model]
    a, t = m["auc"], m["at_0.5"]
    cv = m["cv_auc_mean"]
    return f"""
<p>We evaluated the feasibility of EEG-based apnea detection on N2/N3 sub-epochs
of <strong>{cfg['epoch_len']} s</strong>. On the test set
(<strong>{exp['n_test']}</strong> epochs from <strong>{exp['n_test_patients']}</strong> held-out patients),
the <strong>{best_model}</strong> classifier achieved
<strong>AUC = {a['auc']:.3f}</strong> (95% CI: {a['ci_low']:.3f}–{a['ci_high']:.3f}),
with sensitivity <strong>{t['sens']*100:.1f}%</strong> and specificity
<strong>{t['spec']*100:.1f}%</strong> at the default decision threshold of 0.5.
Balanced accuracy was {t['balanced_acc']:.3f}, and F1-score was {t['f1']:.3f}.
The 5-fold GroupKFold cross-validation on the training set yielded
mean AUC = {cv:.3f} (SD {m['cv_auc_std']:.3f}).
Patient-specific z-score normalization was applied to eliminate
inter-individual variability, with tonic epochs (no events within ±30 s)
serving as baseline.
"""


def generate_report(experiment: dict, output_path: Path) -> Path:
    cfg = experiment["config"]
    models = experiment["models"]

    rows = []
    for name, m in models.items():
        if "error" in m:
            rows.append(f"<tr><td><b>{name}</b></td>"
                        f"<td colspan='7' style='color:red'>ERROR: {m['error']}</td></tr>")
            continue
        a, t = m["auc"], m["at_0.5"]
        rows.append(
            f"<tr><td><b>{name}</b></td>"
            f"<td>{a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]</td>"
            f"<td>{m['auc_pr']:.3f}</td>"
            f"<td>{m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f}</td>"
            f"<td>{t['sens']:.3f}</td><td>{t['spec']:.3f}</td>"
            f"<td>{t['balanced_acc']:.3f}</td><td>{t['f1']:.3f}</td></tr>")

    valid = [(n, m) for n, m in models.items() if "auc" in m]
    best_name = max(valid, key=lambda kv: kv[1]["auc"]["auc"])[0] if valid else None

    plots_html = ""
    for name, m in valid:
        roc, cm = _roc_plot(m, name), _cm_plot(m, 0.5)
        plots_html += f"""
        <div style="margin:20px 0;padding:15px;background:#f8f9fa;border-radius:6px">
          <h3>{name}</h3>
          <div style="display:flex;gap:20px;flex-wrap:wrap">
            <img src="data:image/png;base64,{roc}" width="420"/>
            <img src="data:image/png;base64,{cm}" width="360"/>
          </div>
          {_shap_text(m)}
        </div>"""

    draft = _results_draft(cfg, experiment, best_name) if best_name else "<p>—</p>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>APNEA-ML Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:20px auto;
     padding:0 20px;color:#333;line-height:1.5}}
h1{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px}}
h2{{color:#34495e;margin-top:28px}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}
th,td{{border:1px solid #bdc3c7;padding:6px 10px;text-align:left;font-size:14px}}
th{{background:#ecf0f1}}
.config{{background:#e8f4f8;padding:15px;border-radius:6px;margin:15px 0}}
.draft{{background:#fffbea;padding:15px;border-left:4px solid #f39c12;margin:15px 0;border-radius:4px}}
.footnote{{color:#7f8c8d;font-size:13px;margin-top:30px;
          border-top:1px solid #ecf0f1;padding-top:10px}}
code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}
</style></head><body>

<h1>APNEA-ML Experiment Report</h1>
<p><strong>Experiment:</strong> {cfg['name']}<br>
<strong>Generated:</strong> {datetime.now():%Y-%m-%d %H:%M:%S}</p>

<h2>Configuration</h2>
<div class="config"><table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Epoch length</td><td>{cfg['epoch_len']} s</td></tr>
<tr><td>Overlap threshold t_pos</td><td>{cfg['overlap_t']}</td></tr>
<tr><td>Normalization</td><td>{cfg['normalization']}</td></tr>
<tr><td>Target</td><td>{cfg['target']}</td></tr>
<tr><td>Test fraction</td><td>{cfg['test_frac']}</td></tr>
<tr><td>CV folds</td><td>{cfg['n_folds']}</td></tr>
<tr><td>Random seed</td><td>{cfg['random_state']}</td></tr>
<tr><td>Comorbidity filters</td><td>{cfg.get('comorbidity_filters') or '—'}</td></tr>
</table></div>

<h2>Dataset</h2>
<p>Train: <b>{experiment['n_train']}</b> epochs / <b>{experiment['n_train_patients']}</b> patients<br>
Test: <b>{experiment['n_test']}</b> epochs / <b>{experiment['n_test_patients']}</b> patients<br>
Class balance: <b>{experiment['class_balance']['pos']}</b> positive /
<b>{experiment['class_balance']['neg']}</b> negative<br>
Features: <b>{len(experiment['features'])}</b></p>

<h2>Results</h2>
<table>
<tr><th>Model</th><th>AUC (95% CI)</th><th>AUC-PR</th><th>CV AUC</th>
    <th>Sens@0.5</th><th>Spec@0.5</th><th>Bal.Acc@0.5</th><th>F1@0.5</th></tr>
{''.join(rows)}
</table>

<h2>Per-model Details</h2>
{plots_html}

<h2>Draft Results section (for paper)</h2>
<div class="draft">{draft}</div>

<h2>Notes for Methods section</h2>
<div class="footnote">
<p>Sub-epochs of <b>{cfg['epoch_len']} s</b> were extracted from N2/N3 stages.
An epoch was labeled <i>positive</i> if at least <b>{int(cfg['overlap_t']*100)}%</b>
of its duration overlapped a respiratory event, <i>negative</i> if no overlap
occurred, and <i>discarded</i> otherwise.</p>
<p>Patient-specific z-score normalization used tonic epochs
(no respiratory events within ±30 s from epoch boundaries) as baseline.
Parameters were fit on the training set only.</p>
<p>Splits were performed at the patient level (<code>GroupShuffleSplit</code>);
no patient contributed to both train and test sets. Cross-validation used
<code>GroupKFold({cfg['n_folds']})</code>. AUC confidence intervals were
computed via bootstrap (1000 iterations).</p>
</div>

</body></html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path