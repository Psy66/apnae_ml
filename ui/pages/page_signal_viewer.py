"""Просмотр сырых ЭЭГ-сигналов — красивый визуальный контроль.

Возможности:
- Список эпох исследования с фильтром по меткам.
- Клик по эпохе → карточка метаданных + 3 графика:
  * Waveform всех 6 каналов (offset-plot, SD-нормализация)
  * PSD эпохи (среднее по 6 каналам, заштрихованы бэнды)
  * Bar chart абсолютных мощностей по бэндам
- Автозагрузка сигналов при первом клике.
- Кнопка средних PSD: positive vs negative по всем эпохам.

Технические решения:
- matplotlib.figure.Figure напрямую (без pyplot) — async-safe.
- Рендер через SVG base64 — в 5–10 раз меньше PNG для line-plot.
  Waveform из 12 000 точек в PNG весил 300+ KB → socket.io
  терял апдейт. SVG ~30–50 KB, проходит.
- Downsample waveform до ~1000 точек на канал — визуально
  неотличимо, файл ещё меньше.
- Fallback на PNG dpi=50, если SVG не получился.
- Рендер через @ui.refreshable — гарантированный ui.context.
- Все ошибки — в UI + в лог, никаких «молча ничего».
- last_loaded["sid"] — не перезагружает эпохи при mount.
"""
import asyncio
import base64
import io
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from nicegui import ui

from ui.state import state
from ui.components import help_expansion, section_header, warning_banner
from ui.logger import SafeLog
from ui.help_content import HELP_SIGNAL_VIEWER
from core.edf_reader import signal_cache, compute_psd
from core.config import STANDARD_CHANNELS, BANDS


BAND_COLORS = {
    "delta": "#4a90d9", "theta": "#50b86c", "alpha": "#f5a623",
    "sigma": "#9b59b6", "beta": "#e74c3c", "gamma": "#8e44ad",
}
BAND_LABELS = {
    "delta": "δ 0.5–4", "theta": "θ 4–8", "alpha": "α 8–13",
    "sigma": "σ 12–15", "beta": "β 15–30", "gamma": "γ 30–45",
}
CH_COLORS = {
    "F3": "#1f77b4", "C3": "#2ca02c", "O1": "#ff7f0e",
    "F4": "#9467bd", "C4": "#d62728", "O2": "#17becf",
}
LABEL_EMOJI = {1: "✅", 0: "❌", -1: "⚠️"}
LABEL_ASCII = {1: "POSITIVE", 0: "NEGATIVE", -1: "DISCARD"}
LABEL_BG = {1: "#e8f7ed", 0: "#fdecea", -1: "#f0f0f0"}
LABEL_FG = {1: "#1e7e34", 0: "#a71d2a", -1: "#555555"}

# Downsample: 2000 сэмплов × 6 каналов = 12 000 точек в PNG → 300+ KB.
# При 1000 точек/канал и SVG — ~40–60 KB. Для глаз разницы нет.
WAVEFORM_MAX_POINTS = 1000


# ---------------------------------------------------------------- helpers
def _fig_to_svg_b64(fig: Figure) -> str:
    """Figure → base64 SVG. Вектор: масштабируется без размытия,
    в разы меньше PNG для line-plot'ов."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fig_to_png_b64(fig: Figure, dpi: int = 50) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render_fig(fig: Figure, log: SafeLog | None = None) -> None:
    """Отрисовать Figure в текущем NiceGUI-контексте.

    Приоритет: SVG (маленький) → PNG dpi=50 (совсем маленький) →
    явная ошибка в UI. Вызывать только из sync-функции в UI-контексте
    (например, из @ui.refreshable).
    """
    # --- SVG ---
    try:
        b64 = _fig_to_svg_b64(fig)
        size_kb = len(b64) // 1024
        ui.image(f"data:image/svg+xml;base64,{b64}").classes(
            "w-full bg-white rounded shadow-sm")
        if log:
            log.push(f"  ✓ отрисовано SVG ({size_kb} KB)")
        return
    except Exception as e:
        if log:
            log.push(f"  ✗ SVG failed: {e} — пробую PNG")

    # --- PNG fallback ---
    try:
        b64 = _fig_to_png_b64(fig, dpi=50)
        size_kb = len(b64) // 1024
        ui.image(f"data:image/png;base64,{b64}").classes(
            "w-full bg-white rounded shadow-sm")
        if log:
            log.push(f"  ✓ отрисовано PNG dpi=50 ({size_kb} KB)")
        return
    except Exception as e:
        if log:
            log.push(f"  ✗ PNG fallback failed: {e}")

    ui.label(f"⚠ Не удалось отрисовать график").classes(
        "text-red-600 text-sm")


def _downsample(arr: np.ndarray, max_points: int = WAVEFORM_MAX_POINTS):
    """Прореживание сигнала для отображения. Возвращает (arr, step)."""
    n = len(arr)
    if n <= max_points:
        return arr, 1
    step = int(np.ceil(n / max_points))
    return arr[::step], step


def _style_ax(ax, title: str = "", xlabel: str = "",
              ylabel: str = "") -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, alpha=0.25, linestyle=":", color="#888888")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#888888")
    if title:
        ax.set_title(title, fontsize=10, pad=8, color="#222222")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#444444")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#444444")
    ax.tick_params(labelsize=8, colors="#666666")


def _collect_psd_mean(sid, ts, epoch_len, sfreq):
    psds = []
    freqs = None
    for ch in STANDARD_CHANNELS:
        seg = signal_cache.get_epoch(sid, ts, epoch_len, ch)
        if seg is None or len(seg) < 16:
            continue
        f, p = compute_psd(seg * 1e6, sfreq)
        if len(f) == 0:
            continue
        freqs = f
        psds.append(p)
    if not psds or freqs is None:
        return None, None, 0
    n = min(len(p) for p in psds)
    psd_mean = np.mean([p[:n] for p in psds], axis=0)
    return freqs[:n], psd_mean, len(psds)


# ---------------------------------------------------------------- figures
def _build_waveform_fig(sid, ts, epoch_len, sfreq, mark, ov, lbl,
                        log: SafeLog | None = None) -> Figure:
    """Offset-plot 6 каналов. Downsample до WAVEFORM_MAX_POINTS,
    чтобы SVG был маленьким."""
    fig = Figure(figsize=(9, 3.4), facecolor=LABEL_BG.get(lbl, "#fafafa"))
    ax = fig.add_subplot(111)

    n_drawn = 0
    drawn_channels = []
    for i, ch in enumerate(STANDARD_CHANNELS):
        seg = signal_cache.get_epoch(sid, ts, epoch_len, ch)
        if seg is None or len(seg) < 16:
            continue

        seg_uv, step = _downsample(seg * 1e6, WAVEFORM_MAX_POINTS)
        sd = float(np.std(seg_uv))
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        y = (i * 3.0) + seg_uv / sd
        t = np.arange(len(seg_uv)) * step / sfreq + ts

        ax.plot(t, y, linewidth=0.7, color=CH_COLORS.get(ch, "#333"),
                alpha=0.95, label=ch)
        ax.axhline(y=i * 3.0, color="#aaaaaa", alpha=0.3, linewidth=0.5)
        n_drawn += 1
        drawn_channels.append((i, ch))

    if n_drawn == 0:
        n_cached = 0
        try:
            entry = signal_cache._cache.get(sid)
            if entry:
                n_cached = len(entry["signals"])
        except Exception:
            pass
        raise ValueError(
            f"Нет данных ни по одному каналу из "
            f"{STANDARD_CHANNELS}. В кэше: {n_cached} каналов. "
            f"Возможно, в EDF другие имена каналов."
        )

    ax.set_yticks([i * 3.0 for i, _ in drawn_channels])
    ax.set_yticklabels([ch for _, ch in drawn_channels], fontsize=9)
    _style_ax(
        ax,
        title=(f"Эпоха {ts:.0f}–{ts + epoch_len:.0f} с  |  {mark}  |  "
               f"overlap={ov:.2f}  |  {n_drawn} каналов "
               f"(нормированы на SD)"),
        xlabel="Время, сек",
        ylabel="Каналы",
    )
    ax.legend(loc="upper right", fontsize=8, ncol=6,
              frameon=True, framealpha=0.85, columnspacing=1.0,
              handlelength=1.2)
    fig.tight_layout()
    return fig


def _build_psd_fig(sid, ts, epoch_len, sfreq, mark) -> Figure:
    freqs, psd_mean, n_ch = _collect_psd_mean(sid, ts, epoch_len, sfreq)
    if freqs is None:
        raise ValueError("Не удалось посчитать PSD ни по одному каналу")

    fig = Figure(figsize=(9, 2.8), facecolor="#fafafa")
    ax = fig.add_subplot(111)
    for name, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.10, color=BAND_COLORS[name])
    pos_mask = psd_mean > 0
    if pos_mask.any():
        ax.fill_between(freqs[pos_mask], psd_mean[pos_mask] / 10,
                        psd_mean[pos_mask], alpha=0.15, color="#2c3e50")
    ax.semilogy(freqs, psd_mean, color="#2c3e50", linewidth=1.4)
    ax.set_xlim(0.5, 45)

    psd_pos = psd_mean[psd_mean > 0]
    if len(psd_pos) > 0:
        ymax = float(psd_pos.max())
        ymin = float(psd_pos.min())
        y_range = np.log10(ymax) - np.log10(ymin)
        y_text = 10 ** (np.log10(ymin) + y_range * 0.92)
        for name, (lo, hi) in BANDS.items():
            ax.text((lo + hi) / 2, y_text, BAND_LABELS[name],
                    ha="center", fontsize=8,
                    color=BAND_COLORS[name], alpha=0.9,
                    fontweight="bold")

    _style_ax(
        ax,
        title=(f"PSD эпохи t={ts:.0f} с  |  {mark}  |  "
               f"среднее по {n_ch} каналам"),
        xlabel="Частота, Гц",
        ylabel="PSD, мкВ²/Гц (log)",
    )
    fig.tight_layout()
    return fig


def _build_band_power_fig(sid, ts, epoch_len, sfreq, mark) -> Figure:
    freqs, psd_mean, n_ch = _collect_psd_mean(sid, ts, epoch_len, sfreq)
    if freqs is None:
        raise ValueError("Не удалось посчитать мощности по бэндам")

    powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        powers[name] = (float(np.trapezoid(psd_mean[mask], freqs[mask]))
                        if mask.any() else 0.0)

    fig = Figure(figsize=(9, 2.4), facecolor="#fafafa")
    ax = fig.add_subplot(111)
    names = list(powers.keys())
    values = [max(powers[n], 1e-6) for n in names]
    colors = [BAND_COLORS[n] for n in names]
    bars = ax.bar(names, values, color=colors, alpha=0.85,
                  edgecolor="#333333", linewidth=0.6)
    ax.set_yscale("log")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.15,
                f"{v:.2g}", ha="center", fontsize=8, color="#222222")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([BAND_LABELS[n] for n in names], fontsize=8)

    _style_ax(
        ax,
        title=(f"Абсолютная мощность по бэндам  |  эпоха t={ts:.0f} с  |  "
               f"{mark}  |  среднее по {n_ch} каналам"),
        ylabel="мкВ² (log)",
    )
    fig.tight_layout()
    return fig


def _build_avg_fig(f, p_pos, p_neg, n_pos, n_neg) -> Figure:
    fig = Figure(figsize=(9, 3.0), facecolor="#fafafa")
    ax = fig.add_subplot(111)
    for name, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.08, color=BAND_COLORS[name])
    ax.fill_between(f, p_neg / 10, p_neg, alpha=0.10, color="#e74c3c")
    ax.fill_between(f, p_pos / 10, p_pos, alpha=0.10, color="#2ecc71")
    ax.semilogy(f, p_pos, color="#2ecc71", linewidth=1.8,
                label=f"positive (n={n_pos} эпох)")
    ax.semilogy(f, p_neg, color="#e74c3c", linewidth=1.8,
                label=f"negative (n={n_neg} эпох)")
    ax.set_xlim(0.5, 45)
    _style_ax(
        ax,
        title="Средние спектры: positive vs negative "
              "(усреднено по 6 каналам)",
        xlabel="Частота, Гц",
        ylabel="Средний PSD, мкВ²/Гц (log)",
    )
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()
    return fig


def _build_diff_fig(f, p_pos, p_neg) -> Figure:
    fig = Figure(figsize=(9, 2.4), facecolor="#fafafa")
    ax = fig.add_subplot(111)
    ratio = np.log10(p_pos / np.maximum(p_neg, 1e-12))
    ax.axhline(y=0, color="#888888", linestyle="--", alpha=0.7)
    for name, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.08, color=BAND_COLORS[name])
    ax.plot(f, ratio, color="#8e44ad", linewidth=1.5)
    ax.fill_between(f, 0, ratio, where=ratio > 0,
                    color="#2ecc71", alpha=0.25, label="выше при апноэ")
    ax.fill_between(f, 0, ratio, where=ratio < 0,
                    color="#e74c3c", alpha=0.25, label="ниже при апноэ")
    ax.set_xlim(0.5, 45)
    _style_ax(
        ax,
        title="Разница спектров  (>0 = выше при апноэ, <0 = ниже)",
        xlabel="Частота, Гц",
        ylabel="log10(pos / neg)",
    )
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- page
def page_signal_viewer():
    from ui.app import create_layout
    create_layout()

    with ui.column().classes("p-6 w-full"):
        section_header(
            "🔬 Просмотр сигналов",
            "Сырые волновые формы (6 каналов) и спектры эпох "
            "«апноэ» vs «без апноэ»",
        )
        help_expansion(
            "📖 Как читать сигналы и спектры",
            HELP_SIGNAL_VIEWER, icon="help",
        )

        if not state.api:
            warning_banner(
                "Сначала настройте API на странице ⚙ Settings",
                level="error")
            return

        available = state.feature_store.list_available()
        study_ids = sorted({item[0] for item in available})
        if not study_ids:
            warning_banner(
                "Нет извлечённых данных. Сначала выполните шаги 1 и 2.",
                level="warning")
            return

        current = {
            "epochs_df": None,
            "epoch_len": 10.0,
            "sfreq": None,
            "selected_row": None,
            "loading": False,
        }
        last_loaded = {"sid": None}
        click_lock = {"busy": False}

        with ui.row().classes("w-full gap-4 items-start"):
            with ui.column().classes("w-80 gap-2 shrink-0"):
                study_sel = ui.select(
                    study_ids, value=study_ids[0],
                    label="study_id",
                ).classes("w-full")
                status_lbl = ui.label("Сигналы не загружены").classes(
                    "text-sm text-gray-600 italic")

                with ui.row().classes("gap-2 mt-1"):
                    cb_pos = ui.checkbox("✅ positive", value=True)
                    cb_neg = ui.checkbox("❌ negative", value=True)
                    cb_disc = ui.checkbox("⚠️ discard", value=False)

                ui.label("Эпохи (клик — показать):").classes(
                    "font-bold text-sm mt-2")
                with ui.column().classes(
                        "w-full overflow-auto border rounded p-2 "
                        "bg-gray-50"
                ).style("max-height: 420px"):
                    epochs_area_host = ui.column().classes("w-full")

                ui.separator()
                ui.label("Средние спектры:").classes(
                    "font-bold text-sm mt-2")
                avg_btn = ui.button(
                    "📊 Средние PSD: positive vs negative",
                    color="secondary",
                ).classes("w-full")
                ui.label(
                    "Усреднение по 6 каналам и по всем эпохам класса."
                ).classes("text-xs text-gray-500")

            with ui.column().classes("flex-1 min-w-0"):
                meta_card = ui.card().classes(
                    "w-full bg-blue-50 border-l-4 border-blue-400 "
                    "shadow-sm")
                with meta_card:
                    meta_content = ui.column().classes("w-full")

                with ui.column().classes("w-full mt-3"):
                    plots_content_host = ui.column().classes(
                        "w-full gap-3")

        log = SafeLog(ui.log(max_lines=300).classes(
            "w-full h-32 font-mono text-xs mt-4"))

        # ---------------- метаданные эпохи ----------------
        @ui.refreshable
        def meta_content_ref():
            meta_content.clear()
            row = current["selected_row"]
            with meta_content:
                if row is None:
                    ui.label("Эпоха не выбрана").classes(
                        "text-sm text-gray-500 italic")
                    return
                lbl = int(row["label"])
                emoji = LABEL_EMOJI.get(lbl, "?")
                name = LABEL_ASCII.get(lbl, "?")
                stage = str(row.get("epoch_stage", "?"))
                ts = float(row["epoch_start"])
                elen = float(row.get("epoch_len",
                                     current["epoch_len"]))
                ov = float(row.get("overlap_fraction", 0.0))
                tonic = int(row.get("is_tonic", 0))
                study = int(row["study_id"])

                with ui.row().classes("gap-4 items-center flex-wrap"):
                    ui.label(f"{emoji} {name}").classes(
                        "font-bold text-lg px-3 py-1 rounded "
                        f"text-white").style(
                        f"background:{LABEL_FG.get(lbl, '#555')}")
                    ui.label(f"study_id={study}").classes(
                        "text-sm font-mono text-gray-700")
                    ui.label(f"стадия {stage}").classes(
                        "text-sm font-mono text-gray-700")
                    ui.label(f"t={ts:.0f}–{ts + elen:.0f} с").classes(
                        "text-sm font-mono text-gray-700")
                    ui.label(f"overlap={ov:.2f}").classes(
                        "text-sm font-mono text-gray-700")
                    ui.label(
                        f"tonic={'да' if tonic else 'нет'}"
                    ).classes("text-sm font-mono text-gray-700")

        # ---------------- графики ----------------
        @ui.refreshable
        def plots_content():
            plots_content_host.clear()
            with plots_content_host:
                row = current["selected_row"]
                if row is None:
                    ui.label(
                        "👈 Выберите эпоху слева — здесь появятся "
                        "волновая форма и спектр."
                    ).classes("text-gray-500 italic p-4")
                    return
                if current["loading"]:
                    with ui.row().classes("items-center gap-3 p-6"):
                        ui.spinner(size="lg", color="primary")
                        ui.label("Чтение EDF (5–15 сек)...").classes(
                            "text-gray-600")
                    return

                sid = int(row["study_id"])
                ts = float(row["epoch_start"])
                elen = float(row.get("epoch_len",
                                     current["epoch_len"]))
                lbl = int(row["label"])
                mark = LABEL_ASCII.get(lbl, "?")
                ov = float(row.get("overlap_fraction", 0.0))

                sfreq = current["sfreq"]
                if sfreq is None:
                    ui.label(
                        "⚠ sfreq не определён — сигналы не загружены"
                    ).classes("text-red-600")
                    return

                # 1) Waveform
                try:
                    _render_fig(
                        _build_waveform_fig(
                            sid, ts, elen, sfreq, mark, ov, lbl, log),
                        log)
                except Exception as e:
                    log.push(f"  ✗ waveform: {e}")
                    ui.label(
                        f"⚠ Ошибка waveform: {e}"
                    ).classes("text-red-600 text-sm p-2 "
                              "whitespace-pre-wrap")

                # 2) PSD
                try:
                    _render_fig(
                        _build_psd_fig(sid, ts, elen, sfreq, mark),
                        log)
                except Exception as e:
                    log.push(f"  ✗ PSD: {e}")
                    ui.label(
                        f"⚠ Ошибка PSD: {e}"
                    ).classes("text-red-600 text-sm p-2")

                # 3) Bar chart мощностей
                try:
                    _render_fig(
                        _build_band_power_fig(
                            sid, ts, elen, sfreq, mark),
                        log)
                except Exception as e:
                    log.push(f"  ✗ band power: {e}")
                    ui.label(
                        f"⚠ Ошибка band power: {e}"
                    ).classes("text-red-600 text-sm p-2")

        # ---------------- список эпох ----------------
        @ui.refreshable
        def epochs_list():
            df = current["epochs_df"]
            if df is None:
                ui.label("Загрузка эпох...").classes(
                    "text-sm text-gray-500 italic")
                return
            if df.empty:
                ui.label("DataFrame пуст").classes(
                    "text-sm text-red-600")
                return
            wanted = []
            if cb_pos.value:
                wanted.append(1)
            if cb_neg.value:
                wanted.append(0)
            if cb_disc.value:
                wanted.append(-1)
            if not wanted:
                ui.label("Все метки отключены").classes(
                    "text-sm text-gray-500")
                return
            df_show = df[df["label"].isin(wanted)]
            if df_show.empty:
                ui.label("Нет эпох с выбранными метками").classes(
                    "text-sm text-gray-500")
                return

            for _, row in df_show.head(300).iterrows():
                lbl = int(row["label"])
                mark = LABEL_EMOJI.get(lbl, "?")
                ts = float(row["epoch_start"])
                ov = float(row.get("overlap_fraction", 0.0))

                async def _on_click(r=row):
                    await show_epoch(r)

                ui.button(
                    f"{mark}  t={ts:.0f}s  |  ov={ov:.2f}",
                    on_click=_on_click,
                ).props("flat dense align=left no-caps").classes(
                    "w-full font-mono text-xs text-left")
            if len(df_show) > 300:
                ui.label(
                    f"(показаны первые 300 из {len(df_show)})"
                ).classes("text-xs text-gray-500")

        with epochs_area_host:
            epochs_list()
        meta_content_ref()
        plots_content()

        # ---------------- утилиты ----------------
        async def _ensure_studies() -> bool:
            if state.studies:
                return True
            status_lbl.set_text("Загрузка списка исследований...")
            try:
                state.studies = await asyncio.to_thread(
                    state.api.get_studies)
                log.push(f"Загружено исследований: {len(state.studies)}")
                return True
            except Exception as e:
                log.push(f"Ошибка загрузки списка: {e}")
                return False

        def _get_edf_path(study_id: int):
            study = next(
                (s for s in state.studies if s["study_id"] == study_id),
                None)
            checksum = (study or {}).get("file_checksum", "") or ""
            p = state.edf_cache.get_any(study_id, checksum)
            if p:
                return p
            return state.edf_cache.get_by_study(study_id)

        def _meta_for_study(sid: int):
            matches = [item for item in available if item[0] == sid]
            if not matches:
                return None
            chosen = max(matches, key=lambda x: x[1])
            return float(chosen[1]), float(chosen[2]), float(chosen[3])

        async def _ensure_signals_loaded(sid: int) -> bool:
            if signal_cache.sfreq(sid) is not None:
                return True
            if not await _ensure_studies():
                return False
            edf = _get_edf_path(sid)
            if edf is None:
                ui.notify(
                    f"EDF для study_id={sid} не найден. "
                    f"Скачайте на вкладке «1. Download».",
                    color="negative")
                log.push(f"EDF для {sid} не найден")
                return False
            status_lbl.set_text("Чтение EDF (5–15 сек)...")
            log.push(f"Чтение EDF: {edf.name}")
            try:
                entry = await asyncio.to_thread(
                    signal_cache.load, sid, edf,
                    lambda m: log.push(m))
                current["sfreq"] = entry["sfreq"]
                ch_names = list(entry["channels"])
                status_lbl.set_text(
                    f"✓ {entry['duration']:.0f} с, "
                    f"каналов: {len(ch_names)} {ch_names}, "
                    f"sfreq={entry['sfreq']:.0f} Гц")
                log.push(f"study {sid}: сигналы в памяти")
                if not ch_names:
                    ui.notify(
                        "В EDF не найдены стандартные каналы "
                        "(F3/C3/O1/F4/C4/O2). Смотрите лог.",
                        color="warning")
                return True
            except Exception as e:
                status_lbl.set_text(f"Ошибка: {e}")
                log.push(f"Ошибка чтения EDF: {e}")
                log.push(traceback.format_exc())
                return False

        async def _load_epochs():
            sid = int(study_sel.value)
            if last_loaded["sid"] == sid and current["epochs_df"] is not None:
                return
            meta = _meta_for_study(sid)
            if meta is None:
                ui.notify(
                    "Нет данных признаков для этого исследования",
                    color="warning")
                return
            epoch_len, t_pos, buffer_sec = meta
            current["epoch_len"] = epoch_len
            last_loaded["sid"] = sid
            log.push(f"_load_epochs: загрузка sid={sid}...")

            def _load():
                return state.feature_store.load_all(
                    epoch_len, study_ids=[sid],
                    t_pos=t_pos, buffer_sec=buffer_sec)

            try:
                df = await asyncio.to_thread(_load)
            except Exception as e:
                last_loaded["sid"] = None
                log.push(f"Ошибка загрузки признаков: {e}")
                return
            if df.empty:
                last_loaded["sid"] = None
                ui.notify("Нет эпох для этого исследования",
                          color="warning")
                return
            current["epochs_df"] = df.reset_index(drop=True)
            n_pos = int((df["label"] == 1).sum())
            n_neg = int((df["label"] == 0).sum())
            log.push(
                f"study {sid}: {len(df)} эпох, "
                f"positive={n_pos}, negative={n_neg}")
            epochs_list.refresh()

        # ---------------- клик ----------------
        async def show_epoch(row):
            if click_lock["busy"]:
                return
            click_lock["busy"] = True
            try:
                sid = int(study_sel.value)
                current["selected_row"] = row
                current["loading"] = True
                meta_content_ref.refresh()
                plots_content.refresh()

                log.push(
                    f"--- show_epoch: sid={sid}, "
                    f"ts={float(row['epoch_start']):.1f} ---")

                ok = await _ensure_signals_loaded(sid)
                current["loading"] = False
                if not ok:
                    plots_content.refresh()
                    return
                plots_content.refresh()
            finally:
                click_lock["busy"] = False

        # ---------------- средние PSD ----------------
        async def do_avg_psd():
            df = current["epochs_df"]
            sid = int(study_sel.value)
            if df is None or df.empty:
                ui.notify("Список эпох ещё не загружен",
                          color="warning")
                return
            if not await _ensure_signals_loaded(sid):
                return
            sfreq = current["sfreq"]
            if sfreq is None:
                return
            epoch_len = float(current["epoch_len"])
            avg_btn.disable()
            status_lbl.set_text("Считаем средние PSD...")
            log.push("--- avg PSD: старт ---")

            def collect_avg(label: int):
                ts_list = df[df["label"] == label]["epoch_start"].values
                all_psds = []
                freqs = None
                n_used = 0
                for ts in ts_list:
                    f, p, _ = _collect_psd_mean(
                        sid, float(ts), epoch_len, sfreq)
                    if f is None:
                        continue
                    if freqs is None:
                        freqs = f
                    all_psds.append(p)
                    n_used += 1
                if not all_psds or freqs is None:
                    return None, None, 0
                n = min(len(p) for p in all_psds)
                return (freqs[:n],
                        np.mean([p[:n] for p in all_psds], axis=0),
                        n_used)

            try:
                f_pos, p_pos, n_pos = await asyncio.to_thread(
                    collect_avg, 1)
                f_neg, p_neg, n_neg = await asyncio.to_thread(
                    collect_avg, 0)
            finally:
                avg_btn.enable()

            log.push(
                f"  positive: {n_pos} эпох, negative: {n_neg} эпох")
            if p_pos is None or p_neg is None:
                ui.notify(
                    "Недостаточно эпох positive или negative",
                    color="warning")
                status_lbl.set_text("Готов")
                return

            n = min(len(p_pos), len(p_neg))
            f = f_pos[:n]
            p_pos, p_neg = p_pos[:n], p_neg[:n]

            current["selected_row"] = None
            meta_content_ref.refresh()
            plots_content_host.clear()
            with plots_content_host:
                try:
                    _render_fig(_build_avg_fig(
                        f, p_pos, p_neg, n_pos, n_neg), log)
                except Exception as e:
                    ui.label(f"⚠ {e}").classes("text-red-600")
                try:
                    _render_fig(_build_diff_fig(
                        f, p_pos, p_neg), log)
                except Exception as e:
                    ui.label(f"⚠ {e}").classes("text-red-600")
            status_lbl.set_text("Готов")
            log.push("--- avg PSD: готово ---")

        async def on_study_change(_e=None):
            sid = int(study_sel.value)
            if last_loaded["sid"] == sid and current["epochs_df"] is not None:
                return
            current["epochs_df"] = None
            current["selected_row"] = None
            current["sfreq"] = None
            epochs_list.refresh()
            meta_content_ref.refresh()
            plots_content.refresh()
            await _load_epochs()

        study_sel.on_value_change(on_study_change)
        cb_pos.on_value_change(lambda _e: epochs_list.refresh())
        cb_neg.on_value_change(lambda _e: epochs_list.refresh())
        cb_disc.on_value_change(lambda _e: epochs_list.refresh())
        avg_btn.on_click(do_avg_psd)

        async def _bootstrap():
            try:
                await _load_epochs()
            except Exception as e:
                log.push(f"(bootstrap) {e}")
                log.push(traceback.format_exc())

        ui.timer(0.2, _bootstrap, once=True)