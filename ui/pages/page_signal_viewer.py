"""Просмотр сырых ЭЭГ-сигналов — упрощённый визуальный контроль.

Особенности:
- НЕТ выбора канала: waveform показывает все 6 каналов сразу
  (offset-plot), PSD — усреднённый по 6 каналам.
- НЕТ tabs: все графики в одном контейнере.
- Автозагрузка сигналов: клик по эпохе сам читает EDF.
- matplotlib.figure.Figure напрямую (без pyplot) — async-safe.
- ui.image(base64) — не зависит от версии NiceGUI.
- @ui.refreshable для списка эпох (идиоматично, без race).
- last_loaded_sid: не перезагружает эпохи зря и не стирает их
  при первичной установке value (главный баг старой версии).
- Async-обработчики передаются в on_click НАПРЯМУЮ — NiceGUI
  сам их await-ит, ui.context сохраняется (не теряется notify).
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
LABEL_EMOJI = {1: "✅", 0: "❌", -1: "⚠️"}
LABEL_ASCII = {1: "POSITIVE", 0: "NEGATIVE", -1: "DISCARD"}


# ---------------------------------------------------------------- helpers
def _fig_to_b64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _show_fig(container, fig: Figure, log: SafeLog | None = None) -> bool:
    """Показать Figure в контейнере. Возвращает True при успехе."""
    try:
        b64 = _fig_to_b64(fig)
    except Exception as e:
        if log:
            log.push(f"  ✗ savefig failed: {e}")
        return False
    try:
        with container:
            ui.image(f"data:image/png;base64,{b64}").classes("w-full")
        if log:
            log.push(f"  ✓ график отрисован ({len(b64)//1024} KB)")
        return True
    except Exception as e:
        if log:
            log.push(f"  ✗ ui.image failed: {e}")
        return False


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

        # ------------------------------------------------------------
        # Состояние страницы (замыкание на этот экземпляр страницы)
        # ------------------------------------------------------------
        current = {"epochs_df": None, "epoch_len": 10.0}
        last_loaded = {"sid": None}
        plots_placeholder = {"shown": False}

        # ------------------------------------------------------------
        # Разметка
        # ------------------------------------------------------------
        with ui.row().classes("w-full gap-4 items-start"):
            # ---------------- Левая колонка ----------------
            with ui.column().classes("w-80 gap-2 shrink-0"):
                study_sel = ui.select(
                    study_ids, value=study_ids[0],
                    label="study_id",
                ).classes("w-full")
                status_lbl = ui.label("Сигналы не загружены").classes(
                    "text-sm text-gray-600")

                with ui.row().classes("gap-2 mt-2"):
                    cb_pos = ui.checkbox("✅ positive", value=True)
                    cb_neg = ui.checkbox("❌ negative", value=True)
                    cb_disc = ui.checkbox("⚠️ discard", value=False)

                ui.label("Эпохи (клик — показать):").classes(
                    "font-bold text-sm mt-2")

                with ui.column().classes(
                        "w-full overflow-auto border rounded p-2"
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

            # ---------------- Правая колонка: графики ----------------
            with ui.column().classes("flex-1 min-w-0"):
                plots_area = ui.column().classes("w-full gap-3")
                with plots_area:
                    ui.label(
                        "👈 Выберите эпоху слева — здесь появятся "
                        "волновая форма и спектр."
                    ).classes("text-gray-500 italic")

        # ------------------------------------------------------------
        # Лог
        # ------------------------------------------------------------
        log = SafeLog(ui.log(max_lines=300).classes(
            "w-full h-32 font-mono text-xs mt-4"))

        # ------------------------------------------------------------
        # Refreshable-список эпох — рендерится из current["epochs_df"]
        # ------------------------------------------------------------
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
                    "w-full font-mono text-xs")

            if len(df_show) > 300:
                ui.label(
                    f"(показаны первые 300 из {len(df_show)})"
                ).classes("text-xs text-gray-500")

        # Первичный рендер
        with epochs_area_host:
            epochs_list()

        # ------------------------------------------------------------
        # Вспомогательные
        # ------------------------------------------------------------
        def _clear_plots():
            plots_area.clear()
            plots_placeholder["shown"] = False

        def _reset_plots_placeholder():
            plots_area.clear()
            with plots_area:
                ui.label(
                    "👈 Выберите эпоху слева — здесь появятся "
                    "волновая форма и спектр."
                ).classes("text-gray-500 italic")
            plots_placeholder["shown"] = True

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
                None,
            )
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
                status_lbl.set_text(
                    f"✓ {entry['duration']:.0f} с, "
                    f"каналов: {len(entry['channels'])}, "
                    f"sfreq={entry['sfreq']:.0f} Гц")
                log.push(f"study {sid}: сигналы в памяти")
                return True
            except Exception as e:
                status_lbl.set_text(f"Ошибка: {e}")
                log.push(f"Ошибка чтения EDF: {e}")
                return False

        # ------------------------------------------------------------
        # Загрузка эпох (с защитой от двойной/бессмысленной загрузки)
        # ------------------------------------------------------------
        async def _load_epochs():
            sid = int(study_sel.value)
            # Уже загружено и не изменилось — выходим
            if last_loaded["sid"] == sid and current["epochs_df"] is not None:
                log.push(f"_load_epochs: sid={sid} уже загружен, skip")
                return

            meta = _meta_for_study(sid)
            if meta is None:
                ui.notify(
                    "Нет данных признаков для этого исследования",
                    color="warning")
                return
            epoch_len, t_pos, buffer_sec = meta
            current["epoch_len"] = epoch_len

            # Оптимистичная блокировка ДО await — иначе два параллельных
            # вызова (bootstrap + on_value_change) пойдут одновременно.
            last_loaded["sid"] = sid
            log.push(f"_load_epochs: загрузка sid={sid}...")

            def _load():
                return state.feature_store.load_all(
                    epoch_len, study_ids=[sid],
                    t_pos=t_pos, buffer_sec=buffer_sec)

            try:
                df = await asyncio.to_thread(_load)
            except Exception as e:
                last_loaded["sid"] = None  # сброс блокировки
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

        # ------------------------------------------------------------
        # Waveform + PSD (построение фигур)
        # ------------------------------------------------------------
        def _build_waveform_fig(sid, ts, epoch_len, sfreq, mark, ov) -> Figure:
            fig = Figure(figsize=(9, 3.2))
            ax = fig.add_subplot(111)
            t = None
            n_drawn = 0
            for i, ch in enumerate(STANDARD_CHANNELS):
                seg = signal_cache.get_epoch(sid, ts, epoch_len, ch)
                if seg is None or len(seg) < 16:
                    continue
                seg_uv = seg * 1e6
                sd = float(np.std(seg_uv))
                if sd < 1e-9:
                    sd = 1.0
                y = (i * 3.0) + seg_uv / sd
                if t is None:
                    t = np.arange(len(seg_uv)) / sfreq + ts
                ax.plot(t, y, linewidth=0.5, label=ch)
                ax.axhline(y=i * 3.0, color="gray", alpha=0.15,
                           linewidth=0.5)
                n_drawn += 1
            if t is None or n_drawn == 0:
                raise ValueError("Нет данных ни по одному каналу")
            ax.set_yticks([i * 3.0 for i in range(n_drawn)])
            ax.set_yticklabels(STANDARD_CHANNELS[:n_drawn], fontsize=9)
            ax.set_xlabel("Время, сек")
            ax.set_title(
                f"Эпоха {ts:.0f}–{ts + epoch_len:.0f} с  |  "
                f"{mark}  |  overlap={ov:.2f}  |  "
                f"6 каналов (нормированы на SD)")
            ax.grid(alpha=0.25, axis="x")
            fig.tight_layout()
            return fig

        def _build_psd_fig(sid, ts, epoch_len, sfreq, mark) -> Figure:
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
                raise ValueError("Нет данных для PSD")
            n = min(len(p) for p in psds)
            psd_mean = np.mean([p[:n] for p in psds], axis=0)
            freqs = freqs[:n]

            fig = Figure(figsize=(9, 2.8))
            ax = fig.add_subplot(111)
            for name, (lo, hi) in BANDS.items():
                ax.axvspan(lo, hi, alpha=0.12, color=BAND_COLORS[name])
            ax.semilogy(freqs, psd_mean, color="#2c3e50", linewidth=1.2)
            ax.set_xlim(0.5, 45)
            ax.set_xlabel("Частота, Гц")
            ax.set_ylabel("PSD, мкВ²/Гц (log)")
            ax.set_title(
                f"PSD эпохи t={ts:.0f}s  |  {mark}  |  "
                f"среднее по {len(psds)} каналам")
            psd_pos = psd_mean[psd_mean > 0]
            if len(psd_pos) > 0:
                ymax = float(psd_pos.max())
                for name, (lo, hi) in BANDS.items():
                    ax.text((lo + hi) / 2, ymax * 0.5, BAND_LABELS[name],
                            ha="center", fontsize=8,
                            color=BAND_COLORS[name], alpha=0.85)
            ax.grid(alpha=0.3, which="both")
            fig.tight_layout()
            return fig

        # ------------------------------------------------------------
        # Клик по эпохе → отрисовка
        # ------------------------------------------------------------
        async def show_epoch(row):
            sid = int(study_sel.value)
            ts = float(row["epoch_start"])
            epoch_len = float(row.get("epoch_len", current["epoch_len"]))
            lbl = int(row["label"])
            mark = LABEL_ASCII.get(lbl, "?")
            ov = float(row.get("overlap_fraction", 0.0))

            log.push(f"--- show_epoch: sid={sid}, ts={ts:.1f} ---")

            if not await _ensure_signals_loaded(sid):
                return
            sfreq = signal_cache.sfreq(sid)
            if sfreq is None:
                log.push("  ✗ sfreq=None после загрузки")
                return

            _clear_plots()
            try:
                fig_wave = _build_waveform_fig(
                    sid, ts, epoch_len, sfreq, mark, ov)
                _show_fig(plots_area, fig_wave, log)
            except Exception as e:
                log.push(f"  ✗ waveform failed: {e}")
                log.push(traceback.format_exc().splitlines()[-1])

            try:
                fig_psd = _build_psd_fig(
                    sid, ts, epoch_len, sfreq, mark)
                _show_fig(plots_area, fig_psd, log)
            except Exception as e:
                log.push(f"  ✗ PSD failed: {e}")
                log.push(traceback.format_exc().splitlines()[-1])

        # ------------------------------------------------------------
        # Средние PSD
        # ------------------------------------------------------------
        async def do_avg_psd():
            df = current["epochs_df"]
            sid = int(study_sel.value)
            if df is None or df.empty:
                ui.notify(
                    "Сначала дождитесь загрузки списка эпох (слева)",
                    color="warning")
                return
            if not await _ensure_signals_loaded(sid):
                return
            sfreq = signal_cache.sfreq(sid)
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
                n_epochs_used = 0
                for ts in ts_list:
                    ch_psds = []
                    for ch in STANDARD_CHANNELS:
                        seg = signal_cache.get_epoch(
                            sid, float(ts), epoch_len, ch)
                        if seg is None or len(seg) < 16:
                            continue
                        f, p = compute_psd(seg * 1e6, sfreq)
                        if len(p) == 0:
                            continue
                        if freqs is None:
                            freqs = f
                        ch_psds.append(p)
                    if not ch_psds:
                        continue
                    n = min(len(p) for p in ch_psds)
                    all_psds.append(
                        np.mean([p[:n] for p in ch_psds], axis=0))
                    n_epochs_used += 1
                if not all_psds or freqs is None:
                    return None, None, 0
                n = min(len(p) for p in all_psds)
                return (freqs[:n],
                        np.mean([p[:n] for p in all_psds], axis=0),
                        n_epochs_used)

            try:
                res_pos = await asyncio.to_thread(collect_avg, 1)
                res_neg = await asyncio.to_thread(collect_avg, 0)
            finally:
                avg_btn.enable()

            f_pos, p_pos, n_pos = res_pos
            f_neg, p_neg, n_neg = res_neg
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

            _clear_plots()

            try:
                fig = Figure(figsize=(9, 3.2))
                ax = fig.add_subplot(111)
                for name, (lo, hi) in BANDS.items():
                    ax.axvspan(lo, hi, alpha=0.10,
                               color=BAND_COLORS[name])
                ax.semilogy(f, p_pos, color="#2ecc71", linewidth=1.8,
                            label=f"positive (n={n_pos})")
                ax.semilogy(f, p_neg, color="#e74c3c", linewidth=1.8,
                            label=f"negative (n={n_neg})")
                ax.set_xlim(0.5, 45)
                ax.set_xlabel("Частота, Гц")
                ax.set_ylabel("Средний PSD, мкВ²/Гц (log)")
                ax.set_title(
                    "Средние спектры positive vs negative "
                    "(усреднено по 6 каналам)")
                ax.legend()
                ax.grid(alpha=0.3, which="both")
                fig.tight_layout()
                _show_fig(plots_area, fig, log)
            except Exception as e:
                log.push(f"  ✗ avg PSD plot failed: {e}")

            try:
                fig2 = Figure(figsize=(9, 2.5))
                ax2 = fig2.add_subplot(111)
                ratio = np.log10(p_pos / np.maximum(p_neg, 1e-12))
                ax2.axhline(y=0, color="gray",
                            linestyle="--", alpha=0.6)
                ax2.plot(f, ratio, color="#8e44ad", linewidth=1.5)
                for name, (lo, hi) in BANDS.items():
                    ax2.axvspan(lo, hi, alpha=0.10,
                                color=BAND_COLORS[name])
                ax2.set_xlim(0.5, 45)
                ax2.set_xlabel("Частота, Гц")
                ax2.set_ylabel("log10(pos / neg)")
                ax2.set_title(
                    "Разница спектров  (>0 = выше при апноэ, "
                    "<0 = ниже при апноэ)")
                ax2.grid(alpha=0.3)
                fig2.tight_layout()
                _show_fig(plots_area, fig2, log)
            except Exception as e:
                log.push(f"  ✗ diff plot failed: {e}")

            status_lbl.set_text("Готов")
            log.push("--- avg PSD: готово ---")

        # ------------------------------------------------------------
        # Смена study_id
        # ------------------------------------------------------------
        async def on_study_change(_e=None):
            sid = int(study_sel.value)
            if last_loaded["sid"] == sid and current["epochs_df"] is not None:
                return  # уже загружено
            # Новое исследование — сбрасываем старые данные
            current["epochs_df"] = None
            epochs_list.refresh()  # покажет "Загрузка эпох..."
            _clear_plots()
            _reset_plots_placeholder()
            await _load_epochs()

        # ------------------------------------------------------------
        # Обработчики (async передаём напрямую — NiceGUI их await-ит)
        # ------------------------------------------------------------
        study_sel.on_value_change(on_study_change)
        cb_pos.on_value_change(lambda _e: epochs_list.refresh())
        cb_neg.on_value_change(lambda _e: epochs_list.refresh())
        cb_disc.on_value_change(lambda _e: epochs_list.refresh())
        avg_btn.on_click(do_avg_psd)

        # ------------------------------------------------------------
        # Начальная загрузка (после того как страница отрендерилась)
        # ------------------------------------------------------------
        async def _bootstrap():
            try:
                await _load_epochs()
            except Exception as e:
                log.push(f"(bootstrap) {e}")
                log.push(traceback.format_exc().splitlines()[-1])

        ui.timer(0.2, _bootstrap, once=True)