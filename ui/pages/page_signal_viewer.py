"""Просмотр сырых ЭЭГ-сигналов для образовательных целей.

- Список эпох выбранного исследования с метками (✅/❌/⚠️).
- Волновая форма и PSD для кликнутой эпохи.
- Средние PSD «апноэ» vs «без апноэ» + разница (log10(pos/neg)).

Кэш сигналов — в core/edf_reader.SignalCache (~240 МБ на исследование).

Особенности:
- state.studies подгружается автоматически при первом клике
  «Загрузить сигналы» (не нужно сначала открывать Download).
- Если файл не найден по SHA256 — пробуем by_study fallback.
- Признаки грузятся только для выбранного study_id (быстрее в 15 раз).
"""
import asyncio
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nicegui import ui

from ui.state import state
from ui.components import help_expansion, section_header, warning_banner
from ui.logger import SafeLog, safe_notify
from ui.help_content import HELP_SIGNAL_VIEWER
from core.edf_reader import signal_cache, compute_psd, compute_avg_psd
from core.config import STANDARD_CHANNELS, BANDS


BAND_COLORS = {
    "delta": "#4a90d9", "theta": "#50b86c", "alpha": "#f5a623",
    "sigma": "#9b59b6", "beta": "#e74c3c", "gamma": "#8e44ad",
}
BAND_LABELS = {
    "delta": "δ 0.5–4", "theta": "θ 4–8", "alpha": "α 8–13",
    "sigma": "σ 12–15", "beta": "β 15–30", "gamma": "γ 30–45",
}


def page_signal_viewer():
    from ui.app import create_layout
    create_layout()

    with ui.column().classes("p-6 w-full"):
        section_header(
            "🔬 Просмотр сигналов",
            "Сырые волновые формы и спектры эпох «апноэ» vs «без апноэ»",
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

        # ============================================================
        # Левая панель: выбор и список эпох
        # ============================================================
        with ui.row().classes("w-full gap-4 no-wrap items-start"):
            with ui.column().classes("w-96 gap-2 shrink-0"):
                study_sel = ui.select(
                    study_ids, value=study_ids[0],
                    label="study_id",
                ).classes("w-full")
                chan_sel = ui.select(
                    STANDARD_CHANNELS, value="C3",
                    label="Канал",
                ).classes("w-full")
                load_btn = ui.button(
                    "📥 Загрузить сигналы в память",
                    color="primary",
                ).classes("w-full")
                status_lbl = ui.label("Не загружено").classes(
                    "text-sm text-gray-600")

                with ui.row().classes("gap-2 mt-2"):
                    cb_pos = ui.checkbox("✅ positive", value=True)
                    cb_neg = ui.checkbox("❌ negative", value=True)
                    cb_disc = ui.checkbox("⚠️ discard", value=False)

                ui.label("Эпохи:").classes("font-bold text-sm mt-2")
                epochs_area = ui.column().classes(
                    "w-full overflow-auto border rounded p-2"
                ).style("max-height: 380px")

                ui.separator()
                ui.label("Средние спектры:").classes(
                    "font-bold text-sm mt-2")
                avg_btn = ui.button(
                    "📊 Средние PSD: positive vs negative",
                    color="secondary",
                ).classes("w-full")

            # ============================================================
            # Правая панель: Notebook с графиками
            # ============================================================
            with ui.column().classes("flex-1"):
                with ui.tabs().classes("w-full") as tabs:
                    tab_wave = ui.tab("Волновая форма")
                    tab_psd = ui.tab("Спектр эпохи")
                    tab_avg = ui.tab("Средние спектры")
                with ui.tab_panels(tabs, value=tab_wave).classes("w-full"):
                    with ui.tab_panel(tab_wave):
                        wave_area = ui.column().classes("w-full")
                    with ui.tab_panel(tab_psd):
                        psd_area = ui.column().classes("w-full")
                    with ui.tab_panel(tab_avg):
                        avg_area = ui.column().classes("w-full")

        # ============================================================
        # Лог и состояние
        # ============================================================
        log = SafeLog(ui.log(max_lines=200).classes(
            "w-full h-24 font-mono text-xs"))
        current = {"epochs_df": None, "epoch_len": 10.0}

        # ------------------------------------------------------------
        # Вспомогательные
        # ------------------------------------------------------------
        def _clear(area):
            """Очищает NiceGUI-контейнер (НЕ Tkinter!)."""
            try:
                area.clear()
            except Exception:
                pass

        async def _ensure_studies() -> bool:
            """Гарантирует, что state.studies загружен.
            Если пуст — запрашивает у API. Возвращает True при успехе.
            """
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
            """Ищет EDF: сначала по SHA256 (через state.studies),
            потом fallback — по study_id."""
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
            """(epoch_len, t_pos, buffer_sec) для исследования."""
            matches = [item for item in available if item[0] == sid]
            if not matches:
                return None
            chosen = max(matches, key=lambda x: x[1])
            return float(chosen[1]), float(chosen[2]), float(chosen[3])

        def _clear_plots():
            _clear(wave_area)
            _clear(psd_area)
            _clear(avg_area)

        # ------------------------------------------------------------
        # Загрузка сигналов EDF в кэш
        # ------------------------------------------------------------
        async def do_load():
            load_btn.disable()
            try:
                # 1) гарантируем список исследований
                if not await _ensure_studies():
                    safe_notify(
                        "Не удалось загрузить список исследований. "
                        "Проверьте соединение на вкладке ⚙ Settings.",
                        type="negative")
                    return

                sid = int(study_sel.value)
                edf = _get_edf_path(sid)

                # 2) диагностика, если не нашли
                if not edf:
                    study = next(
                        (s for s in state.studies
                         if s["study_id"] == sid), None)
                    checksum = (study or {}).get(
                        "file_checksum", "") or ""
                    log.push(
                        f"EDF не найден: study_id={sid}, "
                        f"studies в памяти={len(state.studies)}, "
                        f"checksum={'есть' if checksum else 'пусто'}"
                    )
                    safe_notify(
                        f"EDF для study_id={sid} не найден в кэше. "
                        f"Скачайте на вкладке «1. Download».",
                        type="negative")
                    return

                status_lbl.set_text("Чтение EDF (5–15 сек)...")
                entry = await asyncio.to_thread(
                    signal_cache.load, sid, edf,
                    lambda m: log.push(m))
                status_lbl.set_text(
                    f"✓ {entry['duration']:.0f} с, "
                    f"каналов: {len(entry['channels'])}, "
                    f"sfreq={entry['sfreq']:.0f} Гц")
                log.push(f"study {sid}: сигналы в памяти")
                await _load_epochs()
            except Exception as e:
                status_lbl.set_text(f"Ошибка: {e}")
                log.push(f"Ошибка загрузки: {e}")
                log.push(traceback.format_exc().splitlines()[-1])
            finally:
                load_btn.enable()

        # ------------------------------------------------------------
        # Загрузка эпох из FeatureStore (только выбранного исследования)
        # ------------------------------------------------------------
        async def _load_epochs():
            sid = int(study_sel.value)
            meta = _meta_for_study(sid)
            if meta is None:
                safe_notify(
                    "Нет данных признаков для этого исследования",
                    type="warning")
                return
            epoch_len, t_pos, buffer_sec = meta
            current["epoch_len"] = epoch_len

            def _load():
                # грузим только нужный study_id — в 15 раз быстрее
                return state.feature_store.load_all(
                    epoch_len, study_ids=[sid],
                    t_pos=t_pos, buffer_sec=buffer_sec)

            try:
                df = await asyncio.to_thread(_load)
            except Exception as e:
                log.push(f"Ошибка загрузки признаков: {e}")
                return

            if df.empty:
                safe_notify(
                    "Нет эпох для этого исследования", type="warning")
                return
            current["epochs_df"] = df.reset_index(drop=True)
            render_epochs()

        # ------------------------------------------------------------
        # Отрисовка списка эпох
        # ------------------------------------------------------------
        def render_epochs():
            _clear(epochs_area)
            df = current["epochs_df"]
            if df is None or df.empty:
                return
            wanted = []
            if cb_pos.value:
                wanted.append(1)
            if cb_neg.value:
                wanted.append(0)
            if cb_disc.value:
                wanted.append(-1)
            df_show = df[df["label"].isin(wanted)]

            with epochs_area:
                if df_show.empty:
                    ui.label(
                        "Нет эпох с выбранными метками"
                    ).classes("text-sm text-gray-500")
                    return
                for _, row in df_show.head(300).iterrows():
                    lbl = int(row["label"])
                    mark = {1: "✅", 0: "❌", -1: "⚠️"}.get(lbl, "?")
                    ts = float(row["epoch_start"])
                    ov = float(row.get("overlap_fraction", 0.0))
                    btn = ui.button(
                        f"{mark}  t={ts:.0f}s  |  ov={ov:.2f}",
                    ).props("flat dense align=left").classes(
                        "w-full text-left font-mono text-xs")
                    btn.on_click(lambda r=row: show_epoch(r))
                if len(df_show) > 300:
                    ui.label(
                        f"(показаны первые 300 из {len(df_show)})"
                    ).classes("text-xs text-gray-500")

        # ------------------------------------------------------------
        # Волна + PSD эпохи
        # ------------------------------------------------------------
        def show_epoch(row):
            sid = int(study_sel.value)
            ch = chan_sel.value
            ts = float(row["epoch_start"])
            epoch_len = float(row.get("epoch_len", current["epoch_len"]))
            sfreq = signal_cache.sfreq(sid)
            if sfreq is None:
                safe_notify(
                    "Сначала загрузите сигналы в память",
                    type="warning")
                return
            seg = signal_cache.get_epoch(sid, ts, epoch_len, ch)
            if seg is None or len(seg) < 16:
                safe_notify(
                    "Не удалось получить сигнал этой эпохи",
                    type="warning")
                return
            seg_uv = seg * 1e6

            lbl = int(row["label"])
            mark = {1: "✅ positive", 0: "❌ negative",
                    -1: "⚠️ discard"}.get(lbl, "?")
            ov = float(row.get("overlap_fraction", 0.0))

            # --- Волновая форма ---
            _clear(wave_area)
            fig, ax = plt.subplots(figsize=(10, 3))
            t = np.arange(len(seg_uv)) / sfreq + ts
            ax.plot(t, seg_uv, linewidth=0.7, color="#2980b9")
            ax.set_xlabel("Время, сек")
            ax.set_ylabel("Амплитуда, мкВ")
            ax.set_title(
                f"{ch}  |  эпоха {ts:.0f}–{ts + epoch_len:.0f} с  |  "
                f"{mark}  |  overlap={ov:.2f}"
            )
            ax.grid(alpha=0.3)
            fig.tight_layout()
            with wave_area:
                ui.pyplot(fig)
            plt.close(fig)

            # --- PSD эпохи ---
            _clear(psd_area)
            freqs, psd = compute_psd(seg_uv, sfreq)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            for name, (lo, hi) in BANDS.items():
                ax2.axvspan(lo, hi, alpha=0.12,
                            color=BAND_COLORS[name])
            ax2.semilogy(freqs, psd, color="#2c3e50", linewidth=1.2)
            ax2.set_xlim(0.5, 45)
            ax2.set_xlabel("Частота, Гц")
            ax2.set_ylabel("PSD, мкВ²/Гц (log)")
            ax2.set_title(
                f"PSD эпохи t={ts:.0f}s, канал {ch}  |  {mark}")
            if len(freqs) > 0 and len(psd) > 0:
                positive_psd = psd[psd > 0]
                ymin = positive_psd.min() if len(positive_psd) else 1e-6
                ymax = psd.max() if psd.max() > 0 else 1.0
                if ymax <= 0:
                    ymax = 1.0
                for name, (lo, hi) in BANDS.items():
                    ax2.text((lo + hi) / 2,
                             ymax * 0.5,
                             BAND_LABELS[name],
                             ha="center", fontsize=8,
                             color=BAND_COLORS[name],
                             alpha=0.85)
            ax2.grid(alpha=0.3, which="both")
            fig2.tight_layout()
            with psd_area:
                ui.pyplot(fig2)
            plt.close(fig2)

        # ------------------------------------------------------------
        # Средние PSD (positive vs negative) + разница
        # ------------------------------------------------------------
        async def do_avg_psd():
            df = current["epochs_df"]
            sid = int(study_sel.value)
            ch = chan_sel.value
            if df is None or df.empty:
                safe_notify(
                    "Сначала загрузите сигналы и эпохи", type="warning")
                return
            sfreq = signal_cache.sfreq(sid)
            if sfreq is None:
                safe_notify(
                    "Сначала загрузите сигналы в память", type="warning")
                return
            epoch_len = float(current["epoch_len"])
            avg_btn.disable()

            def collect(label):
                segs = []
                for ts in df[df["label"] == label]["epoch_start"].values:
                    seg = signal_cache.get_epoch(
                        sid, float(ts), epoch_len, ch)
                    if seg is not None and len(seg) >= 16:
                        segs.append(seg * 1e6)
                return segs

            segs_pos = collect(1)
            segs_neg = collect(0)
            log.push(
                f"Сегментов positive: {len(segs_pos)}, "
                f"negative: {len(segs_neg)}"
            )
            res_pos = await asyncio.to_thread(
                compute_avg_psd, segs_pos, sfreq,
                lambda m: log.push(m))
            res_neg = await asyncio.to_thread(
                compute_avg_psd, segs_neg, sfreq,
                lambda m: log.push(m))
            avg_btn.enable()

            if res_pos is None or res_neg is None:
                safe_notify(
                    "Недостаточно данных для средних PSD",
                    type="warning")
                return

            f_pos, p_pos = res_pos
            f_neg, p_neg = res_neg
            n = min(len(p_pos), len(p_neg))
            f_pos, p_pos = f_pos[:n], p_pos[:n]
            f_neg, p_neg = f_neg[:n], p_neg[:n]

            _clear(avg_area)
            with avg_area:
                # --- Средние PSD ---
                fig, ax = plt.subplots(figsize=(11, 4.5))
                ax.semilogy(f_pos, p_pos, color="#2ecc71",
                            linewidth=1.8,
                            label=f"Positive (n={len(segs_pos)})")
                ax.semilogy(f_neg, p_neg, color="#e74c3c",
                            linewidth=1.8,
                            label=f"Negative (n={len(segs_neg)})")
                ax.set_xlim(0.5, 45)
                ax.set_xlabel("Частота, Гц")
                ax.set_ylabel("Средний PSD, мкВ²/Гц (log)")
                ax.set_title(
                    f"Средние спектры: positive vs negative ({ch})")
                ax.legend()
                ax.grid(alpha=0.3, which="both")
                fig.tight_layout()
                ui.pyplot(fig)
                plt.close(fig)

                # --- Разница ---
                fig2, ax2 = plt.subplots(figsize=(11, 3))
                ratio = np.log10(p_pos / np.maximum(p_neg, 1e-12))
                ax2.axhline(y=0, color="gray",
                            linestyle="--", alpha=0.6)
                ax2.plot(f_pos, ratio,
                         color="#8e44ad", linewidth=1.5)
                for name, (lo, hi) in BANDS.items():
                    ax2.axvspan(lo, hi, alpha=0.1,
                                color=BAND_COLORS[name])
                ax2.set_xlim(0.5, 45)
                ax2.set_xlabel("Частота, Гц")
                ax2.set_ylabel("log10(pos / neg)")
                ax2.set_title(
                    "Разница спектров (>0 = выше при апноэ, "
                    "<0 = ниже при апноэ)")
                ax2.grid(alpha=0.3)
                fig2.tight_layout()
                ui.pyplot(fig2)
                plt.close(fig2)

        # ------------------------------------------------------------
        # Переключение исследования/канала
        # ------------------------------------------------------------
        def _on_change():
            sid = int(study_sel.value)
            if signal_cache.is_loaded(sid):
                # Перезагружаем эпохи для нового study_id
                asyncio.create_task(_load_epochs())
            else:
                _clear(epochs_area)
                _clear_plots()
                current["epochs_df"] = None
                status_lbl.set_text("Нажмите «Загрузить сигналы»")

        # ------------------------------------------------------------
        # Обработчики
        # ------------------------------------------------------------
        load_btn.on_click(do_load)
        study_sel.on_value_change(lambda e: _on_change())
        chan_sel.on_value_change(lambda e: _on_change())
        cb_pos.on_value_change(lambda e: render_epochs())
        cb_neg.on_value_change(lambda e: render_epochs())
        cb_disc.on_value_change(lambda e: render_epochs())
        avg_btn.on_click(do_avg_psd)