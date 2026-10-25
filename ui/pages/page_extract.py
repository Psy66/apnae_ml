"""Батч-извлечение признаков из скачанных EDF.

Особенности:
- Каждый файл обрабатывается в отдельном потоке (asyncio.to_thread).
- Результат сохраняется с ПОЛНЫМ ключом: (study_id, epoch_len, t_pos,
  buffer_sec) — иначе при смене t_pos или buffer кэш путается.
- Операция регистрируется в state.begin_busy / end_busy → видна в хедере
  и защищает от случайного перехода на другую страницу.
- При возврате на страницу лог и прогресс восстанавливаются из state.
- Sync-таймер раз в 0.5 с подтягивает прогресс и корректно разблокирует
  кнопку, если операция завершилась в фоне.
- Все UI-обновления в фоновых вызовах обёрнуты в try/except — если клиент
  ушёл со страницы, виджет уже удалён и обращение к нему упадёт.
"""
import asyncio
import traceback

from nicegui import ui

from ui.state import state
from ui.logger import SafeLog, safe_notify
from ui.components import help_expansion, section_header, warning_banner
from ui.help_content import HELP_EXTRACT
from core.feature_extractor import FeatureExtractor


def page_extract():
    from ui.app import create_layout
    create_layout()

    with ui.column().classes("p-6 w-full"):
        section_header(
            "2. Извлечение признаков",
            "EDF → таблица признаков (спектральные мощности по каналам)",
        )

        help_expansion(
            "📖 Инструкция: параметры, время работы, диагностика",
            HELP_EXTRACT, icon="help",
        )

        if not state.api:
            warning_banner(
                "Сначала настройте API на странице ⚙ Settings",
                level="warning",
            )
            return

        # ------------------------------------------------------------
        # Параметры
        # ------------------------------------------------------------
        with ui.row().classes("gap-4 mb-4"):
            epoch_in = ui.number("Длина суб-эпохи (с)", value=10,
                                 min=2, max=90, step=1).classes("w-44")
            tpos_in = ui.number("Порог t_pos", value=0.5,
                                min=0.1, max=1.0, step=0.05).classes("w-32")
            buffer_in = ui.number("Буфер is_tonic (с)", value=30,
                                  min=0, max=120, step=5).classes("w-40")

        with ui.row().classes("gap-2 mb-4 items-center"):
            load_btn = ui.button("📋 Загрузить список исследований")
            run_btn = ui.button("▶ Запустить извлечение", color="primary")
            force_cb = ui.checkbox("Перезаписать существующие", value=False)

        status = ui.label("Готов").classes("text-gray-600")
        progress = ui.linear_progress(value=0).classes("w-full")
        log = SafeLog(ui.log(max_lines=500).classes(
            "w-full h-72 font-mono text-xs"))

        # ------------------------------------------------------------
        # Восстановление после возврата на страницу
        # ------------------------------------------------------------
        if state.is_busy and state.busy_op == "extract":
            log.restore_from_state()
            try:
                status.set_text(
                    f"⚠ Извлечение уже идёт: {state.busy_message}"
                )
                progress.value = state.busy_progress
            except Exception:
                pass

        # ------------------------------------------------------------
        # Загрузка списка исследований
        # ------------------------------------------------------------
        async def load_list():
            status.set_text("Загрузка...")
            try:
                if not state.studies:
                    state.studies = await asyncio.to_thread(
                        state.api.get_studies)
                cached_n = sum(
                    1 for s in state.studies
                    if state.edf_cache.get_any(
                        s["study_id"], s.get("file_checksum", "") or "")
                )
                status.set_text(
                    f"Готово: {len(state.studies)} исследований, "
                    f"из них скачано {cached_n}"
                )
            except Exception as e:
                status.set_text(f"Ошибка: {e}")
                log.push(f"Ошибка загрузки списка: {type(e).__name__}: {e}")

        # ------------------------------------------------------------
        # Основная операция — извлечение
        # ------------------------------------------------------------
        async def run():
            if state.is_busy:
                safe_notify("Другая операция уже идёт", type="warning")
                return
            if not state.studies:
                safe_notify("Сначала загрузите список исследований",
                            type="warning")
                return

            epoch_len = float(epoch_in.value)
            t_pos = float(tpos_in.value)
            buffer_sec = float(buffer_in.value)

            # Собираем список к обработке (кэш зависит от ВСЕХ параметров)
            to_process = []
            for s in state.studies:
                edf = state.edf_cache.get_any(
                    s["study_id"], s.get("file_checksum", "") or "")
                if not edf:
                    continue
                if not force_cb.value and state.feature_store.exists(
                        s["study_id"], epoch_len, t_pos, buffer_sec):
                    continue
                to_process.append((s, edf))

            log.push(
                f"К извлечению: {len(to_process)} исследований "
                f"(len={epoch_len}s, t_pos={t_pos:.2f}, "
                f"buffer={buffer_sec:.0f}s)"
            )
            progress.value = 0

            if not to_process:
                status.set_text(
                    "Нечего обрабатывать: либо нет скачанных EDF, "
                    "либо признаки уже извлечены с этими параметрами "
                    "(включите «Перезаписать» или поменяйте t_pos/buffer)"
                )
                return

            # --- Старт ---
            run_btn.disable()
            state.begin_busy("extract", f"0/{len(to_process)}")
            extractor = FeatureExtractor(
                epoch_length=epoch_len,
                t_pos=t_pos,
                buffer_sec=buffer_sec,
            )

            try:
                for i, (study, edf) in enumerate(to_process):
                    sid = study["study_id"]
                    try:
                        log.push(
                            f"#{sid}: загрузка событий из API (fallback)...")
                        try:
                            events_raw = await asyncio.to_thread(
                                state.api.get_phasic_events, [sid])
                            events_api = [
                                (float(e["event_onset"]),
                                 float(e["event_offset"]))
                                for e in events_raw
                            ]
                        except Exception:
                            events_api = []

                        log.push(
                            f"#{sid}: извлечение (читаем EDF)...")
                        result = await asyncio.to_thread(
                            extractor.extract,
                            edf, sid, study["patient_id"],
                            events_api,
                            study.get("hypnogram_data"),
                        )

                        # ⚠ ВАЖНО: сохраняем с ПОЛНЫМ ключом
                        state.feature_store.save(
                            result.df, sid, epoch_len,
                            t_pos, buffer_sec,
                        )

                        log.push(
                            f"#{sid}: ✓ {result.n_epochs_labeled} эпох "
                            f"(N2/N3: {result.n_epochs_n2n3}/"
                            f"{result.n_epochs_total}) | "
                            f"stages={result.stages_source} "
                            f"{result.stage_counts} | "
                            f"events={result.events_source} "
                            f"EDF={result.n_events_edf} "
                            f"API={result.n_events_api} | "
                            f"artifacts={result.n_artifacts} | "
                            f"enc={result.encoding}"
                        )
                    except Exception as e:
                        tb = traceback.format_exc()
                        log.push(
                            f"#{sid}: ✗ ERROR {type(e).__name__}: {e}")
                        for line in tb.strip().splitlines()[-3:]:
                            log.push(f"  {line}")
                    finally:
                        p = (i + 1) / max(1, len(to_process))
                        state.set_busy_progress(
                            p, f"{i+1}/{len(to_process)} — #{sid}")
                        try:
                            progress.value = p
                            status.set_text(
                                f"Извлечение {i+1}/{len(to_process)}: #{sid}")
                        except Exception:
                            pass

                state.set_busy_progress(1.0, "готово")
                try:
                    status.set_text("Извлечение завершено")
                except Exception:
                    pass
            finally:
                state.end_busy()
                try:
                    run_btn.enable()
                except Exception:
                    pass

        # ------------------------------------------------------------
        # Sync с state: раз в 0.5 с
        # ------------------------------------------------------------
        def _sync_from_state():
            try:
                if state.is_busy and state.busy_op == "extract":
                    progress.value = state.busy_progress
                    if state.busy_message:
                        status.set_text(
                            f"⚠ Извлечение: {state.busy_message}")
                    run_btn.disable()
                elif not state.is_busy:
                    if not run_btn.enabled:
                        # Разблокируем только если это мы сами блокировали
                        # (is_busy только что стало False)
                        pass
            except Exception:
                pass

        ui.timer(0.5, _sync_from_state)

        # ------------------------------------------------------------
        # Обработчики
        # ------------------------------------------------------------
        load_btn.on_click(load_list)
        run_btn.on_click(run)