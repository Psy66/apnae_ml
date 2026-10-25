"""Скачивание EDF-файлов в локальный кэш.

Особенности:
- Операция регистрируется в state.begin_busy/end_busy → видна в хедере
  на любой странице, защищает от случайного перехода.
- При возврате на страницу лог и прогресс восстанавливаются из state.
- Sync-таймер раз в 0.5 с подтягивает прогресс/сообщения и корректно
  разблокирует кнопку, если операция завершилась в фоне.
- Все UI-обновления в фоновых вызовах обёрнуты в try/except — если клиент
  ушёл со страницы, виджет уже удалён и обращение к нему упадёт.
"""
import asyncio
from nicegui import ui

from ui.state import state
from ui.logger import SafeLog, safe_notify
from ui.components import help_expansion, section_header
from ui.help_content import HELP_DOWNLOAD
from core.edf_cache import EDFCache


def page_download():
    from ui.app import create_layout
    create_layout()

    with ui.column().classes("p-6 w-full"):
        section_header(
            "1. Скачивание EDF",
            "Загрузка полисомнографических записей для локальной обработки",
        )

        help_expansion(
            "📖 Инструкция: как выбрать пациентов и скачать EDF",
            HELP_DOWNLOAD, icon="help",
        )

        if not state.api:
            ui.label(
                "⚠ Сначала настройте API на странице ⚙ Settings"
            ).classes("text-red-600 text-lg")
            return

        status = ui.label("Готов").classes("text-gray-600")
        progress = ui.linear_progress(value=0).classes("w-full")
        log = SafeLog(ui.log(max_lines=500).classes(
            "w-full h-56 font-mono text-xs"))

        checkboxes: dict = {}
        is_downloading = [False]

        # ------------------------------------------------------------
        # Восстановление после возврата на страницу
        # ------------------------------------------------------------
        if state.is_busy and state.busy_op == "download":
            log.restore_from_state()
            try:
                status.set_text(
                    f"⚠ Скачивание уже идёт: {state.busy_message}"
                )
                progress.value = state.busy_progress
            except Exception:
                pass

        # ------------------------------------------------------------
        # Вспомогательные функции
        # ------------------------------------------------------------
        def _collect_selected() -> list:
            return [sid for sid, cb in checkboxes.items() if cb.value]

        def _count_selected() -> int:
            return sum(1 for cb in checkboxes.values() if cb.value)

        # ------------------------------------------------------------
        # Загрузка списка исследований
        # ------------------------------------------------------------
        async def load_studies():
            status.set_text("Загрузка списка исследований...")
            try:
                state.studies = await asyncio.to_thread(state.api.get_studies)
                render_table()
                n_cached = sum(
                    1 for s in state.studies
                    if state.edf_cache.get_any(
                        s["study_id"], s.get("file_checksum", "") or "")
                )
                status.set_text(
                    f"Загружено {len(state.studies)} исследований. "
                    f"Уже в кэше: {n_cached}."
                )
            except Exception as e:
                status.set_text(f"Ошибка: {e}")
                log.push(f"Ошибка загрузки списка: {type(e).__name__}: {e}")

        # ------------------------------------------------------------
        # Таблица исследований
        # ------------------------------------------------------------
        def render_table():
            table_area.clear()
            checkboxes.clear()

            with table_area:
                with ui.row().classes("gap-2 mb-2 items-center"):
                    ui.button("Выбрать все",
                              on_click=lambda: toggle_all(True)
                              ).props("dense outline")
                    ui.button("Снять все",
                              on_click=lambda: toggle_all(False)
                              ).props("dense outline")
                    ui.label(f"Всего: {len(state.studies)}").classes(
                        "text-sm text-gray-600 ml-2")
                    ui.space()
                    ui.label("✓ = уже в кэше").classes(
                        "text-sm text-green-700")

                with ui.column().classes(
                        "border rounded p-2 w-full overflow-auto"
                ).style("max-height: 480px"):
                    for s in state.studies:
                        sid = s["study_id"]
                        checksum = s.get("file_checksum", "") or ""
                        cached = state.edf_cache.get_any(sid, checksum) is not None
                        mark = "✓ " if cached else ""
                        sev = s.get("breathing_impairment_severity", "?")
                        ahi = s.get("ahi", "?")
                        quality = s.get("rec_quality", "?")
                        label = (f"{mark}#{sid} | AHI={ahi} | {sev} | "
                                 f"quality={quality}")
                        cb = ui.checkbox(label)
                        cb.value = cached
                        checkboxes[sid] = cb

        def toggle_all(val: bool):
            if state.is_busy:
                safe_notify("Идёт операция — подождите завершения",
                            type="info")
                return
            for cb in checkboxes.values():
                cb.value = val

        # ------------------------------------------------------------
        # Скачивание
        # ------------------------------------------------------------
        async def download_selected():
            if state.is_busy:
                safe_notify("Другая операция уже идёт", type="info")
                return
            if is_downloading[0]:
                safe_notify("Загрузка уже идёт", type="info")
                return

            ids = _collect_selected()
            if not ids:
                safe_notify("Отметьте галочками хотя бы одно исследование",
                            type="warning", position="top")
                return

            is_downloading[0] = True
            state.begin_busy("download", f"0/{len(ids)}")
            try:
                download_btn.disable()
            except Exception:
                pass

            progress.value = 0
            done = 0
            try:
                for i, sid in enumerate(ids):
                    try:
                        study = next(
                            (s for s in state.studies
                             if s["study_id"] == sid), None)
                        if not study:
                            log.push(f"#{sid}: не найдено в state.studies")
                            continue
                        checksum = study.get("file_checksum", "") or ""
                        if state.edf_cache.get_any(sid, checksum):
                            log.push(f"#{sid}: ✓ уже в кэше")
                            done += 1
                            continue
                        log.push(f"#{sid}: скачивание...")
                        tmp = state.config.edf_dir / f"tmp_{sid}.edf"
                        await asyncio.to_thread(
                            state.api.download_edf, sid, tmp, None)
                        if not tmp.exists() or tmp.stat().st_size == 0:
                            log.push(f"#{sid}: ✗ пустой файл")
                            tmp.unlink(missing_ok=True)
                            continue
                        if checksum:
                            actual = EDFCache.sha256(tmp)
                            if actual != checksum:
                                tmp.unlink(missing_ok=True)
                                log.push(
                                    f"#{sid}: ✗ checksum mismatch "
                                    f"({actual[:8]}… vs {checksum[:8]}…)")
                                continue
                        final = state.edf_cache.store_any(
                            tmp, sid, checksum)
                        size_mb = final.stat().st_size / 1024 / 1024
                        log.push(f"#{sid}: ✓ OK ({size_mb:.1f} MB)")
                        done += 1
                        if sid in checkboxes:
                            try:
                                cb = checkboxes[sid]
                                if not cb.text.startswith("✓"):
                                    cb.set_text("✓ " + cb.text)
                            except Exception:
                                pass
                    except Exception as e:
                        log.push(f"#{sid}: ✗ ERROR "
                                 f"{type(e).__name__}: {e}")
                    finally:
                        p = (i + 1) / max(1, len(ids))
                        state.set_busy_progress(
                            p, f"{i+1}/{len(ids)} — #{sid}")
                        try:
                            progress.value = p
                            status.set_text(
                                f"Скачивание {i+1}/{len(ids)}: #{sid}")
                        except Exception:
                            pass

                state.set_busy_progress(1.0, f"готово {done}/{len(ids)}")
                try:
                    status.set_text(f"Готово: {done}/{len(ids)}")
                except Exception:
                    pass
            finally:
                is_downloading[0] = False
                state.end_busy()
                try:
                    download_btn.enable()
                except Exception:
                    pass

        # ------------------------------------------------------------
        # Кнопки и счётчик
        # ------------------------------------------------------------
        with ui.row().classes("gap-2 mb-4 items-center"):
            ui.button("📋 Загрузить список", on_click=load_studies)
            download_btn = ui.button("⬇ Скачать выбранные",
                                     on_click=download_selected,
                                     color="primary")
            counter_lbl = ui.label("Выбрано: 0").classes(
                "ml-4 text-sm text-gray-700")

        def _tick():
            try:
                counter_lbl.set_text(f"Выбрано: {_count_selected()}")
            except Exception:
                pass

        ui.timer(0.5, _tick)

        # ------------------------------------------------------------
        # Sync с state: раз в 0.5 с
        #   - если операция идёт → прогресс/статус/кнопка заблокированы
        #   - если операция только что завершилась в фоне → разблокируем
        # ------------------------------------------------------------
        def _sync_from_state():
            try:
                if state.is_busy and state.busy_op == "download":
                    progress.value = state.busy_progress
                    if state.busy_message:
                        status.set_text(
                            f"⚠ Скачивание: {state.busy_message}")
                    download_btn.disable()
                    is_downloading[0] = True
                elif is_downloading[0] and not state.is_busy:
                    # Операция только что завершилась (в фоне).
                    is_downloading[0] = False
                    download_btn.enable()
                    progress.value = 1.0
                    status.set_text("Готово (завершено в фоне)")
            except Exception:
                pass

        ui.timer(0.5, _sync_from_state)

        # --- контейнер для таблицы ---
        table_area = ui.column().classes("w-full")

        # --- начальная синхронизация UI, если операция уже шла ---
        if state.is_busy and state.busy_op == "download":
            try:
                download_btn.disable()
                is_downloading[0] = True
            except Exception:
                pass