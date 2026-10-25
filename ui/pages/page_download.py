"""Скачивание EDF-файлов в локальный кэш."""
import asyncio
from nicegui import ui
from ui.state import state
from core.edf_cache import EDFCache


def page_download():
    from ui.app import create_layout
    create_layout()
    with ui.column().classes("p-6 w-full"):
        ui.label("1. Скачивание EDF").classes("text-2xl font-bold mb-4")

        if not state.api:
            ui.label("⚠ Сначала настройте API (⚙ Settings)").classes("text-red-600")
            return

        status = ui.label("Готов").classes("text-gray-600")
        progress = ui.linear_progress(value=0).classes("w-full")
        log = ui.log(max_lines=300).classes("w-full h-48 font-mono text-xs")

        selected = set()
        checkboxes = {}

        async def load_studies():
            status.set_text("Загрузка списка исследований...")
            try:
                state.studies = await asyncio.to_thread(state.api.get_studies)
                render_table()
                status.set_text(f"Загружено {len(state.studies)} исследований")
            except Exception as e:
                status.set_text(f"Ошибка: {e}")
                log.push(f"Ошибка: {e}")

        def render_table():
            table_area.clear()
            checkboxes.clear()
            selected.clear()
            with table_area:
                with ui.row().classes("gap-2 mb-2"):
                    ui.button("Все", on_click=lambda: toggle_all(True)).props("dense")
                    ui.button("Ничего", on_click=lambda: toggle_all(False)).props("dense")
                    ui.label(f"{len(state.studies)} исследований").classes("text-sm text-gray-600")
                with ui.column().classes("border rounded p-2 h-96 w-full overflow-auto"):
                    for s in state.studies:
                        sid = s["study_id"]
                        checksum = s.get("file_checksum", "")
                        cached = state.edf_cache.get(checksum) is not None
                        mark = "✓ " if cached else "  "
                        sev = s.get("breathing_impairment_severity", "?")
                        ahi = s.get("ahi", "?")
                        cb = ui.checkbox(f"{mark}#{sid} | AHI={ahi} | {sev} "
                                         f"| quality={s.get('rec_quality', '?')}")
                        cb.value = cached
                        if cached:
                            selected.add(sid)
                        checkboxes[sid] = cb
                        cb.on_value_change(
                            lambda e, sid=sid: selected.add(sid) if e.value else selected.discard(sid)
                        )

        def toggle_all(val: bool):
            for sid, cb in checkboxes.items():
                cb.value = val

        async def download_selected():
            ids = list(selected)
            if not ids:
                ui.notify("Выберите хотя бы одно исследование", type="warning"); return
            download_btn.disable()
            progress.value = 0
            done = 0
            for i, sid in enumerate(ids):
                study = next(s for s in state.studies if s["study_id"] == sid)
                checksum = study.get("file_checksum", "")
                if not checksum:
                    log.push(f"#{sid}: нет checksum, пропуск"); continue
                if state.edf_cache.get(checksum):
                    log.push(f"#{sid}: уже в кэше"); done += 1; continue
                try:
                    log.push(f"#{sid}: скачивание...")
                    tmp = state.config.edf_dir / f"tmp_{sid}.edf"

                    def on_prog(cur, tot, sid_=sid):
                        pct = cur / tot if tot else 0
                        log.push(f"#{sid_}: {pct*100:.1f}%")

                    await asyncio.to_thread(state.api.download_edf, sid, tmp, None)
                    actual = EDFCache.sha256(tmp)
                    if actual != checksum:
                        tmp.unlink()
                        log.push(f"#{sid}: checksum mismatch!")
                        continue
                    state.edf_cache.store(tmp, checksum)
                    done += 1
                    log.push(f"#{sid}: OK ({tmp.stat().st_size if tmp.exists() else 0})")
                except Exception as e:
                    log.push(f"#{sid}: ERROR {e}")
                progress.value = (i + 1) / len(ids)
            status.set_text(f"Готово: {done}/{len(ids)}")
            download_btn.enable()

        with ui.row().classes("gap-2 mb-4"):
            ui.button("Загрузить список", on_click=load_studies)
            download_btn = ui.button("Скачать выбранные", on_click=download_selected,
                                     color="primary")
            download_btn.disable()

        table_area = ui.column().classes("w-full")