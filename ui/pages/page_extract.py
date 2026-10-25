"""Батч-извлечение признаков из скачанных EDF."""
import asyncio
from nicegui import ui
from ui.state import state
from core.feature_extractor import FeatureExtractor


def page_extract():
    from ui.app import create_layout
    create_layout()
    with ui.column().classes("p-6 w-full"):
        ui.label("2. Извлечение признаков").classes("text-2xl font-bold mb-4")

        if not state.api:
            ui.label("⚠ Сначала настройте API").classes("text-red-600")
            return

        with ui.row().classes("gap-4 mb-4"):
            epoch_in = ui.number("Длина суб-эпохи (с)", value=10,
                                 min=2, max=90, step=1).classes("w-40")
            tpos_in = ui.number("Порог t_pos", value=0.5,
                                min=0.1, max=1.0, step=0.05).classes("w-32")
            buffer_in = ui.number("Буфер для is_tonic (с)", value=30,
                                  min=0, max=120, step=5).classes("w-40")

        with ui.row().classes("gap-2 mb-4"):
            load_btn = ui.button("Загрузить список исследований")
            run_btn = ui.button("Запустить извлечение", color="primary")
            run_btn.disable()
            force_cb = ui.checkbox("Перезаписать существующие", value=False)

        status = ui.label("Готов").classes("text-gray-600")
        progress = ui.linear_progress(value=0).classes("w-full")
        log = ui.log(max_lines=300).classes("w-full h-64 font-mono text-xs")

        async def load_list():
            status.set_text("Загрузка...")
            try:
                if not state.studies:
                    state.studies = await asyncio.to_thread(state.api.get_studies)
                status.set_text(f"Готово: {len(state.studies)} исследований")
                run_btn.enable()
            except Exception as e:
                status.set_text(f"Ошибка: {e}")

        async def run():
            if not state.studies:
                ui.notify("Сначала загрузите список", type="warning"); return
            run_btn.disable()
            epoch_len = float(epoch_in.value)
            t_pos = float(tpos_in.value)
            buffer_sec = float(buffer_in.value)
            extractor = FeatureExtractor(epoch_length=epoch_len,
                                         t_pos=t_pos, buffer_sec=buffer_sec)

            to_process = []
            for s in state.studies:
                edf = state.edf_cache.get(s.get("file_checksum", ""))
                if not edf:
                    continue
                if not force_cb.value and state.feature_store.exists(s["study_id"], epoch_len):
                    continue
                to_process.append((s, edf))

            log.push(f"К извлечению: {len(to_process)} исследований (len={epoch_len}s)")
            progress.value = 0

            for i, (study, edf) in enumerate(to_process):
                sid = study["study_id"]
                try:
                    events_raw = await asyncio.to_thread(
                        state.api.get_phasic_events, [sid])
                    events = [(float(e["event_onset"]), float(e["event_offset"]))
                              for e in events_raw]
                    result = await asyncio.to_thread(
                        extractor.extract, edf, sid, study["patient_id"],
                        events, study.get("hypnogram_data"),
                    )
                    state.feature_store.save(result.df, sid, epoch_len)
                    log.push(f"#{sid}: {result.n_epochs_labeled} эпох "
                             f"(из {result.n_epochs_n2n3} N2/N3, {result.n_epochs_total} всего)")
                except Exception as e:
                    log.push(f"#{sid}: ERROR {e}")
                progress.value = (i + 1) / max(1, len(to_process))

            status.set_text("Извлечение завершено")
            run_btn.enable()

        load_btn.on_click(load_list)
        run_btn.on_click(run)