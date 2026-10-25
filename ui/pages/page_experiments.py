"""Запуск ML-эксперимента."""
import asyncio
import pandas as pd
from nicegui import ui
from ui.state import state
from ml.pipeline import ExperimentConfig, run_experiment
from ml.report import generate_report


def page_experiments():
    from ui.app import create_layout
    create_layout()

    with ui.column().classes("p-6 w-full"):
        ui.label("3. ML-эксперимент").classes("text-2xl font-bold mb-4")

        available = state.feature_store.list_available()
        epoch_lengths = sorted({l for _, l in available}) or [10]

        with ui.row().classes("gap-4 mb-2"):
            epoch_sel = ui.select(epoch_lengths, value=epoch_lengths[0],
                                  label="Длина эпохи (с)").classes("w-40")
            target_sel = ui.select(
                {"has_apnea": "Апноэ в эпохе (label)",
                 "ahi_ge_15": "Тяжесть AHI ≥ 15 (по пациенту)"},
                value="has_apnea", label="Целевая переменная",
            ).classes("w-72")
            norm_sel = ui.select(
                {"none": "Без нормализации",
                 "global": "Global z-score",
                 "patient": "Patient-z (тонич. baseline)"},
                value="patient", label="Нормализация",
            ).classes("w-72")

        ui.label("Модели:").classes("mt-2")
        with ui.row().classes("gap-4 mb-2"):
            model_cbs = {}
            for name in ("logreg", "rf", "xgb", "mlp"):
                model_cbs[name] = ui.checkbox(name, value=name in ("logreg", "rf"))

        with ui.expansion("Клинические фильтры (сопутствующие)", icon="filter").classes("w-full"):
            ui.label("ignore / only_positive / only_negative — по каждому признаку").classes("text-sm text-gray-600")
            filters = {}
            for label, col in [
                ("Гипертензия", "cvd_hypertension"),
                ("Сахарный диабет", "endocrine_diabetes"),
                ("ИБС", "cvd_ihd"),
                ("Инсомния", "som_insomnia"),
            ]:
                filters[col] = ui.select(
                    ["ignore", "only_positive", "only_negative"],
                    value="ignore", label=label,
                ).classes("w-48")

        run_btn = ui.button("Запустить эксперимент", color="primary").classes("mt-4")
        status = ui.label("Готов")
        progress = ui.linear_progress(value=0).classes("w-full")
        log = ui.log(max_lines=500).classes("w-full h-72 font-mono text-xs")

        async def run():
            run_btn.disable()
            progress.value = 0
            try:
                epoch_len = float(epoch_sel.value)

                df = state.feature_store.load_all(epoch_len)
                if df.empty:
                    ui.notify("Нет извлечённых данных для этой длины", type="negative")
                    return
                log.push(f"Загружено эпох: {len(df)}")

                # Метаданные (клиника + AHI)
                if not state.studies:
                    state.studies = await asyncio.to_thread(state.api.get_studies)
                if not state.clinical:
                    state.clinical = await asyncio.to_thread(state.api.get_clinical)

                clin = pd.DataFrame(state.clinical).drop_duplicates(
                    "patient_id", keep="last")
                clin_cols = ["patient_id", "cvd_hypertension", "endocrine_diabetes",
                             "cvd_ihd", "som_insomnia"]
                clin = clin[[c for c in clin_cols if c in clin.columns]]
                df = df.merge(clin, on="patient_id", how="left")

                st = pd.DataFrame(state.studies)[["study_id", "ahi"]]
                df = df.merge(st, on="study_id", how="left")
                log.push(f"После merge: {len(df)} строк")

                models = [n for n, cb in model_cbs.items() if cb.value]
                if not models:
                    ui.notify("Выберите хотя бы одну модель", type="warning")
                    return

                active_filters = {c: s.value for c, s in filters.items()
                                  if s.value != "ignore"}

                cfg = ExperimentConfig(
                    name=f"{target_sel.value}_len{int(epoch_len)}_{norm_sel.value}",
                    epoch_len=epoch_len,
                    models=models,
                    normalization=norm_sel.value,
                    target=target_sel.value,
                    comorbidity_filters=active_filters,
                )

                def progress_cb(msg):
                    log.push(msg)

                results = await asyncio.to_thread(
                    run_experiment, df, cfg, state.config.experiments_dir, progress_cb)

                # Отчёт
                report_path = state.config.experiments_dir / f"{cfg.name}_report.html"
                generate_report(results, report_path)

                # Итог
                ui.notify("Готово", type="positive")
                with ui.dialog() as dlg, ui.card():
                    ui.label("Результаты (test AUC):").classes("font-bold")
                    for m, r in results["models"].items():
                        if "auc" in r:
                            a = r["auc"]
                            ui.label(f"{m}: {a['auc']:.3f} "
                                     f"[{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
                    ui.label(f"Отчёт: {report_path}")
                    ui.button("OK", on_click=dlg.close)
                dlg.open()
            except Exception as e:
                import traceback
                log.push(f"FATAL: {e}\n{traceback.format_exc()}")
                ui.notify(f"Ошибка: {e}", type="negative")
            finally:
                run_btn.enable()
                progress.value = 1

        run_btn.on_click(run)