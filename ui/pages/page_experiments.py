"""Запуск ML-эксперимента + сравнение нормализаций + диагностика.

Особенности:
- Автоматический подбор доступных (epoch_len, t_pos, buffer_sec) из кэша.
- Проверка state.api до запуска.
- Совместимость со старым форматом feature_store (2-tuple) и новым (4-tuple).
- Кнопки блокируются вместе — нельзя запустить «Запустить» и «Сравнить»
  одновременно.
- Отчёт открывается через /report/{filename} (HTTP, не file://).
- Выбор лучшей нормализации — по CV AUC, не по test.
- Защита от перехода: state.begin_busy/end_busy + восстановление при возврате.
- Все UI-обновления в фоне обёрнуты в try/except.
- Keep-alive для WebSocket, чтобы не показывать «Connection loss» при
  долгом обучении RF/XGB.
"""
import asyncio
import traceback
from pathlib import Path

import pandas as pd
from nicegui import ui

from ui.state import state
from ui.components import help_expansion, section_header, warning_banner
from ui.logger import SafeLog, safe_notify, safe_set
from ui.help_content import HELP_EXPERIMENTS
from ml.pipeline import (
    ExperimentConfig,
    run_experiment,
    run_comparison,
)
from ml.report import generate_report


# ------------------------------------------------------------------ helpers
def _parse_available(available) -> list:
    """Универсальный парсер list_available().

    Поддерживает:
    - старый формат: [(sid, epoch_len), ...]
    - новый формат:  [(sid, epoch_len, t_pos, buffer_sec), ...]

    Возвращает [(sid, epoch_len, t_pos, buffer_sec), ...].
    """
    parsed = []
    for item in available:
        try:
            if len(item) == 2:
                sid, l = item
                parsed.append((int(sid), float(l), 0.5, 30.0))
            elif len(item) >= 4:
                sid, l, t, b = item[:4]
                parsed.append((int(sid), float(l), float(t), float(b)))
        except (ValueError, TypeError):
            continue
    return parsed


# ------------------------------------------------------------------ merge
async def _load_and_merge(epoch_len: float,
                          t_pos: float = 0.5,
                          buffer_sec: float = 30.0) -> pd.DataFrame:
    """Загружает признаки + приклеивает клинику и AHI."""
    if state.api is None:
        raise ValueError(
            "API не настроен. Перейдите на страницу ⚙ Settings, "
            "введите токен и нажмите «Сохранить», затем «Проверить соединение»."
        )

    # Совместимость: если feature_store старый (без t_pos/buffer) — вызываем
    # без них; если новый — с параметрами
    try:
        df = state.feature_store.load_all(
            epoch_len, t_pos=t_pos, buffer_sec=buffer_sec)
    except TypeError:
        df = state.feature_store.load_all(epoch_len)

    if df.empty:
        raise ValueError(
            f"Нет извлечённых данных для длины {epoch_len} с "
            f"(t_pos={t_pos:.2f}, buffer={buffer_sec:.0f} с). "
            f"Сначала выполните извлечение на шаге 2 с теми же параметрами."
        )

    if not state.studies:
        state.studies = await asyncio.to_thread(state.api.get_studies)
    if not state.clinical:
        state.clinical = await asyncio.to_thread(state.api.get_clinical)

    if state.clinical:
        clin = pd.DataFrame(state.clinical).drop_duplicates(
            "patient_id", keep="last")
        clin_cols = ["patient_id", "cvd_hypertension", "endocrine_diabetes",
                     "cvd_ihd", "som_insomnia"]
        clin = clin[[c for c in clin_cols if c in clin.columns]]
        df = df.merge(clin, on="patient_id", how="left")
    else:
        for c in ["cvd_hypertension", "endocrine_diabetes",
                  "cvd_ihd", "som_insomnia"]:
            df[c] = 0

    if state.studies:
        st = pd.DataFrame(state.studies)[["study_id", "ahi"]]
        df = df.merge(st, on="study_id", how="left")
    else:
        df["ahi"] = None
    return df


# ------------------------------------------------------------------ ui
def page_experiments():
    from ui.app import create_layout
    create_layout()

    with ui.column().classes("p-6 w-full"):
        section_header(
            "3. ML-эксперимент",
            "Обучение моделей и оценка качества классификации эпох",
        )

        help_expansion(
            "📖 Инструкция: параметры, минимум данных, план сравнения",
            HELP_EXPERIMENTS, icon="help",
        )

        # --- проверка API ---
        if state.api is None:
            warning_banner(
                "API не настроен. Перейдите в ⚙ Settings, введите токен "
                "и нажмите «Сохранить». Затем вернитесь сюда.",
                level="error",
            )
            return

        # --- диагностика доступных данных ---
        available_raw = state.feature_store.list_available()
        available = _parse_available(available_raw)
        epoch_lengths = sorted({l for _, l, _, _ in available}) or [10]
        n_studies_cached = len({sid for sid, _, _, _ in available})

        if n_studies_cached == 0:
            warning_banner(
                "Нет извлечённых признаков. Сначала выполните шаги 1 и 2.",
                level="warning")
        elif n_studies_cached < 10:
            warning_banner(
                f"⚠ ТОЛЬКО {n_studies_cached} пациентов. При split 80/20 "
                f"в test будет 1–2 пациента — AUC на таком test "
                f"статистически бессмыслен. Нужно ≥20 (пилот), лучше 50+.",
                level="error")
        elif n_studies_cached < 20:
            warning_banner(
                f"⚠ {n_studies_cached} пациентов. Хватит для проверки "
                f"работоспособности, но НЕ для научного вывода. "
                f"Для статьи — 50+.",
                level="warning")
        elif n_studies_cached < 50:
            warning_banner(
                f"В кэше {n_studies_cached} пациентов — пилотный уровень. "
                f"Для статьи желательно 50+.",
                level="info")

        # --- форма параметров ---
        with ui.row().classes("gap-4 mb-2 items-center"):
            epoch_sel = ui.select(
                epoch_lengths, value=epoch_lengths[0],
                label="Длина эпохи (с)").classes("w-36")

            combo_sel = ui.select(
                {}, value=None,
                label="Параметры разметки",
            ).classes("w-56")

            target_sel = ui.select(
                {"has_apnea": "Апноэ в эпохе (label)",
                 "ahi_ge_15": "Тяжесть AHI ≥ 15 (по пациенту)"},
                value="has_apnea", label="Целевая переменная",
            ).classes("w-72")
            norm_sel = ui.select(
                {"none": "Без нормализации",
                 "global": "Global z-score",
                 "patient": "Patient-z (рекомендуется)"},
                value="patient", label="Нормализация",
            ).classes("w-56")

        ui.label("Модели:").classes("mt-2 font-bold")
        with ui.row().classes("gap-4 mb-2"):
            model_cbs = {}
            for name in ("logreg", "rf", "xgb", "mlp"):
                model_cbs[name] = ui.checkbox(
                    name, value=name in ("logreg", "rf"))

        with ui.expansion("Клинические фильтры (для анализа подгрупп)",
                          icon="filter").classes("w-full"):
            ui.label(
                "ignore = не фильтровать | only_positive = только с болезнью | "
                "only_negative = только без болезни"
            ).classes("text-sm text-gray-600 mb-2")
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

        # --- кнопки ---
        with ui.row().classes("gap-3 mt-4 items-center"):
            run_btn = ui.button("▶ Запустить эксперимент", color="primary")
            compare_btn = ui.button(
                "🔬 Сравнить нормализации (none / global / patient)",
                color="secondary")

        status = ui.label("Готов").classes("text-gray-600")
        progress = ui.linear_progress(value=0).classes("w-full")
        log = SafeLog(ui.log(max_lines=1000).classes(
            "w-full h-72 font-mono text-xs"))

        # --------------------------------------------------- combo map
        combos_map: dict = {}

        def _refresh_combos():
            nonlocal combos_map
            try:
                epoch_len = float(epoch_sel.value)
            except (TypeError, ValueError):
                return
            combos = sorted({
                (t, b) for _, l, t, b in available if l == epoch_len
            })
            if not combos:
                combos = [(0.5, 30.0)]
            labels = [f"t_pos={t:.2f}, buffer={int(b)}s" for t, b in combos]
            combos_map = dict(zip(labels, combos))
            try:
                combo_sel.options = labels
                combo_sel.value = labels[0]
                combo_sel.update()
            except Exception:
                pass

        _refresh_combos()
        try:
            epoch_sel.on_value_change(lambda e: _refresh_combos())
        except Exception:
            pass

        # --------------------------------------------------- helpers
        def _collect_common():
            epoch_len = float(epoch_sel.value)
            combo_label = combo_sel.value
            t_pos, buffer_sec = combos_map.get(combo_label, (0.5, 30.0))
            models = [n for n, cb in model_cbs.items() if cb.value]
            active_filters = {c: s.value for c, s in filters.items()
                              if s.value != "ignore"}
            return epoch_len, t_pos, buffer_sec, models, active_filters

        def _disable_all_buttons():
            try:
                run_btn.disable()
                compare_btn.disable()
            except Exception:
                pass

        def _enable_all_buttons():
            try:
                run_btn.enable()
                compare_btn.enable()
            except Exception:
                pass

        def _show_results_dialog(results: dict, report_path: str):
            ok = [(m, r) for m, r in results["models"].items()
                  if "auc" in r and "error" not in r]
            err = [(m, r) for m, r in results["models"].items()
                   if "error" in r]
            diag = results.get("diagnostics", {})

            with ui.dialog() as dlg, ui.card().classes(
                    "min-w-[720px] max-w-[95vw]"):
                ui.label("Результаты эксперимента").classes(
                    "text-xl font-bold mb-2")

                warnings = diag.get("warnings", [])
                if warnings:
                    for w in warnings:
                        ui.label(w).classes(
                            "text-sm text-orange-700 bg-orange-50 p-2 "
                            "rounded mb-1 whitespace-pre-wrap")
                    ui.separator()

                if ok:
                    ui.label("Test AUC:").classes("font-bold mt-2")
                    for m, r in ok:
                        a = r["auc"]
                        cv = r.get("cv_auc_mean")
                        cv_str = (f" | CV={cv:.3f}"
                                  if cv is not None else " | CV=N/A")
                        color = ("text-green-700" if a["auc"] >= 0.85
                                 else "text-blue-700" if a["auc"] >= 0.7
                                 else "text-orange-700")
                        ui.label(
                            f"{m}: AUC={a['auc']:.3f} "
                            f"[{a['ci_low']:.3f}, {a['ci_high']:.3f}]"
                            f"{cv_str}"
                        ).classes(f"font-mono {color}")

                    best_m, best_r = max(
                        ok, key=lambda kv: kv[1]["auc"]["auc"])
                    per = best_r.get("per_patient_test", [])
                    if per:
                        with ui.expansion(
                            f"Подробно по test-пациентам (модель {best_m})",
                            icon="person",
                        ).classes("w-full mt-2"):
                            for p in per:
                                auc_v = (f"AUC={p['auc']:.3f}"
                                         if p["auc"] is not None
                                         else f"AUC=n/a ({p['note']})")
                                ui.label(
                                    f"pid={p['patient_id']} | "
                                    f"{p['n_positive']}/{p['n_epochs']} "
                                    f"positive ({p['prevalence']*100:.1f}%) "
                                    f"| {auc_v}"
                                ).classes("font-mono text-xs")

                    best_auc = best_r["auc"]["auc"]
                    best_ci_low = best_r["auc"]["ci_low"]
                    if best_auc >= 0.85 and best_ci_low >= 0.80:
                        ui.label(
                            "✅ AUC ≥ 0.85 и нижняя граница CI ≥ 0.80 — "
                            "модель хорошо разделяет классы. Для "
                            "клинического применения нужна независимая "
                            "валидация на большей выборке."
                        ).classes("text-green-700 font-bold mt-2 "
                                  "whitespace-pre-wrap")
                    elif best_auc >= 0.70:
                        ui.label(
                            "⚠ AUC 0.70–0.85 — сигнал есть, но для "
                            "клинического скрининга пока недостаточно. "
                            "Сравните CV AUC и test AUC: если CV сильно "
                            "ниже test — результат нестабильный."
                        ).classes("text-blue-700 mt-2 "
                                  "whitespace-pre-wrap")
                    else:
                        ui.label(
                            "⚠ AUC < 0.70 — сигнал слабый или его нет. "
                            "Причины смотрите в диагностике выше и в "
                            "разделе «Правила ML» на странице «О проекте»."
                        ).classes("text-orange-700 mt-2 "
                                  "whitespace-pre-wrap")

                if err:
                    ui.label(f"⚠ {len(err)} моделей упали:").classes(
                        "text-orange-600 font-bold mt-2")
                    for m, r in err:
                        ui.label(f"{m}: {r['error'][:250]}").classes(
                            "font-mono text-xs text-red-700")

                with ui.row().classes("mt-3"):
                    report_name = Path(report_path).name
                    ui.link("📄 Открыть полный отчёт",
                            target=f"/report/{report_name}",
                            new_tab=True).classes("text-blue-600")
                    ui.button("OK", on_click=dlg.close)
            dlg.open()

        # --------------------------------------------------- run
        async def run():
            if state.is_busy:
                safe_notify("Другая операция уже идёт", type="warning")
                return

            epoch_len, t_pos, buffer_sec, models, active_filters = \
                _collect_common()
            if not models:
                safe_notify("Выберите хотя бы одну модель", type="warning")
                return

            _disable_all_buttons()
            progress.value = 0
            state.begin_busy("experiment", "загрузка данных")
            try:
                log.push(
                    f"Параметры: len={epoch_len}s, "
                    f"t_pos={t_pos:.2f}, buffer={buffer_sec:.0f}s"
                )
                df = await _load_and_merge(epoch_len, t_pos, buffer_sec)
                log.push(f"После merge: {len(df)} строк")

                cfg = ExperimentConfig(
                    name=(f"{target_sel.value}_len{int(epoch_len)}_"
                          f"t{t_pos:.2f}_{norm_sel.value}"),
                    epoch_len=epoch_len,
                    models=models,
                    normalization=norm_sel.value,
                    target=target_sel.value,
                    overlap_t=t_pos,
                    comorbidity_filters=active_filters,
                )
                state.set_busy_progress(0.1, "обучение моделей")
                try:
                    status.set_text("Обучение моделей...")
                except Exception:
                    pass

                log.push("--- старт эксперимента ---")
                results = await asyncio.to_thread(
                    run_experiment, df, cfg,
                    state.config.experiments_dir,
                    lambda msg: log.push(msg))

                state.set_busy_progress(0.9, "формирование отчёта")
                report_path = (state.config.experiments_dir /
                               f"{cfg.name}_report.html")
                generate_report(results, str(report_path))
                state.set_busy_progress(1.0, "готово")

                try:
                    _show_results_dialog(results, str(report_path))
                except Exception as e:
                    # Пользователь мог уйти — диалог не показываем
                    log.push(f"(диалог не показан: {e})")
                safe_notify("Готово", type="positive")
            except Exception as e:
                log.push(f"FATAL: {e}\n{traceback.format_exc()}")
                safe_notify(f"Ошибка: {e}", type="negative")
            finally:
                _enable_all_buttons()
                state.end_busy()
                try:
                    progress.value = 1
                except Exception:
                    pass

        # --------------------------------------------------- compare
        async def compare():
            if state.is_busy:
                safe_notify("Другая операция уже идёт", type="warning")
                return

            epoch_len, t_pos, buffer_sec, models, active_filters = \
                _collect_common()
            if not models:
                safe_notify("Выберите хотя бы одну модель", type="warning")
                return

            _disable_all_buttons()
            progress.value = 0
            state.begin_busy("comparison", "сравнение нормализаций")
            try:
                log.push(
                    f"Сравнение нормализаций: len={epoch_len}s, "
                    f"t_pos={t_pos:.2f}, buffer={buffer_sec:.0f}s"
                )
                df = await _load_and_merge(epoch_len, t_pos, buffer_sec)
                log.push(f"После merge: {len(df)} строк")

                base = ExperimentConfig(
                    name=(f"COMPARE_base_len{int(epoch_len)}_"
                          f"t{t_pos:.2f}"),
                    epoch_len=epoch_len,
                    models=models,
                    normalization="patient",
                    target=target_sel.value,
                    overlap_t=t_pos,
                    comorbidity_filters=active_filters,
                )

                state.set_busy_progress(0.1, "3 прогона подряд")
                try:
                    status.set_text(
                        "Сравнение нормализаций (3 прогона)...")
                except Exception:
                    pass

                log.push("--- старт сравнения нормализаций ---")
                summary = await asyncio.to_thread(
                    run_comparison, df, base,
                    state.config.experiments_dir,
                    ("none", "global", "patient"),
                    lambda msg: log.push(msg))

                state.set_busy_progress(1.0, "готово")

                # --- Диалог сравнения ---
                try:
                    with ui.dialog() as dlg, ui.card().classes(
                            "min-w-[700px] max-w-[95vw]"):
                        ui.label("Сравнение нормализаций").classes(
                            "text-xl font-bold mb-2")
                        ui.label(
                            "Один и тот же split (same random_state). "
                            "Выбирайте по CV AUC, не по test!"
                        ).classes("text-sm text-gray-600 mb-3")

                        comp = summary["comparison"]
                        for m in models:
                            ui.label(f"— {m} —").classes("font-bold mt-2")
                            best_norm = None
                            best_cv = -1.0
                            for norm in ("none", "global", "patient"):
                                r = (comp.get(norm, {})
                                     .get("models", {})
                                     .get(m, {}))
                                auc_v = r.get("auc")
                                err = r.get("error")
                                if err:
                                    ui.label(
                                        f"  {norm}: ERROR {err[:80]}"
                                    ).classes(
                                        "font-mono text-xs text-red-600")
                                    continue
                                if auc_v is None:
                                    ui.label(f"  {norm}: n/a").classes(
                                        "font-mono text-xs")
                                    continue
                                cv = r.get("cv_auc_mean")
                                ci_low = r.get("ci_low")
                                ci_high = r.get("ci_high")
                                ci_str = (
                                    f"[{ci_low:.3f}, {ci_high:.3f}]"
                                    if (ci_low is not None
                                        and ci_high is not None) else ""
                                )
                                cv_str = (f" | CV={cv:.3f}"
                                          if cv is not None else "")
                                color = (
                                    "text-green-700" if auc_v >= 0.7
                                    else "text-orange-600" if auc_v < 0.55
                                    else "text-blue-700"
                                )
                                ui.label(
                                    f"  {norm}: AUC={auc_v:.3f} "
                                    f"{ci_str}{cv_str}"
                                ).classes(f"font-mono text-sm {color}")

                                # Выбор по CV, НЕ по test
                                if cv is not None and cv > best_cv:
                                    best_cv = cv
                                    best_norm = norm

                            if best_norm:
                                ui.label(
                                    f"  → лучший по CV: {best_norm} "
                                    f"(CV={best_cv:.3f})"
                                ).classes(
                                    "text-xs italic text-green-700")

                        ui.label(
                            f"Сводка: {summary['summary_path']}"
                        ).classes("text-xs text-gray-500 mt-3")
                        ui.button("OK", on_click=dlg.close)
                    dlg.open()
                except Exception as e:
                    log.push(f"(диалог не показан: {e})")

                safe_notify("Сравнение завершено", type="positive")
            except Exception as e:
                log.push(f"FATAL: {e}\n{traceback.format_exc()}")
                safe_notify(f"Ошибка: {e}", type="negative")
            finally:
                _enable_all_buttons()
                state.end_busy()
                try:
                    progress.value = 1
                except Exception:
                    pass

        # --------------------------------------------------- sync + keep-alive
        def _sync_from_state():
            try:
                if state.is_busy and state.busy_op in (
                        "experiment", "comparison"):
                    progress.value = state.busy_progress
                    if state.busy_message:
                        status.set_text(f"⚠ {state.busy_message}")
                    run_btn.disable()
                    compare_btn.disable()
                elif not state.is_busy:
                    if not run_btn.enabled:
                        run_btn.enable()
                    if not compare_btn.enabled:
                        compare_btn.enable()
            except Exception:
                pass

        def _keep_alive():
            # no-op обновление, чтобы WebSocket не рвался при долгом RF/XGB
            safe_set(status, "text", status.text)

        ui.timer(0.5, _sync_from_state)
        ui.timer(20.0, _keep_alive)

        # --- Восстановление после возврата на страницу ---
        if state.is_busy and state.busy_op in ("experiment", "comparison"):
            log.restore_from_state()
            try:
                status.set_text(
                    f"⚠ Идёт операция: {state.busy_message}")
                progress.value = state.busy_progress
                run_btn.disable()
                compare_btn.disable()
            except Exception:
                pass

        # --- Обработчики регистрируются РОВНО ОДИН РАЗ ---
        run_btn.on_click(run)
        compare_btn.on_click(compare)