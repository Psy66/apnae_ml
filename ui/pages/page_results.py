"""Список прошлых экспериментов + открыть HTML-отчёт."""
from pathlib import Path

from nicegui import ui

from ui.state import state
from ui.components import help_expansion, section_header
from ui.logger import safe_notify
from ui.help_content import HELP_RESULTS
from ml.pipeline import list_experiments, load_experiment


def page_results():
    from ui.app import create_layout
    create_layout()

    with ui.column().classes("p-6 w-full"):
        section_header("4. Результаты экспериментов",
                       "Список прогонов, сводные AUC и HTML-отчёты")

        help_expansion("ℹ Как читать результаты (AUC, CI, CV, SHAP)",
                       HELP_RESULTS, icon="help")

        experiments = list_experiments(state.config.experiments_dir)
        if not experiments:
            ui.label("Экспериментов ещё нет. Запустите первый на странице "
                     "«3. Experiments».").classes("text-gray-500 text-lg")
            return

        ui.button("🔄 Обновить список",
                  on_click=lambda: ui.navigate.to("/results")).props("dense")

        # Контейнер, чтобы можно было перерисовывать
        cards_area = ui.column().classes("w-full")

        def _regen_report(json_path: Path):
            """Перегенерировать HTML из существующего JSON."""
            try:
                from ml.report import generate_report
                exp = load_experiment(json_path)
                html_path = json_path.with_suffix(".html")
                generate_report(exp, html_path)
                safe_notify(f"Отчёт перегенерирован: {html_path.name}",
                            type="positive")
            except Exception as e:
                safe_notify(f"Ошибка: {e}", type="negative")

        def _render_cards():
            cards_area.clear()
            with cards_area:
                for path in experiments:
                    try:
                        exp = load_experiment(path)
                    except Exception as e:
                        ui.label(f"Не удалось прочитать {path.name}: {e}"
                                 ).classes("text-red-600")
                        continue

                    cfg = exp["config"]
                    ok_models = {m: r for m, r in exp["models"].items()
                                 if "auc" in r and "error" not in r}
                    err_models = {m: r for m, r in exp["models"].items()
                                  if "error" in r}

                    with ui.card().classes("w-full mb-2"):
                        with ui.row().classes(
                                "w-full items-start justify-between gap-4"):
                            with ui.column().classes("flex-1"):
                                ui.label(
                                    f"{cfg['name']}  (len={cfg['epoch_len']}s)"
                                ).classes("font-bold text-lg")
                                ui.label(
                                    f"{exp['timestamp'][:19]}  |  "
                                    f"target={cfg['target']}  |  "
                                    f"norm={cfg['normalization']}  |  "
                                    f"train: {exp['n_train_patients']} пац. / "
                                    f"{exp['n_train']} эпох  |  "
                                    f"test: {exp['n_test_patients']} пац. / "
                                    f"{exp['n_test']} эпох"
                                ).classes("text-sm text-gray-600")

                                if ok_models:
                                    for m_name, m in ok_models.items():
                                        a = m["auc"]
                                        cv = m.get("cv_auc_mean")
                                        cv_str = (f", CV={cv:.3f}"
                                                  if cv is not None else "")
                                        color = ("text-green-700"
                                                 if a["auc"] >= 0.85
                                                 else "text-blue-700"
                                                 if a["auc"] >= 0.7
                                                 else "text-orange-700")
                                        ui.label(
                                            f"  {m_name}: AUC={a['auc']:.3f} "
                                            f"[{a['ci_low']:.3f}, "
                                            f"{a['ci_high']:.3f}]"
                                            f"{cv_str}"
                                        ).classes(
                                            f"font-mono text-sm {color}")
                                if err_models:
                                    ui.label(
                                        f"  ⚠ {len(err_models)} моделей упали: "
                                        f"{', '.join(err_models.keys())}"
                                    ).classes("text-sm text-orange-600")

                            # Кнопки справа
                            with ui.column().classes("items-end gap-2"):
                                # Отчёт ищем как <stem>.html (уникальное имя)
                                report = path.with_suffix(".html")
                                if report.exists():
                                    ui.link("📄 Отчёт",
                                            target=f"/report/{report.name}",
                                            new_tab=True).classes(
                                        "text-blue-600 font-bold text-lg")
                                else:
                                    ui.label("(отчёт не создан)").classes(
                                        "text-gray-400 text-sm")
                                ui.button(
                                    "🔄 Перегенерировать отчёт",
                                    on_click=lambda p=path: _regen_report(p),
                                ).props("flat dense")

                        if err_models:
                            with ui.expansion("Показать ошибки моделей",
                                              icon="error").classes("w-full"):
                                for m_name, r in err_models.items():
                                    ui.label(
                                        f"{m_name}: {r['error']}"
                                    ).classes(
                                        "text-xs font-mono text-red-700 "
                                        "whitespace-pre-wrap")

        _render_cards()