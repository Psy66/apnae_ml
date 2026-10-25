"""Список прошлых экспериментов + открыть HTML-отчёт."""
from pathlib import Path
from nicegui import ui
from ui.state import state
from ml.pipeline import list_experiments, load_experiment


def page_results():
    from ui.app import create_layout
    create_layout()
    with ui.column().classes("p-6 w-full"):
        ui.label("4. Результаты").classes("text-2xl font-bold mb-4")

        experiments = list_experiments(state.config.experiments_dir)
        if not experiments:
            ui.label("Экспериментов ещё нет").classes("text-gray-500")
            return

        for path in experiments:
            try:
                exp = load_experiment(path)
            except Exception as e:
                ui.label(f"Не удалось прочитать {path.name}: {e}").classes("text-red-600")
                continue
            cfg = exp["config"]
            models_summary = ", ".join(
                f"{m}={r['auc']['auc']:.3f}"
                for m, r in exp["models"].items() if "auc" in r
            ) or "—"
            with ui.card().classes("w-full mb-2"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column():
                        ui.label(f"{cfg['name']}  (len={cfg['epoch_len']}s)").classes("font-bold")
                        ui.label(f"{exp['timestamp'][:19]}  |  "
                                 f"target={cfg['target']}  |  norm={cfg['normalization']}  |  "
                                 f"test: {exp['n_test']} эпох / {exp['n_test_patients']} пац.").classes("text-sm text-gray-600")
                        ui.label(models_summary).classes("font-mono text-sm mt-1")
                    report = path.with_name(f"{cfg['name']}_report.html")
                    if report.exists():
                        ui.link("📄 Отчёт", target=f"file://{report.resolve()}",
                                new_tab=True).classes("text-blue-600")
                    else:
                        ui.label("(отчёт не найден)").classes("text-gray-400 text-sm")