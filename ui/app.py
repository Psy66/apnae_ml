"""Главный роутер NiceGUI."""
from __future__ import annotations
from nicegui import ui, app as fastapi_app
from fastapi.responses import FileResponse, HTMLResponse

from ui.state import state
from ui.components import busy_badge, guarded_link      # ← новое
from ui.pages.page_download import page_download
from ui.pages.page_extract import page_extract
from ui.pages.page_experiments import page_experiments
from ui.pages.page_results import page_results
from ui.pages.page_logs import page_logs
from ui.pages.page_settings import page_settings
from ui.pages.page_about import page_about
from ui.pages.page_signal_viewer import page_signal_viewer


def create_layout():
    with ui.header().classes("bg-blue-8 text-white items-center"):
        ui.label("🧠 APNEA-ML").classes("text-xl font-bold")
        ui.space()
        busy_badge()
        ui.space()
        with ui.row().classes("gap-2"):
            guarded_link("ℹ О проекте", "/about")
            guarded_link("⚙ Settings", "/settings")
            guarded_link("1. Download", "/download")
            guarded_link("2. Extract", "/extract")
            guarded_link("3. Experiments", "/experiments")
            guarded_link("🔬 Сигналы", "/signals")
            guarded_link("4. Results", "/results")
            guarded_link("📋 Logs", "/logs")


@ui.page("/")
def index():
    ui.navigate.to("/about")


@fastapi_app.get("/report/{filename}")
def serve_report(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        return HTMLResponse("Bad request", status_code=400)
    path = state.config.experiments_dir / filename
    if path.exists() and path.is_file() and path.suffix == ".html":
        return FileResponse(str(path), media_type="text/html")
    return HTMLResponse(
        f"<h1>404</h1><p>Файл {filename} не найден в "
        f"{state.config.experiments_dir.resolve()}</p>",
        status_code=404,
    )


ui.page("/about")(page_about)
ui.page("/settings")(page_settings)
ui.page("/download")(page_download)
ui.page("/extract")(page_extract)
ui.page("/experiments")(page_experiments)
ui.page("/results")(page_results)
ui.page("/logs")(page_logs)
ui.page("/signals")(page_signal_viewer)

def start_app():
    state.init_api()
    ui.run(
        title="APNEA-ML",
        port=8080,
        reload=False,
        show=False,
        favicon="🧠",
        uvicorn_logging_level="warning",
        reconnect_timeout=120.0,
    )