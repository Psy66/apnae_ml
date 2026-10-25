"""Главный роутер NiceGUI."""
from __future__ import annotations
from nicegui import ui

from ui.state import state
from ui.pages.page_download import page_download
from ui.pages.page_extract import page_extract
from ui.pages.page_experiments import page_experiments
from ui.pages.page_results import page_results
from ui.pages.page_logs import page_logs
from ui.pages.page_settings import page_settings


def create_layout():
    with ui.header().classes("bg-blue-8 text-white items-center"):
        ui.label("APNEA-ML").classes("text-xl font-bold")
        ui.space()
        with ui.row().classes("gap-4"):
            ui.link("⚙ Settings", "/settings").classes("text-white no-underline")
            ui.link("1. Download", "/download").classes("text-white no-underline")
            ui.link("2. Extract", "/extract").classes("text-white no-underline")
            ui.link("3. Experiments", "/experiments").classes("text-white no-underline")
            ui.link("4. Results", "/results").classes("text-white no-underline")
            ui.link("📋 Logs", "/logs").classes("text-white no-underline")


@ui.page("/")
def index():
    ui.navigate.to("/download")


# Регистрация страниц
ui.page("/settings")(page_settings)
ui.page("/download")(page_download)
ui.page("/extract")(page_extract)
ui.page("/experiments")(page_experiments)
ui.page("/results")(page_results)
ui.page("/logs")(page_logs)


def start_app():
    state.init_api()
    ui.run(title="APNEA-ML", port=8080, reload=False,
           show=True, favicon="🧠", uvicorn_logging_level="warning")