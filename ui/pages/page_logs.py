"""Просмотр логов из logging."""
import logging
from pathlib import Path
from nicegui import ui
from ui.state import state


def page_logs():
    from ui.app import create_layout
    create_layout()

    log_path = Path("apnea_ml.log")
    with ui.column().classes("p-6 w-full"):
        ui.label("Логи").classes("text-2xl font-bold mb-4")
        ui.label(f"Файл: {log_path.absolute()}").classes("text-sm text-gray-600")
        area = ui.textarea(label="").classes("w-full h-96 font-mono text-xs")

        def refresh():
            if log_path.exists():
                content = log_path.read_text(encoding="utf-8", errors="ignore")
                # последние 200 строк
                area.value = "\n".join(content.splitlines()[-200:])
            else:
                area.value = "(лог пуст)"

        refresh()
        ui.button("Обновить", on_click=refresh)