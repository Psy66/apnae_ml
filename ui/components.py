"""Общие UI-компоненты: help-панели, warning-баннеры, guarded-ссылки."""
from __future__ import annotations
from nicegui import ui


def help_expansion(title: str, content_md: str, icon: str = "info",
                   expanded: bool = False):
    with ui.expansion(title, icon=icon, value=expanded).classes(
            "w-full mb-4 bg-blue-50 border-l-4 border-blue-400"):
        with ui.column().classes("p-2 w-full prose prose-sm max-w-none"):
            ui.markdown(content_md)


def warning_banner(message: str, level: str = "warning"):
    colors = {
        "info": "bg-blue-100 border-blue-500 text-blue-900",
        "warning": "bg-orange-100 border-orange-500 text-orange-900",
        "error": "bg-red-100 border-red-500 text-red-900",
    }
    icons = {"info": "ℹ", "warning": "⚠", "error": "⛔"}
    with ui.row().classes(
            f"{colors.get(level, colors['warning'])} "
            f"border-l-4 p-3 mb-3 rounded items-center gap-2 w-full"
    ):
        ui.label(icons.get(level, "ℹ")).classes("text-xl")
        ui.label(message).classes("text-sm flex-1 whitespace-pre-wrap")


def section_header(title: str, subtitle: str = ""):
    ui.label(title).classes("text-2xl font-bold mb-1")
    if subtitle:
        ui.label(subtitle).classes("text-sm text-gray-600 mb-4")


# --- Защита от перехода во время операции ---
_OP_LABELS = {
    "download": "скачивание EDF",
    "extract": "извлечение признаков",
    "experiment": "обучение модели",
    "comparison": "сравнение нормализаций",
}


def busy_badge():
    """Индикатор в хедере: виден с любой страницы, пока идёт операция."""
    from ui.state import state

    lbl = ui.label("").classes(
        "bg-orange-500 px-3 py-1 rounded text-white text-sm font-mono")
    lbl.set_visibility(False)

    def tick():
        try:
            if state.is_busy:
                pct = int(state.busy_progress * 100)
                op = _OP_LABELS.get(state.busy_op, state.busy_op)
                msg = f" ({state.busy_message})" if state.busy_message else ""
                lbl.set_text(f"⏳ {op} {pct}%{msg}")
                lbl.set_visibility(True)
            else:
                lbl.set_visibility(False)
        except Exception:
            pass

    ui.timer(0.5, tick)
    tick()


def guarded_link(text: str, target: str):
    """Кнопка-ссылка в хедере. Спрашивает подтверждение, если идёт операция."""
    from ui.state import state

    def click():
        if state.is_busy:
            op = _OP_LABELS.get(state.busy_op, state.busy_op)
            with ui.dialog() as dlg, ui.card().classes("min-w-[420px]"):
                ui.label(f"⚠ Идёт операция: {op}").classes(
                    "font-bold text-lg text-orange-700")
                if state.busy_message:
                    ui.label(f"Сейчас: {state.busy_message}").classes(
                        "text-sm text-gray-600")
                ui.label(
                    "Если вы уйдёте со страницы, операция продолжится в фоне. "
                    "Прогресс и лог сохранятся — вернётесь и увидите, что "
                    "происходит и чем закончилось."
                ).classes("text-sm mt-2")
                with ui.row().classes("mt-4 justify-end w-full"):
                    ui.button("Остаться", on_click=dlg.close).props("flat")
                    ui.button(
                        "Уйти",
                        color="primary",
                        on_click=lambda: (dlg.close(),
                                          ui.navigate.to(target)),
                    )
            dlg.open()
        else:
            ui.navigate.to(target)

    ui.button(text, on_click=click).props("flat no-caps").classes("text-white")