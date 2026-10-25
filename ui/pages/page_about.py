"""Страница 'О проекте' — обучающая программа + глоссарий + правила ML + шаблоны статьи."""
from nicegui import ui
from ui.help_content import (
    ABOUT_MAIN,
    ABOUT_GLOSSARY,
    ABOUT_ML_RULES,
    ABOUT_PAPER,
)


def page_about():
    from ui.app import create_layout
    create_layout()

    with ui.column().classes("p-6 max-w-5xl mx-auto w-full"):
        ui.label("📚 Учебный курс APNEA-ML").classes(
            "text-3xl font-bold mb-2 text-blue-800")
        ui.label(
            "Полный цикл ML-исследования: от сырых ПСГ до публикации статьи"
        ).classes("text-lg text-gray-600 mb-6")

        # Навигация по разделам
        with ui.tabs().classes("w-full") as tabs:
            tab_main = ui.tab("🎯 О проекте")
            tab_glossary = ui.tab("📖 Глоссарий")
            tab_rules = ui.tab("⚖️ Правила ML")
            tab_paper = ui.tab("📝 Как писать статью")

        with ui.tab_panels(tabs, value=tab_main).classes("w-full"):
            with ui.tab_panel(tab_main):
                with ui.column().classes("prose prose-base max-w-none w-full"):
                    ui.markdown(ABOUT_MAIN)

            with ui.tab_panel(tab_glossary):
                with ui.column().classes("prose prose-base max-w-none w-full"):
                    ui.markdown(ABOUT_GLOSSARY)

            with ui.tab_panel(tab_rules):
                with ui.column().classes("prose prose-base max-w-none w-full"):
                    ui.markdown(ABOUT_ML_RULES)

            with ui.tab_panel(tab_paper):
                with ui.column().classes("prose prose-base max-w-none w-full"):
                    ui.markdown(ABOUT_PAPER)