from nicegui import ui
from ui.state import state
from ui.components import help_expansion, section_header
from ui.help_content import HELP_SETTINGS
from core.api_client import APIClient


def page_settings():
    from ui.app import create_layout
    create_layout()
    with ui.column().classes("p-6 max-w-2xl mx-auto w-full"):
        section_header("⚙ Настройки подключения",
                       "API URL, токен и ограничение скорости загрузки")

        help_expansion("ℹ Инструкция: что здесь настроить и как проверить",
                       HELP_SETTINGS, icon="help")

        url_in = ui.input("API URL",
                          value=state.config.api_url).classes("w-full")
        token_in = ui.input("API Token", value=state.config.token,
                            password=True,
                            password_toggle_button=True).classes("w-full")
        speed_in = ui.number("Скорость загрузки (МБ/с)",
                             value=state.config.max_speed_mbps,
                             min=0.1, max=10, step=0.1).classes("w-full")

        def save():
            state.config.api_url = url_in.value
            state.config.token = token_in.value
            state.config.max_speed_mbps = float(speed_in.value)
            state.config.to_ini()
            ok = state.init_api()
            ui.notify("Сохранено" + ("" if ok else " (токен пуст)"),
                      type="positive" if ok else "warning")

        async def test():
            import asyncio
            if not token_in.value:
                ui.notify("Введите токен", type="negative"); return
            try:
                api = APIClient(url_in.value, token_in.value, max_speed_mbps=2)
                studies = await asyncio.to_thread(api.get_studies)
                ui.notify(f"✅ OK, {len(studies)} исследований", type="positive")
            except Exception as e:
                ui.notify(f"❌ Ошибка: {e}", type="negative")

        with ui.row().classes("mt-4 gap-2"):
            ui.button("💾 Сохранить", on_click=save).props("color=primary")
            ui.button("🔌 Проверить соединение", on_click=test).props("outline")