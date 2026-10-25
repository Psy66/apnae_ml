"""Безопасное логирование: и в UI-виджет, и в state.busy_log.

- state.busy_log — накопительный буфер (не теряется при переходах).
- self._widget.push — обновляет живой лог, если пользователь на этой странице.
  Если он ушёл со страницы — виджет удалён, push молча пропускается.
- logger.info — всегда пишет в файл apnea_ml.log.
"""
from __future__ import annotations
import logging
from datetime import datetime

logger = logging.getLogger("apnea_ml")


class SafeLog:
    def __init__(self, widget):
        self._widget = widget

    def push(self, msg: str) -> None:
        """Пишет в UI + буфер. В файл — только через вызывающий код
        (progress_cb → logger.info), чтобы не дублировать строки."""
        try:
            from ui.state import state
            ts = datetime.now().strftime("%H:%M:%S")
            state.append_busy_log(f"[{ts}] {msg}")
        except Exception:
            pass
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            self._widget.push(f"[{ts}] {msg}")
        except Exception:
            pass

    def restore_from_state(self) -> None:
        """При возврате на страницу — восстановить буфер в свежий виджет."""
        try:
            from ui.state import state
            for line in state.busy_log:
                self._widget.push(line)
        except Exception:
            pass


def safe_notify(msg: str, **kwargs):
    from nicegui import ui
    try:
        ui.notify(msg, **kwargs)
    except Exception:
        logger.warning(f"(notify not shown) {msg}")


def safe_set(target, attr: str, value):
    """Устанавливает атрибут UI-элемента, если он ещё существует.

    Пример: safe_set(progress, "value", 0.5) вместо progress.value = 0.5
    в фоновой задаче, которая может работать после ухода со страницы.
    """
    try:
        setattr(target, attr, value)
    except Exception:
        pass