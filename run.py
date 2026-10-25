"""Точка входа. Запуск: python run.py → http://localhost:8080"""
import logging

# ВАЖНО: логи в файл, чтобы видеть полные ошибки при отладке
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("apnea_ml.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

from ui.app import start_app

if __name__ == "__main__":
    start_app()