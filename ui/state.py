"""Глобальное состояние приложения (singleton)."""
from __future__ import annotations
from datetime import datetime

from core.config import AppConfig
from core.api_client import APIClient
from core.edf_cache import EDFCache
from core.feature_store import FeatureStore


class AppState:
    def __init__(self):
        self.config = AppConfig.from_ini()
        self.config.ensure_dirs()
        self.api = None
        self.studies = []
        self.clinical = []
        self.patients = []
        self.edf_cache = EDFCache(self.config.edf_dir)
        self.feature_store = FeatureStore(self.config.features_dir)

        # --- Глобальное состояние длительной операции ---
        self.is_busy = False
        self.busy_op = ""            # "download" | "extract" | "experiment" | "comparison"
        self.busy_started_at = None  # datetime начала
        self.busy_log = []           # буфер последних 500 строк
        self.busy_progress = 0.0     # 0..1
        self.busy_message = ""       # например, "#607 (5/10)"

    def init_api(self) -> bool:
        if not self.config.token:
            return False
        self.api = APIClient(
            base_url=self.config.api_url,
            token=self.config.token,
            max_speed_mbps=self.config.max_speed_mbps,
            max_retries=self.config.max_retries,
        )
        return True

    # --- Операции ---
    def begin_busy(self, op: str, message: str = ""):
        self.is_busy = True
        self.busy_op = op
        self.busy_started_at = datetime.now()
        self.busy_log = []
        self.busy_progress = 0.0
        self.busy_message = message

    def end_busy(self):
        self.is_busy = False
        self.busy_op = ""
        self.busy_started_at = None
        self.busy_progress = 1.0

    def append_busy_log(self, line: str):
        self.busy_log.append(line)
        if len(self.busy_log) > 500:
            del self.busy_log[: len(self.busy_log) - 500]

    def set_busy_progress(self, value: float, message: str = ""):
        self.busy_progress = float(value)
        if message:
            self.busy_message = message


state = AppState()