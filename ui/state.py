"""Глобальное состояние приложения (singleton)."""
from __future__ import annotations
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


state = AppState()