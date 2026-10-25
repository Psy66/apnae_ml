"""Централизованная конфигурация. Читает/пишет config.ini, хранит константы."""
from __future__ import annotations
import configparser
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.ini")

# --- Спектральные диапазоны ---
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta":  (15.0, 30.0),
    "gamma": (30.0, 45.0),
}

# --- Извлечение признаков ---
FEATURE_VERSION = 1          # повышайте при изменении логики извлечения
STANDARD_CHANNELS = ["F3", "C3", "O1", "F4", "C4", "O2"]
TARGET_SR = 200.0
LOWCUT = 0.5
HIGHCUT = 45.0
USE_AVERAGE_REFERENCE = True
STAGES_OF_INTEREST = ("N2", "N3")
DEFAULT_BUFFER_SEC = 30.0

STAGE_NORMALIZE = {
    "W": "W", "Wake": "W", "0": "W",
    "1": "N1", "N1": "N1",
    "2": "N2", "N2": "N2",
    "3": "N3", "N3": "N3",
    "R": "R", "REM": "R",
}

# --- ML ---
DEFAULT_EPOCH_LENGTHS = [5, 10, 15, 20, 30]
DEFAULT_OVERLAP_T = 0.5
DEFAULT_RANDOM_STATE = 42


@dataclass
class AppConfig:
    api_url: str = "https://med.smed96.synology.me/psgapp/api/"
    token: str = ""
    cache_dir: Path = field(default_factory=lambda: Path("./data"))
    max_speed_mbps: float = 2.0
    max_parallel_downloads: int = 2
    max_retries: int = 5

    @property
    def edf_dir(self) -> Path: return self.cache_dir / "edf"

    @property
    def features_dir(self) -> Path: return self.cache_dir / "features"

    @property
    def experiments_dir(self) -> Path: return self.cache_dir / "experiments"

    def ensure_dirs(self):
        for d in (self.cache_dir, self.edf_dir, self.features_dir, self.experiments_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_ini(cls, path: Path = DEFAULT_CONFIG_PATH) -> "AppConfig":
        if not path.exists():
            cfg = cls()
            cfg.to_ini(path)
            return cfg
        cp = configparser.ConfigParser()
        cp.read(path, encoding="utf-8")
        return cls(
            api_url=cp.get("api", "url", fallback=cls.api_url),
            token=cp.get("api", "token", fallback=""),
            cache_dir=Path(cp.get("paths", "cache_dir", fallback="./data")),
            max_speed_mbps=cp.getfloat("limits", "max_speed_mbps", fallback=2.0),
            max_parallel_downloads=cp.getint("limits", "max_parallel_downloads", fallback=2),
            max_retries=cp.getint("limits", "max_retries", fallback=5),
        )

    def to_ini(self, path: Path = DEFAULT_CONFIG_PATH):
        cp = configparser.ConfigParser()
        cp["api"] = {"url": self.api_url, "token": self.token}
        cp["paths"] = {"cache_dir": str(self.cache_dir)}
        cp["limits"] = {
            "max_speed_mbps": str(self.max_speed_mbps),
            "max_parallel_downloads": str(self.max_parallel_downloads),
            "max_retries": str(self.max_retries),
        }
        with open(path, "w", encoding="utf-8") as f:
            cp.write(f)