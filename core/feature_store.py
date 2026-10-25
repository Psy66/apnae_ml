"""Parquet-кэш извлечённых признаков по (study_id, epoch_len, version)."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from core.config import FEATURE_VERSION


class FeatureStore:
    def __init__(self, features_dir: Path):
        self.dir = Path(features_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, study_id: int, epoch_len: float) -> Path:
        return self.dir / f"study_{study_id}_len{epoch_len}_v{FEATURE_VERSION}.parquet"

    def exists(self, study_id: int, epoch_len: float) -> bool:
        return self._path(study_id, epoch_len).exists()

    def load(self, study_id: int, epoch_len: float) -> Optional[pd.DataFrame]:
        p = self._path(study_id, epoch_len)
        return pd.read_parquet(p) if p.exists() else None

    def save(self, df: pd.DataFrame, study_id: int, epoch_len: float) -> Path:
        p = self._path(study_id, epoch_len)
        df.to_parquet(p, index=False)
        return p

    def load_all(self, epoch_len: float,
                 study_ids: Optional[List[int]] = None) -> pd.DataFrame:
        pattern = f"study_*_len{epoch_len}_v{FEATURE_VERSION}.parquet"
        dfs = []
        for p in self.dir.glob(pattern):
            if study_ids is not None:
                try:
                    sid = int(p.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if sid not in study_ids:
                    continue
            dfs.append(pd.read_parquet(p))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def list_available(self) -> List[Tuple[int, float]]:
        out = []
        for p in self.dir.glob(f"study_*_v{FEATURE_VERSION}.parquet"):
            parts = p.stem.split("_")
            try:
                out.append((int(parts[1]), float(parts[2].replace("len", ""))))
            except (IndexError, ValueError):
                continue
        return out