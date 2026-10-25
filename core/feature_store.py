"""Parquet-кэш извлечённых признаков.

Кэш-ключ: (study_id, epoch_len, t_pos, buffer_sec, version).
Разные параметры разметки → разные файлы, никакой путаницы.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from core.config import FEATURE_VERSION


class FeatureStore:
    def __init__(self, features_dir: Path):
        self.dir = Path(features_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, study_id: int, epoch_len: float,
              t_pos: float = 0.5, buffer_sec: float = 30.0) -> Path:
        t_str = f"{t_pos:.2f}".replace(".", "")
        b_str = f"{int(buffer_sec)}"
        return self.dir / (
            f"study_{study_id}_len{epoch_len}_t{t_str}_b{b_str}"
            f"_v{FEATURE_VERSION}.parquet"
        )

    def exists(self, study_id: int, epoch_len: float,
               t_pos: float = 0.5, buffer_sec: float = 30.0) -> bool:
        return self._path(study_id, epoch_len, t_pos, buffer_sec).exists()

    def load(self, study_id: int, epoch_len: float,
             t_pos: float = 0.5, buffer_sec: float = 30.0
             ) -> Optional[pd.DataFrame]:
        p = self._path(study_id, epoch_len, t_pos, buffer_sec)
        return pd.read_parquet(p) if p.exists() else None

    def save(self, df: pd.DataFrame, study_id: int, epoch_len: float,
             t_pos: float = 0.5, buffer_sec: float = 30.0) -> Path:
        p = self._path(study_id, epoch_len, t_pos, buffer_sec)
        df.to_parquet(p, index=False)
        return p

    def load_all(self, epoch_len: float,
                 study_ids: Optional[List[int]] = None,
                 t_pos: float = 0.5, buffer_sec: float = 30.0
                 ) -> pd.DataFrame:
        t_str = f"{t_pos:.2f}".replace(".", "")
        b_str = f"{int(buffer_sec)}"
        pattern = (f"study_*_len{epoch_len}_t{t_str}_b{b_str}"
                   f"_v{FEATURE_VERSION}.parquet")
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

    def list_available(self) -> List[Tuple[int, float, float, float]]:
        """Возвращает (study_id, epoch_len, t_pos, buffer_sec)."""
        out = []
        for p in self.dir.glob(f"study_*_v{FEATURE_VERSION}.parquet"):
            # формат: study_XXX_lenYY_t0ZZ_b30_vN.parquet
            parts = p.stem.split("_")
            try:
                sid = int(parts[1])
                length = float(parts[2].replace("len", ""))
                t_pos = float(parts[3][1:]) / 100 if len(parts[3]) > 2 else 0.5
                buffer_sec = float(parts[4][1:])
                out.append((sid, length, t_pos, buffer_sec))
            except (IndexError, ValueError):
                continue
        return out