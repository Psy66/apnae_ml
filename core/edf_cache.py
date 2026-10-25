"""Content-addressed кэш EDF-файлов по SHA256."""
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from typing import Optional


class EDFCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, checksum: str) -> Path:
        prefix = checksum[:2]
        d = self.cache_dir / prefix
        d.mkdir(exist_ok=True)
        return d / f"{checksum}.edf"

    def get(self, checksum: str) -> Optional[Path]:
        if not checksum:
            return None
        p = self._path(checksum)
        return p if p.exists() else None

    def store(self, src: Path, checksum: str) -> Path:
        dst = self._path(checksum)
        if dst.exists():
            src.unlink()
            return dst
        shutil.move(str(src), str(dst))
        return dst

    @staticmethod
    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()