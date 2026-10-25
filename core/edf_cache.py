"""Локальный кэш EDF-файлов.

Два режима хранения:
- content-addressed по SHA256 (если checksum известен из API);
- by_study_id (fallback, если checksum недоступен).

SHA256 всё равно вычисляется после скачивания и сохраняется рядом
с файлом (.sha256) для воспроизводимости в статьях.
"""
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from typing import Optional


class EDFCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.by_study_dir = self.cache_dir / "by_study"
        self.by_study_dir.mkdir(exist_ok=True)

    # -------------------- content-addressed по SHA256 --------------------
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
            src.unlink(missing_ok=True)
            return dst
        shutil.move(str(src), str(dst))
        # сохраняем sha рядом (даже если пришёл из API — пригодится)
        (dst.with_suffix(".edf.sha256")).write_text(checksum)
        return dst

    # -------------------- by_study_id --------------------
    def get_by_study(self, study_id: int) -> Optional[Path]:
        p = self.by_study_dir / f"study_{study_id}.edf"
        return p if p.exists() else None

    def store_by_study(self, src: Path, study_id: int,
                       meta_checksum: str = "") -> Path:
        dst = self.by_study_dir / f"study_{study_id}.edf"
        if dst.exists():
            src.unlink(missing_ok=True)
            # перезаписываем sha на случай, если раньше не было
            if meta_checksum:
                (dst.with_suffix(".edf.sha256")).write_text(meta_checksum)
            return dst
        shutil.move(str(src), str(dst))
        sha = meta_checksum or self.sha256(dst)
        (dst.with_suffix(".edf.sha256")).write_text(sha)
        return dst

    # -------------------- универсальный API --------------------
    def get_any(self, study_id: int, checksum: str = "") -> Optional[Path]:
        """Вернёт файл, если он есть хотя бы в одном из кэшей."""
        if checksum:
            p = self.get(checksum)
            if p:
                return p
        return self.get_by_study(study_id)

    def store_any(self, src: Path, study_id: int,
                  checksum: str = "") -> Path:
        """Кладёт файл в кэш; при наличии checksum — в content-addressed."""
        if checksum:
            return self.store(src, checksum)
        return self.store_by_study(src, study_id)

    def invalidate_study(self, study_id: int) -> bool:
        p = self.get_by_study(study_id)
        if p and p.exists():
            p.unlink()
            (p.with_suffix(".edf.sha256")).unlink(missing_ok=True)
            return True
        return False

    def stats(self) -> dict:
        n_by_study = len(list(self.by_study_dir.glob("*.edf")))
        n_by_sha = sum(1 for _ in self.cache_dir.glob("*/*.edf")
                       if _.parent.name != "by_study")
        return {"by_study": n_by_study, "by_checksum": n_by_sha}

    @staticmethod
    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()