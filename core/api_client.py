"""HTTP-клиент с client-side rate limiting (чтобы не бить сервер)."""
from __future__ import annotations
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class TokenBucket:
    """Ограничитель скорости: не более rate байт/сек."""
    def __init__(self, rate_bytes_per_sec: float):
        self.rate = rate_bytes_per_sec
        self.tokens = rate_bytes_per_sec
        self.last = time.monotonic()
        self.lock = Lock()

    def consume(self, n: int):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                need = (n - self.tokens) / self.rate
            time.sleep(need)


class APIClient:
    def __init__(self, base_url: str, token: str,
                 max_speed_mbps: float = 2.0, max_retries: int = 5,
                 timeout: int = 90, download_timeout: int = 900):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self.download_timeout = download_timeout
        self.bucket = TokenBucket(max_speed_mbps * 1024 * 1024)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _request(self, method, endpoint, params=None, stream=False, **kwargs):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(
                    method, url, params=params, stream=stream,
                    timeout=kwargs.get("timeout", self.timeout),
                )
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning(f"429 rate-limited, sleep {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as e:
                last_exc = e
                logger.warning(f"Request failed ({attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))
        raise IOError(f"All attempts failed: {last_exc}")

    def _paginated_get(self, endpoint, params=None, page_size=1000,
                       progress_callback: Optional[Callable] = None) -> List[Dict]:
        all_items, page = [], 1
        while True:
            p = dict(params or {})
            p.update({"limit": page_size, "page": page})
            resp = self._request("GET", endpoint, params=p)
            data = resp.json()
            items = data.get("data", [])
            all_items.extend(items)
            total_pages = data.get("meta", {}).get("pages", 1)
            if progress_callback:
                progress_callback(page, total_pages, len(items))
            if page >= total_pages:
                break
            page += 1
        return all_items

    # --- Эндпоинты ---
    def get_studies(self): return self._paginated_get("studies")
    def get_patients(self): return self._paginated_get("patients")
    def get_clinical(self): return self._paginated_get("clinical")

    def get_phasic_events(self, study_ids: List[int]) -> List[Dict]:
        params = {}
        for sid in study_ids:
            params.setdefault("study_id[]", []).append(sid)
        return self._paginated_get("phasic", params=params)

    def download_edf(self, study_id: int, dest: Path,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        resp = self._request("GET", f"studies/{study_id}/download",
                             stream=True, timeout=self.download_timeout)
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                self.bucket.consume(len(chunk))
                downloaded += len(chunk)
                if progress_callback and total:
                    progress_callback(downloaded, total)
        tmp.rename(dest)
        return dest