"""Shared plumbing for source adapters: cached HTTP, provenance records, offline guard."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

console = Console(stderr=True)
USER_AGENT = "train-connectivity-sim/0.1 (+https://github.com/; research tool)"
# Credentials passed as query parameters (OpenCellID `token=` / `key=`) must never reach logs or error messages.
_SECRET_PARAM = re.compile(r"(?i)([?&](?:token|key|api_?key|access_token|password|secret)=)[^&#\s'\"]+")


class SourceUnavailable(RuntimeError):
    """Raised when a live source cannot be used (no key, offline, HTTP failure). Callers fall back."""


@dataclass
class Provenance:
    source: str
    url: str | None
    fetched_at: str
    version: str | None = None
    notes: str | None = None
    synthetic: bool = False

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def redact(text: object) -> str:
    """Mask credential query-parameter values in a URL, or in a message (e.g. a requests error) that quotes one."""
    return _SECRET_PARAM.sub(r"\1***", str(text))


def cache_path(raw_dir: Path, name: str, key: str, ext: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return raw_dir / name / f"{digest}.{ext}"


def http_get(url: str, *, raw_dir: Path, name: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
             data: str | dict | None = None, ext: str = "bin", timeout: int = 120, retries: int = 3, offline: bool = False, ttl_days: float = 30) -> Path:
    """GET/POST with on-disk cache under data/raw/<route>/<name>/. Returns the cached file path."""
    key = json.dumps({"url": url, "params": params, "data": data}, sort_keys=True)
    path = cache_path(raw_dir, name, key, ext)
    if path.exists() and (offline or (time.time() - path.stat().st_mtime) < ttl_days * 86400):
        return path
    if offline:
        raise SourceUnavailable(f"offline mode and no cached copy for {name}")
    headers = {"User-Agent": USER_AGENT, **(headers or {})}   # overpass-api.de rejects anonymous default agents (406)
    last_err: Exception | None = None
    for attempt in range(retries):
        r = None
        temporary = None
        try:
            console.log(f"[dim]{name}[/dim] fetching {redact(url)[:90]}…")
            if data is not None:
                r = requests.post(url, data=data, headers=headers, timeout=timeout)
            else:
                r = requests.get(url, params=params, headers=headers, timeout=timeout, stream=True)
            if r.status_code == 429:
                last_err = SourceUnavailable(f"{name}: HTTP 429 rate limit")
                time.sleep(5 * (attempt + 1))
                continue
            # 4xx other than rate limiting is permanent (e.g. the Ofcom checker has no UPRNs for a postcode):
            # retrying just burns the backoff budget, so fail fast and let the caller record the gap.
            if 400 <= r.status_code < 500:
                raise SourceUnavailable(f"{name}: HTTP {r.status_code} for {redact(url)}")
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            # Publish only complete downloads; preserve an older cache on failure.
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as fh:
                temporary = Path(fh.name)
                fh.writelines(r.iter_content(1 << 20))
            temporary.replace(path)
            return path
        except requests.RequestException as exc:  # pragma: no cover - network
            last_err = exc
            time.sleep(2 * (attempt + 1))
        finally:
            if r is not None:
                r.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise SourceUnavailable(f"{name}: {redact(last_err)}")
