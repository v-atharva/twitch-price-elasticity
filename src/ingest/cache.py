"""Polite HTTP plumbing shared by every scraper.

Three invariants, enforced here so no scraper can violate them:
1. Nothing is fetched twice: every response body lands in an on-disk cache
   keyed by URL (+ namespace), and re-runs hit the cache first.
2. Nothing is fetched fast: a global rate limiter enforces a minimum delay
   with jitter between live requests, with exponential backoff on errors.
3. Nothing disallowed is fetched: robots rules for the sites we touch are
   encoded and checked on every request.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

DEFAULT_CACHE_DIR = Path("data/cache")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# robots.txt rules verified 2026-07-15 (see PLAN.md). Prefix match on path.
ROBOTS_DISALLOW: dict[str, tuple[str, ...]] = {
    "twitchtracker.com": ("/cdn-cgi/", "/api/", "/u/"),
    "www.twitchtracker.com": ("/cdn-cgi/", "/api/", "/u/"),
    "sullygnome.com": (),
    "web.archive.org": (),
    "api.frankfurter.dev": (),
}


class RobotsDisallowedError(RuntimeError):
    pass


class BlockedError(RuntimeError):
    """Source appears to be refusing us (repeated challenges) — stop, don't circumvent."""


def assert_allowed(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    rules = ROBOTS_DISALLOW.get(host)
    if rules is None:
        raise RobotsDisallowedError(f"no robots policy encoded for host {host!r}; add it first")
    path = parsed.path or "/"
    if "/search" in path and host.endswith("twitchtracker.com") and parsed.query:
        raise RobotsDisallowedError(f"robots disallows {url}")
    for prefix in rules:
        if path.startswith(prefix):
            raise RobotsDisallowedError(f"robots disallows {url}")


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


@dataclass
class FileCache:
    root: Path = DEFAULT_CACHE_DIR
    namespace: str = "http"

    def _path(self, url: str) -> Path:
        key = cache_key(url)
        return self.root / self.namespace / key[:2] / f"{key}.json"

    def get(self, url: str) -> dict | None:
        p = self._path(url)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def put(self, url: str, status: int, text: str, meta: dict | None = None) -> dict:
        record = {
            "url": url,
            "status": status,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "text": text,
            "meta": meta or {},
        }
        p = self._path(url)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record))
        return record


@dataclass
class RateLimiter:
    min_delay: float = 2.0
    jitter: float = 1.5
    _last: float = field(default=0.0, repr=False)

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        target = self.min_delay + random.uniform(0, self.jitter)
        if elapsed < target:
            time.sleep(target - elapsed)
        self._last = time.monotonic()


@dataclass
class PoliteClient:
    """Cached, rate-limited, robots-checked GET for plain-HTTP sources (Wayback, APIs)."""

    cache: FileCache = field(default_factory=FileCache)
    limiter: RateLimiter = field(default_factory=RateLimiter)
    max_retries: int = 4
    timeout: float = 60.0

    def get(self, url: str, *, force: bool = False) -> dict:
        assert_allowed(url)
        if not force and (hit := self.cache.get(url)) is not None:
            return hit
        delay = 2.0
        last_status = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                resp = httpx.get(
                    url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                last_status = resp.status_code
                if resp.status_code == 200:
                    return self.cache.put(url, 200, resp.text)
                if resp.status_code == 404:
                    return self.cache.put(url, 404, "")
            except httpx.HTTPError:
                last_status = -1
            time.sleep(delay * (2**attempt) + random.uniform(0, 1))
        raise BlockedError(
            f"giving up on {url} after {self.max_retries} tries (last={last_status})"
        )
