"""Headed real-browser fetcher for Cloudflare-fronted pages (TwitchTracker, SullyGnome).

Policy (user-approved): drive a real, visible Chromium at human pace. We only
load robots-allowed HTML pages and let them render — we never call the sites'
/api endpoints ourselves. If a site challenges the real browser repeatedly,
we STOP (BlockedError) and surface it; we do not attempt to defeat challenges.

Rendered HTML is cached on disk so a page is never loaded twice across runs.
"""

from __future__ import annotations

import contextlib
import random
import time
from dataclasses import dataclass, field
from typing import Any

from src.ingest.cache import BlockedError, FileCache, RateLimiter, assert_allowed

CHALLENGE_MARKERS = ("just a moment", "challenges.cloudflare.com", "cf-chl")


def looks_like_challenge(html: str) -> bool:
    head = html[:4000].lower()
    return any(m in head for m in CHALLENGE_MARKERS)


@dataclass
class BrowserFetcher:
    cache: FileCache = field(default_factory=lambda: FileCache(namespace="rendered"))
    limiter: RateLimiter = field(default_factory=lambda: RateLimiter(min_delay=2.5, jitter=2.0))
    challenge_patience: float = 25.0  # seconds to let a challenge auto-resolve
    max_consecutive_challenges: int = 3
    _pw: Any = field(default=None, repr=False)
    _browser: Any = field(default=None, repr=False)
    _page: Any = field(default=None, repr=False)
    _consecutive_challenges: int = field(default=0, repr=False)

    def __enter__(self) -> BrowserFetcher:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        try:
            # real installed Chrome: a human browser fingerprint, headed
            self._browser = self._pw.chromium.launch(channel="chrome", headless=False)
        except Exception:
            self._browser = self._pw.chromium.launch(headless=False)
        context = self._browser.new_context(viewport={"width": 1360, "height": 900})
        self._page = context.new_page()
        return self

    def __exit__(self, *exc: object) -> None:
        for closer in (self._browser, self._pw):
            try:
                if closer is self._pw:
                    closer.stop()
                else:
                    closer.close()
            except Exception:
                pass

    def fetch(self, url: str, *, wait_selector: str | None = None, force: bool = False) -> str:
        """Return rendered HTML for `url`, from cache when available."""
        assert_allowed(url)
        if not force and (hit := self.cache.get(url)) is not None and hit["status"] == 200:
            return str(hit["text"])

        self.limiter.wait()
        self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        html = self._settle(wait_selector)

        if looks_like_challenge(html):
            # give Cloudflare's managed challenge a chance to clear on its own
            deadline = time.monotonic() + self.challenge_patience
            while time.monotonic() < deadline:
                time.sleep(2.0 + random.uniform(0, 1))
                html = self._page.content()
                if not looks_like_challenge(html):
                    break
            if looks_like_challenge(html):
                self._consecutive_challenges += 1
                if self._consecutive_challenges >= self.max_consecutive_challenges:
                    raise BlockedError(
                        f"{self._consecutive_challenges} consecutive Cloudflare challenges "
                        f"(last: {url}). Stopping per no-circumvention policy."
                    )
                return self.fetch(url, wait_selector=wait_selector, force=force)
            html = self._settle(wait_selector)

        self._consecutive_challenges = 0
        self.cache.put(url, 200, html)
        return html

    def _settle(self, wait_selector: str | None) -> str:
        with contextlib.suppress(Exception):  # busy pages never go idle
            self._page.wait_for_load_state("networkidle", timeout=20_000)
        if wait_selector:
            with contextlib.suppress(Exception):  # parser sees whatever rendered
                self._page.wait_for_selector(wait_selector, timeout=15_000)
        return str(self._page.content())
