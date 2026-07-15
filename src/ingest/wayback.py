"""Wayback Machine access: CDX index queries + snapshot fetches.

Wayback is the one source with no bot wall; still cached and rate-limited.
Used for: archived TwitchTracker leaderboards (validation series + roster),
archived language rankings (survivorship checks), archived press (provenance).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.parse import quote

from src.ingest.cache import FileCache, PoliteClient

CDX = "https://web.archive.org/cdx/search/cdx"


@dataclass
class Wayback:
    client: PoliteClient = field(
        default_factory=lambda: PoliteClient(cache=FileCache(namespace="wayback"))
    )

    def snapshots(
        self,
        url: str,
        *,
        from_date: str = "2020",
        to_date: str = "2023",
        collapse: str = "timestamp:6",  # one per month by default
        match_type: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """List archived captures of `url` as dicts with timestamp/statuscode."""
        q = (
            f"{CDX}?url={quote(url, safe='')}&output=json&fl=timestamp,statuscode,original"
            f"&from={from_date}&to={to_date}&collapse={collapse}&limit={limit}"
        )
        if match_type:
            q += f"&matchType={match_type}"
        record = self.client.get(q)
        rows = json.loads(record["text"]) if record["text"].strip() else []
        return [
            {"timestamp": r[0], "statuscode": r[1], "original": r[2]}
            for r in rows[1:]
            if r[1] in ("200", "-")
        ]

    def fetch(self, timestamp: str, url: str) -> str:
        """Fetch the archived page body for a capture (id_ = raw, no toolbar rewrite)."""
        record = self.client.get(f"https://web.archive.org/web/{timestamp}id_/{url}")
        return str(record["text"])
