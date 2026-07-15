"""Pure HTML -> records parsers for TwitchTracker pages.

No network access here: every function takes HTML text and returns plain
records, so the whole module is unit-testable on fixture files.
"""

from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup

_NUM = re.compile(r"[-+]?[\d,.]+")


def _num(text: str) -> float | None:
    """'87,355' -> 87355.0; '?' / '' -> None."""
    m = _NUM.search(text.replace(" ", " "))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def parse_ranking_page(html: str) -> list[dict[str, Any]]:
    """TwitchTracker /channels/ranking[/language] page -> ranked channel rows."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for tr in soup.select("table tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        link = tr.find("a", href=re.compile(r"^/[a-z0-9_]+$", re.I))
        if link is None:
            continue
        rank_text = cells[0].get_text(strip=True).lstrip("#")
        out.append(
            {
                "rank": int(rank_text) if rank_text.isdigit() else None,
                "channel": str(link["href"]).lstrip("/").lower(),
                "display_name": link.get_text(strip=True),
            }
        )
    return out


def parse_subscribers_leaderboard(html: str) -> list[dict[str, Any]]:
    """TwitchTracker /subscribers leaderboard (live or archived) -> rows.

    Columns observed 2020-2026: rank, trend, [badge], channel, total, prime,
    tier1(+), gifted, ... — header names vary slightly across years, so we
    map by position of the channel link and parse numerics defensively.
    """
    soup = BeautifulSoup(html, "lxml")
    out = []
    for tr in soup.select("table tr"):
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue
        link = tr.find(
            "a",
            href=re.compile(
                r"^(/web/\d+(?:id_)?/https?://twitchtracker\.com)?/[a-z0-9_]+(/subscribers)?$", re.I
            ),
        )
        if link is None:
            continue
        href = str(link["href"])
        channel = href.rstrip("/").split("/")[-1]
        if channel == "subscribers":
            channel = href.rstrip("/").split("/")[-2]
        nums = [_num(c.get_text(strip=True)) for c in cells]
        numeric = [n for n in nums[1:] if n is not None]  # skip the rank cell
        out.append(
            {
                "channel": channel.lower(),
                "total_subs": numeric[0] if numeric else None,
                "raw_numbers": nums,
            }
        )
    return out


def parse_channel_subscribers_table(html: str) -> list[dict[str, Any]]:
    """TwitchTracker /<channel>/subscribers monthly table -> one record per month.

    Table headers: Month | Total | Prime | Tier 1 | Tier 2 | Tier 3 | Unlisted | Gifted
    (rows are populated client-side, so this must be given RENDERED html).
    """
    soup = BeautifulSoup(html, "lxml")
    target = None
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if headers and headers[0].startswith("month") and any("prime" in h for h in headers):
            target = (table, headers)
            break
    if target is None:
        return []
    table, headers = target
    keys = []
    for h in headers:
        h = h.split("(")[0].strip()
        keys.append(
            {
                "month": "month", "total": "total", "prime": "prime", "tier 1": "tier1",
                "tier 2": "tier2", "tier 3": "tier3", "unlisted": "unlisted", "gifted": "gifted",
            }.get(h, h.replace(" ", "_"))
        )  # fmt: skip
    out = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) != len(keys):
            continue
        rec: dict[str, Any] = {}
        for key, cell in zip(keys, cells, strict=True):
            text = cell.get_text(strip=True)
            rec[key] = _parse_month(text) if key == "month" else _num(text)
        if rec.get("month"):
            out.append(rec)
    return out


_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}


def _parse_month(text: str) -> str | None:
    """'May 2021' / '2021-05' -> '2021-05'."""
    text = text.strip().lower()
    m = re.match(r"([a-z]{3})[a-z]*\s+(\d{4})", text)
    if m and m.group(1) in _MONTHS:
        return f"{int(m.group(2)):04d}-{_MONTHS[m.group(1)]:02d}"
    m = re.match(r"(\d{4})-(\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    return None


def parse_channel_meta(html: str) -> dict[str, Any]:
    """TwitchTracker channel page -> {language, country_hint}."""
    soup = BeautifulSoup(html, "lxml")
    language = None
    lang_link = soup.find("a", href=re.compile(r"/languages/", re.I))
    if lang_link:
        language = lang_link.get_text(strip=True).lower()
    text = soup.get_text(" ", strip=True)
    created = None
    m = re.search(r"(?:created|joined)\s*[:\-]?\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})", text, re.I)
    if m:
        created = m.group(1)
    return {"language": language, "created_hint": created}
