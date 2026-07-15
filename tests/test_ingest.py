"""Scraper-infrastructure tests: parsers on fixtures, cache behavior, robots guard."""

from pathlib import Path

import pytest

from src.ingest.browser import looks_like_challenge
from src.ingest.cache import FileCache, RobotsDisallowedError, assert_allowed
from src.ingest.parsers import (
    _parse_month,
    parse_channel_subscribers_table,
    parse_ranking_page,
    parse_subscribers_leaderboard,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_channel_subscribers_table() -> None:
    rows = parse_channel_subscribers_table(
        (FIXTURES / "channel_subscribers_table.html").read_text()
    )
    assert len(rows) == 3
    may = rows[0]
    assert may["month"] == "2021-05"
    assert may["total"] == 12345
    assert may["prime"] == 2100
    assert may["tier1"] == 9000
    assert may["gifted"] == 3400
    july = rows[2]
    assert july["month"] == "2021-07"
    assert july["total"] is None  # '?' -> missing, not zero
    assert july["tier1"] == 13000


def test_parse_ranking_page_real_structure() -> None:
    rows = parse_ranking_page((FIXTURES / "ranking_page.html").read_text())
    assert len(rows) >= 4
    assert rows[0]["channel"] == "jynxzi"
    assert rows[0]["rank"] == 1
    channels = [r["channel"] for r in rows]
    assert channels == sorted(set(channels), key=channels.index)  # unique, ordered


def test_parse_subscribers_leaderboard_real_structure() -> None:
    rows = parse_subscribers_leaderboard((FIXTURES / "subscribers_leaderboard.html").read_text())
    assert len(rows) >= 4
    assert rows[0]["channel"] == "jynxzi"
    assert rows[0]["total_subs"] == 87355


def test_parse_month_variants() -> None:
    assert _parse_month("May 2021") == "2021-05"
    assert _parse_month("september 2020") == "2020-09"
    assert _parse_month("2021-5") == "2021-05"
    assert _parse_month("garbage") is None


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = FileCache(root=tmp_path, namespace="t")
    url = "https://example.org/x"
    assert cache.get(url) is None
    cache.put(url, 200, "<html>hi</html>")
    hit = cache.get(url)
    assert hit is not None and hit["status"] == 200 and "hi" in hit["text"]


def test_robots_guard() -> None:
    assert_allowed("https://twitchtracker.com/elraenn/subscribers")
    assert_allowed("https://twitchtracker.com/channels/ranking/turkish?page=2")
    with pytest.raises(RobotsDisallowedError):
        assert_allowed("https://twitchtracker.com/api/channels/summary/xqc")
    with pytest.raises(RobotsDisallowedError):
        assert_allowed("https://twitchtracker.com/u/someone")
    with pytest.raises(RobotsDisallowedError):
        assert_allowed("https://unknown-host.example/anything")


def test_challenge_detection() -> None:
    assert looks_like_challenge("<html><title>Just a moment...</title></html>")
    assert not looks_like_challenge("<html><title>Twitch Subscribers</title></html>")
