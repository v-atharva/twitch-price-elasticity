"""Build the channel roster from TwitchTracker language rankings.

Selection rule (documented, config-driven): top-N channels per broadcast
language from /channels/ranking/<language>, pages 1..P. This over-samples
currently-large channels — survivorship and top-channel selection bias are
acknowledged limitations, partially quantified later against archived
(2021-era) ranking snapshots where Wayback has them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.ingest.browser import BrowserFetcher
from src.ingest.parsers import parse_ranking_page

RANKING_URL = "https://twitchtracker.com/channels/ranking/{language}?page={page}"
# TwitchTracker uses full language names in URLs
LANGUAGE_SLUG = {
    "tr": "turkish",
    "pt": "portuguese",
    "es": "spanish",
    "en": "english",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "pl": "polish",
    "th": "thai",
    "ko": "korean",
    "ja": "japanese",
}


def build_roster(fetcher: BrowserFetcher, languages: dict[str, int], pages: int) -> pd.DataFrame:
    frames = []
    for lang, top_n in languages.items():
        slug = LANGUAGE_SLUG[lang]
        rows: list[dict] = []
        for page in range(1, pages + 1):
            if len(rows) >= top_n:
                break
            url = RANKING_URL.format(language=slug, page=page)
            html = fetcher.fetch(url, wait_selector="table")
            parsed = parse_ranking_page(html)
            for r in parsed:
                r["language"] = lang
                r["source_url"] = url
            rows.extend(parsed)
            if not parsed:  # ran out of pages
                break
        frames.append(pd.DataFrame(rows[:top_n]))
        print(f"roster: {lang} -> {min(len(rows), top_n)} channels")
    roster = pd.concat(frames, ignore_index=True)
    roster = roster.drop_duplicates(subset="channel", keep="first")
    return roster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/study.yaml")
    parser.add_argument("--out", default="data/interim/roster.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    pilot = config.raw["scrape"]["pilot"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with BrowserFetcher() as fetcher:
        roster = build_roster(
            fetcher,
            languages=dict(pilot["languages"]),
            pages=int(pilot["ranking_pages_per_language"]),
        )
    roster.to_csv(out, index=False)
    print(f"wrote {out}: {len(roster)} unique channels")


if __name__ == "__main__":
    main()
