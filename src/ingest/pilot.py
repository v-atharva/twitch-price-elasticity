"""Pilot pull: per-channel monthly subscriber tables for the roster.

For each roster channel, load (rendered) twitchtracker.com/<channel>/subscribers
and parse the monthly tier table. Resumable: page-level HTML is cached by the
fetcher; channels already parsed are skipped via the output manifest.

Ends by printing the coverage report that feeds the M4 decision gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.ingest.browser import BrowserFetcher
from src.ingest.cache import BlockedError
from src.ingest.parsers import parse_channel_meta, parse_channel_subscribers_table
from src.panel.months import to_mindex

CHANNEL_URL = "https://twitchtracker.com/{channel}"
SUBS_URL = "https://twitchtracker.com/{channel}/subscribers"


def scrape_channel(fetcher: BrowserFetcher, channel: str) -> dict:
    meta_html = fetcher.fetch(CHANNEL_URL.format(channel=channel))
    meta = parse_channel_meta(meta_html)
    subs_html = fetcher.fetch(SUBS_URL.format(channel=channel), wait_selector="table tbody tr")
    months = parse_channel_subscribers_table(subs_html)
    return {"channel": channel, "meta": meta, "months": months}


def coverage_report(records: list[dict], roster: pd.DataFrame, treat_mindex: int) -> pd.DataFrame:
    rows = []
    lang_by_channel = dict(zip(roster["channel"], roster["language"], strict=False))
    for rec in records:
        months = rec["months"]
        mindexes = [to_mindex(m["month"]) for m in months if m.get("month")]
        pre = sum(1 for i in mindexes if i < treat_mindex)
        post = sum(1 for i in mindexes if i >= treat_mindex)
        rows.append(
            {
                "channel": rec["channel"],
                "language": lang_by_channel.get(rec["channel"], "?"),
                "n_months": len(mindexes),
                "pre_months": pre,
                "post_months": post,
                "usable_6_6": pre >= 6 and post >= 6,
            }
        )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("language")
        .agg(
            n_channels=("channel", "count"),
            with_any_data=("n_months", lambda s: int((s > 0).sum())),
            median_months=("n_months", "median"),
            usable_6_6=("usable_6_6", "sum"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/study.yaml")
    parser.add_argument("--roster", default="data/interim/roster.csv")
    parser.add_argument("--out-dir", default="data/interim/pilot")
    parser.add_argument("--limit", type=int, default=None, help="scrape at most N channels")
    args = parser.parse_args()

    config = load_config(args.config)
    roster = pd.read_csv(args.roster)
    out_dir = Path(args.out_dir)
    (out_dir / "channels").mkdir(parents=True, exist_ok=True)

    todo = [
        ch
        for ch in roster["channel"].tolist()
        if not (out_dir / "channels" / f"{ch}.json").exists()
    ]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"pilot: {len(todo)} channels to scrape ({len(roster) - len(todo)} already done)")

    scraped = 0
    try:
        with BrowserFetcher() as fetcher:
            for channel in todo:
                try:
                    rec = scrape_channel(fetcher, channel)
                except BlockedError:
                    raise
                except Exception as e:  # keep going on per-channel parse/nav failures
                    rec = {"channel": channel, "meta": {}, "months": [], "error": str(e)}
                (out_dir / "channels" / f"{channel}.json").write_text(json.dumps(rec))
                scraped += 1
                if scraped % 10 == 0:
                    print(f"  {scraped}/{len(todo)} scraped")
    except BlockedError as e:
        print(f"STOPPED (blocked): {e}")
        print("Partial results are cached; re-run to resume once appropriate.")

    records = [json.loads(p.read_text()) for p in sorted((out_dir / "channels").glob("*.json"))]
    flat = [{"channel": r["channel"], **m} for r in records for m in r["months"]]
    if flat:
        pd.DataFrame(flat).to_parquet(out_dir / "pilot_subscribers.parquet", index=False)
    # earliest treatment month in the study (TR/MX cohort) as the pre/post split
    treat_mindex = min(c.treat_mindex for c in config.cohorts)
    summary = coverage_report(records, roster, treat_mindex)
    summary.to_csv(out_dir / "coverage_report.csv", index=False)
    print("\n=== coverage report (pre/post split at first cohort) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
