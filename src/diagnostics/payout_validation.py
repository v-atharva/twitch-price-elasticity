"""External validation against the 2021 Twitch payout records (STRICTLY LOCAL).

This module cross-checks our tracker-scraped subscriber counts against Twitch's
own subscription-revenue records for the same channel-months, and re-estimates
the treatment effect on *revenue* as an independent second outcome.

DATA HANDLING — non-negotiable:
  - The payout data is breach-obtained personal financial data. It is read from
    a LOCAL path OUTSIDE this repository and is NEVER copied in, committed, or
    pushed. The path is supplied via `--payouts` / env `TWITCH_PAYOUTS_MATCHED`.
  - This module writes ONLY AGGREGATE statistics (correlations, medians, an ATT)
    to a gitignored path (data/external/, default). No individual channel's
    revenue is ever emitted, committed, or published.
  - If the local data is absent (e.g. CI, a fresh clone), it no-ops cleanly so
    the rest of the pipeline is unaffected.

Input parquet schema (produced locally from the archive; see scripts, kept local):
  slug, month ("YYYY-MM"), sub_rev (USD gross), prime_rev (USD gross)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.estimate.cs import run_cs

DEFAULT_MATCHED = os.environ.get("TWITCH_PAYOUTS_MATCHED", "")


def _load_matched(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not path or not p.exists():
        return None
    df = pd.read_parquet(p)
    # collapse duplicate leak snapshots within a channel-month to the month value
    return df.groupby(["slug", "month"], as_index=False).agg(
        sub_rev=("sub_rev", "max"), prime_rev=("prime_rev", "max")
    )


def validate(panel_path: str, matched_path: str, config) -> dict | None:
    pay = _load_matched(matched_path)
    if pay is None:
        return None
    panel = pd.read_parquet(panel_path)
    j = (
        panel[["channel_id", "month", "mindex", "cohort_mindex", "subs", "pre_size", "country"]]
        .rename(columns={"channel_id": "slug"})
        .merge(pay, on=["slug", "month"], how="inner")
    )
    j = j[(j["subs"] > 0) & (j["sub_rev"] > 0)].copy()

    # 1) sample authenticity: how many panel channels appear in Twitch's records
    n_panel = panel["channel_id"].nunique()
    n_matched = j["slug"].nunique()

    # 2) do scraped subs track real subscription revenue?
    lx, ly = np.log(j["subs"]), np.log(j["sub_rev"])
    r_pooled = float(np.corrcoef(lx, ly)[0, 1])
    per = []
    for _s, g in j.groupby("slug"):
        if len(g) >= 8 and g["subs"].std() > 0 and g["sub_rev"].std() > 0:
            per.append(float(np.corrcoef(np.log(g["subs"]), np.log(g["sub_rev"]))[0, 1]))
    per_arr = np.array(per)

    # 3) implied revenue per active sub (economic sanity: ~$2.50 net on a $4.99 sub)
    rps = (j["sub_rev"] / j["subs"]).replace([np.inf, -np.inf], np.nan).dropna()

    # 4) revenue as a second outcome through the SAME CS estimator
    rev_panel = j.rename(columns={"slug": "channel_id"}).copy()
    rev_panel["log_subs"] = np.log(rev_panel["sub_rev"])  # reuse the outcome slot
    rev_panel["cat_share_games"] = 0.5
    rev_att: dict[str, float | str]
    try:
        cs = run_cs(
            rev_panel, config, outcome="log_subs",
            anticipation=int(config.estimator["anticipation_real"]),
        )  # fmt: skip
        rev_att = {
            "att": float(cs.simple["att"].iloc[0]),
            "se": float(cs.simple["se"].iloc[0]),
            "pct": float(np.exp(cs.simple["att"].iloc[0]) - 1),
        }
    except Exception as e:  # small/degenerate join -> report absence, don't crash
        rev_att = {"error": str(e)[:120]}

    return {
        "sample_authenticity": {
            "panel_channels": int(n_panel),
            "found_in_twitch_records": int(n_matched),
            "match_rate": round(n_matched / n_panel, 3),
        },
        "outcome_validation": {
            "corr_log_subs_vs_log_sub_revenue_pooled": round(r_pooled, 3),
            "within_channel_monthly_corr_median": round(float(np.median(per_arr)), 3),
            "within_channel_corr_iqr": [
                round(float(np.percentile(per_arr, 25)), 3),
                round(float(np.percentile(per_arr, 75)), 3),
            ],
            "n_channels_scored": len(per_arr),
        },
        "revenue_per_sub_usd": {
            "median": round(float(rps.median()), 2),
            "p25": round(float(rps.quantile(0.25)), 2),
            "p75": round(float(rps.quantile(0.75)), 2),
        },
        "revenue_as_second_outcome": rev_att,
        "joined_channel_months": len(j),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/study.yaml")
    parser.add_argument("--panel", default="data/processed/pilot_panel.parquet")
    parser.add_argument("--payouts", default=DEFAULT_MATCHED,
                        help="LOCAL path to matched monthly payout parquet (never committed)")  # fmt: skip
    # default output goes OUTSIDE the repo (data/external is gitignored) so
    # nothing derived from the leaked data can land in the committed tree
    parser.add_argument("--out", default="data/external/payout_validation.json")
    args = parser.parse_args()

    res = validate(args.panel, args.payouts, load_config(args.config))
    if res is None:
        print("payout data not present locally — skipping external validation (this is fine).")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
