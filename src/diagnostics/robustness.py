"""Robustness battery (M7): each variant re-runs the identical CS pipeline.

Variants: not-yet-treated controls; drop-one-country; language-only cohort
assignment (Spanish pooled at the modal Spanish-market date instead of split
by streamer country); winsorized outcome; size subsamples.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config
from src.estimate.cs import run_cs
from src.panel.months import to_mindex


def _overall(panel: pd.DataFrame, config: Config, anticipation: int, **kw) -> dict:
    cs = run_cs(panel, config, outcome="log_subs", anticipation=anticipation, **kw)
    return {"att": float(cs.simple["att"].iloc[0]), "se": float(cs.simple["se"].iloc[0])}


def run_battery(panel: pd.DataFrame, config: Config, anticipation: int) -> pd.DataFrame:
    rows = []

    def add(name: str, res: dict, note: str = "") -> None:
        rows.append({"variant": name, **res, "note": note})

    add("baseline (never-treated controls)", _overall(panel, config, anticipation))
    add(
        "not-yet-treated controls",
        _overall(panel, config, anticipation, control_group="not_yet_treated"),
        "identification from timing only while g259 pending",
    )

    for country in sorted(panel.loc[panel["cohort_mindex"].notna(), "country"].unique()):
        sub = panel[panel["country"] != country]
        add(f"drop {country}", _overall(sub, config, anticipation))

    # language-only assignment: all Spanish-language channels get one cohort at
    # the modal Spanish-market treatment month (2021-08) regardless of country
    lang_only = panel.copy()
    es_mask = lang_only["language"] == "es"
    lang_only.loc[es_mask, "cohort_mindex"] = float(to_mindex("2021-08"))
    lang_only.loc[es_mask, "cohort_name"] = "g_es_pooled"
    add(
        "language-only assignment (es pooled)",
        _overall(lang_only, config, anticipation),
        "ignores MX/AR vs ES timing differences - misassignment stress test",
    )

    # winsorize log outcome within month at 1%/99%
    wins = panel.copy()
    lo = wins.groupby("mindex")["log_subs"].transform(lambda s: s.quantile(0.01))
    hi = wins.groupby("mindex")["log_subs"].transform(lambda s: s.quantile(0.99))
    wins["log_subs"] = wins["log_subs"].clip(lo, hi)
    add("winsorized 1/99 within month", _overall(wins, config, anticipation))

    med = panel.groupby("channel_id")["pre_size"].first().median()
    small = panel[panel.groupby("channel_id")["pre_size"].transform("first") <= med]
    large = panel[panel.groupby("channel_id")["pre_size"].transform("first") > med]
    add("below-median pre-size", _overall(small, config, anticipation))
    add("above-median pre-size", _overall(large, config, anticipation))

    return pd.DataFrame(rows)
