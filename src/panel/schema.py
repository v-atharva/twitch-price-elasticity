"""Canonical channel-month panel schema.

Both the synthetic generator and the real-data pipeline emit this exact shape,
so everything downstream (estimation, diagnostics, figures) is source-agnostic.
"""

from __future__ import annotations

import pandas as pd

# column -> dtype kind check ("i" int, "f" float, "O" object/str)
REQUIRED_COLUMNS: dict[str, str] = {
    "channel_id": "O",
    "month": "O",  # "YYYY-MM"
    "mindex": "i",  # months since 2000-01
    "cohort_mindex": "f",  # NaN = never-treated (US/English audience)
    "cohort_name": "O",  # "never" or cohort key from config
    "country": "O",  # dominant audience country (proxy)
    "language": "O",  # broadcast language
    "subs": "f",  # active subscriber count (paid non-gift where separable)
    "log_subs": "f",
    "followers": "f",
    "log_followers": "f",  # placebo outcome: free action, price should not move it
    "pre_size": "f",  # pre-treatment log size covariate (no look-ahead)
    "cat_share_games": "f",  # pre-treatment category mix covariate
}

OPTIONAL_COLUMNS: dict[str, str] = {
    "true_effect": "f",  # synthetic panels only: the injected treatment effect
    "dlogp": "f",  # treatment dose: log price change of the channel's cohort
}


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Raise on schema violations; return df unchanged for chaining."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    for col, kind in REQUIRED_COLUMNS.items():
        actual = df[col].dtype.kind
        ok = actual == kind or (kind == "f" and actual == "i") or (kind == "O" and actual == "U")
        if not ok:
            raise ValueError(f"column {col}: expected dtype kind {kind!r}, got {actual!r}")
    dup = df.duplicated(subset=["channel_id", "mindex"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate channel-month rows")
    treated = df["cohort_mindex"].notna()
    if not (df.loc[treated, "cohort_name"] != "never").all():
        raise ValueError("treated rows must have a cohort_name")
    if not (df.loc[~treated, "cohort_name"] == "never").all():
        raise ValueError("never-treated rows must have cohort_name == 'never'")
    # A channel's cohort must not vary over time (treatment is absorbing, assigned per channel).
    per_channel = df.groupby("channel_id")["cohort_mindex"].nunique(dropna=False)
    if (per_channel > 1).any():
        raise ValueError("cohort_mindex varies within channel")
    return df
