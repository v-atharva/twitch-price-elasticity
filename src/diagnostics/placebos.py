"""Placebo and permutation diagnostics (M7).

1. Prime-subs placebo: same design, outcome = log Prime subs (free via
   Amazon Prime -> price-insensitive). A large "effect" here flags common
   shocks (popularity booms) rather than price response.
2. Fake treatment dates on never-treated channels only: assign the real
   cohort months to random US-audience channels; effect should be ~0.
3. Permutation inference: shuffle which channels are treated (holding the
   cohort structure fixed) B times; the rank of the observed ATT in the
   permutation distribution gives a design-based p-value robust to the tiny
   number of treated clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.estimate.cs import run_cs


def prime_placebo(panel: pd.DataFrame, config: Config, anticipation: int) -> dict:
    sub = panel[panel["log_followers"].notna()]
    cs = run_cs(sub, config, outcome="log_followers", anticipation=anticipation)
    return {
        "att": float(cs.simple["att"].iloc[0]),
        "se": float(cs.simple["se"].iloc[0]),
        "event_study": cs.event_study,
    }


def fake_dates_on_controls(
    panel: pd.DataFrame, config: Config, anticipation: int, seed: int = 11
) -> dict:
    rng = np.random.default_rng(seed)
    never = panel[panel["cohort_mindex"].isna()].copy()
    channels = never["channel_id"].unique()
    real_gs = panel.loc[panel["cohort_mindex"].notna(), "cohort_mindex"].unique()
    n_fake = len(channels) // 2
    fake_ids = rng.choice(channels, size=n_fake, replace=False)
    assignment = dict(zip(fake_ids, rng.choice(real_gs, size=n_fake), strict=True))
    never["cohort_mindex"] = never["channel_id"].map(assignment).astype(float)
    never["cohort_name"] = np.where(never["cohort_mindex"].notna(), "fake", "never")
    cs = run_cs(never, config, outcome="log_subs", anticipation=anticipation)
    return {"att": float(cs.simple["att"].iloc[0]), "se": float(cs.simple["se"].iloc[0])}


def permutation_test(
    panel: pd.DataFrame,
    config: Config,
    anticipation: int,
    n_perm: int = 200,
    seed: int = 13,
) -> dict:
    """Permute treated-status across channels, keep cohort-size structure."""
    observed = float(
        run_cs(panel, config, outcome="log_subs", anticipation=anticipation).simple["att"].iloc[0]
    )
    assign = panel.groupby("channel_id")["cohort_mindex"].first()  # channel -> g (NaN = control)
    channels = assign.index.to_numpy()
    gs = assign.to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_perm):
        perm = rng.permutation(gs)
        mapping = dict(zip(channels, perm, strict=True))
        p = panel.copy()
        p["cohort_mindex"] = p["channel_id"].map(mapping)
        p["cohort_name"] = np.where(p["cohort_mindex"].notna(), "perm", "never")
        try:
            att = float(
                run_cs(p, config, outcome="log_subs", anticipation=anticipation)
                .simple["att"]
                .iloc[0]
            )
        except Exception:
            continue  # degenerate permutation (e.g. a cohort with no pre-period)
        draws.append(att)
    draws_arr = np.array(draws)
    p_two = float((np.abs(draws_arr) >= abs(observed)).mean()) if len(draws_arr) else np.nan
    return {
        "observed_att": observed,
        "n_permutations_effective": len(draws_arr),
        "p_value_two_sided": p_two,
        "perm_quantiles": {
            "q025": float(np.percentile(draws_arr, 2.5)),
            "q975": float(np.percentile(draws_arr, 97.5)),
        }
        if len(draws_arr)
        else None,
    }
