"""Elasticity from cohort-level dose variation.

epsilon = d log(subs) / d log(price). Each cohort g supplies one point:
its "mature" treatment effect in log points (average ATT(g,t) over event
times in the mature window, where the ramp-in has finished) against its dose
dlogp_g. Weighted regression through the origin across cohorts gives epsilon.

This is one movement along a demand curve — a local elasticity at the 2021
price change, not a full demand system. The analytic SE here treats cohort
ATTs as independent (they share the control pool, so it is approximate);
M6 adds a bootstrap CI over channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.estimate.cs import CSResult


@dataclass
class ElasticityResult:
    epsilon: float
    se: float
    ci_low: float
    ci_high: float
    by_cohort: pd.DataFrame  # cohort_mindex, dlogp, att_mature, se, epsilon_ratio


def cohort_mature_atts(cs: CSResult, config: Config) -> pd.DataFrame:
    """Average ATT(g,t) per cohort over mature event times (post ramp-in)."""
    lo, hi = config.estimator.get("mature_window", [6, 12])
    gt = cs.group_time.copy()
    gt["rel_period"] = gt["mindex"] - gt["cohort_mindex"]
    mature = gt[(gt["rel_period"] >= lo) & (gt["rel_period"] <= hi)]
    if mature.empty:
        raise ValueError(f"no group-time ATTs inside mature window [{lo}, {hi}]")

    def agg(grp: pd.DataFrame) -> pd.Series:
        # mean of estimates; SE of the mean assuming independence across t (approximate)
        return pd.Series(
            {
                "att_mature": grp["att"].mean(),
                "se": np.sqrt((grp["se"] ** 2).sum()) / len(grp),
                "n_periods": len(grp),
            }
        )

    return mature.groupby("cohort_mindex").apply(agg, include_groups=False).reset_index()


def estimate_elasticity(cs: CSResult, config: Config, doses: pd.DataFrame) -> ElasticityResult:
    """doses: DataFrame with columns [cohort_mindex, dlogp]."""
    by = cohort_mature_atts(cs, config).merge(doses, on="cohort_mindex", how="inner")
    if by.empty:
        raise ValueError("no cohorts matched between ATTs and dose table")
    by["epsilon_ratio"] = by["att_mature"] / by["dlogp"]

    # WLS through origin: att_g = epsilon * dlogp_g, weights 1/se^2
    w = 1.0 / by["se"] ** 2
    x, y = by["dlogp"].to_numpy(), by["att_mature"].to_numpy()
    sxx = float((w * x * x).sum())
    epsilon = float((w * x * y).sum()) / sxx
    se = float(np.sqrt(1.0 / sxx))  # delta-method SE from cohort ATT SEs
    z = 1.959963984540054
    return ElasticityResult(
        epsilon=epsilon,
        se=se,
        ci_low=epsilon - z * se,
        ci_high=epsilon + z * se,
        by_cohort=by,
    )
