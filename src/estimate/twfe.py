"""Naive two-way fixed effects benchmark.

Kept as the didactic contrast: with staggered adoption and dynamic,
heterogeneous effects, the single TWFE coefficient mixes clean and forbidden
comparisons (Goodman-Bacon 2021). M6 adds the full decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pyfixest as pf


@dataclass
class TWFEResult:
    att: float
    se: float
    ci_low: float
    ci_high: float


def run_twfe(panel: pd.DataFrame, outcome: str = "log_subs") -> TWFEResult:
    df = panel.copy()
    df["treat"] = (df["cohort_mindex"].notna() & (df["mindex"] >= df["cohort_mindex"])).astype(int)
    fit = pf.feols(f"{outcome} ~ treat | channel_id + mindex", data=df, vcov={"CRV1": "channel_id"})
    ci = fit.confint().loc["treat"]
    return TWFEResult(
        att=float(fit.coef().loc["treat"]),
        se=float(fit.se().loc["treat"]),
        ci_low=float(ci.iloc[0]),
        ci_high=float(ci.iloc[1]),
    )
