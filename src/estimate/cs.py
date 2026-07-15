"""Callaway & Sant'Anna (2021) estimator wrapper around the `differences` package.

Normalizes the package's MultiIndex outputs into flat DataFrames with stable
column names so downstream code (figures, diagnostics, R cross-check) never
touches backend-specific structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from differences import ATTgt

from src.config import Config

FLAT_NAMES = {"ATT": "att", "std_error": "se", "lower": "ci_low", "upper": "ci_high"}


@dataclass
class CSResult:
    event_study: pd.DataFrame  # rel_period, att, se, ci_low, ci_high
    cohort: pd.DataFrame  # cohort_mindex, att, se, ci_low, ci_high
    simple: pd.DataFrame  # single row: att, se, ci_low, ci_high
    group_time: pd.DataFrame  # cohort_mindex, base_period, mindex, att, se, ...
    model: ATTgt  # backend object, for diagnostics that need it


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [FLAT_NAMES.get(c[-1], c[-1]) for c in out.columns]
    return out.reset_index()


def run_cs(
    panel: pd.DataFrame,
    config: Config,
    outcome: str = "log_subs",
    *,
    control_group: str | None = None,
    anticipation: int | None = None,
    covariates: list[str] | None = None,
    boot_iterations: int = 0,
    random_state: int = 0,
) -> CSResult:
    """Group-time ATTs + event-study / cohort / overall aggregations.

    Options default to config['estimator']; keyword overrides exist for
    robustness variants (e.g. control_group='not_yet_treated').
    """
    est: dict[str, Any] = config.estimator
    control_group = control_group or est["control_group"]
    anticipation = est["anticipation"] if anticipation is None else anticipation
    covariates = est.get("covariates", []) if covariates is None else covariates
    # a zero-variance covariate makes the DR design singular (e.g. pilot placeholders)
    covariates = [c for c in covariates if panel[c].nunique(dropna=True) > 1]

    cols = ["channel_id", "mindex", "cohort_mindex", outcome, *covariates]
    data = panel[cols].set_index(["channel_id", "mindex"]).sort_index()

    att = ATTgt(data=data, cohort_column="cohort_mindex", anticipation=anticipation)
    formula = outcome if not covariates else f"{outcome} ~ {' + '.join(covariates)}"
    att.fit(
        formula=formula,
        est_method=est["est_method"],
        control_group=control_group,
        alpha=float(est.get("alpha", 0.05)),
        boot_iterations=boot_iterations,
        random_state=random_state,
        progress_bar=False,
    )

    event = _flatten(att.aggregate("event")).rename(columns={"relative_period": "rel_period"})
    lo, hi = est.get("event_window", [-12, 12])
    event = event[(event["rel_period"] >= lo) & (event["rel_period"] <= hi)].reset_index(drop=True)

    cohort = _flatten(att.aggregate("cohort")).rename(columns={"cohort": "cohort_mindex"})
    simple = _flatten(att.aggregate("simple"))
    group_time = _flatten(att.results()).rename(
        columns={"cohort": "cohort_mindex", "time": "mindex"}
    )
    return CSResult(
        event_study=event, cohort=cohort, simple=simple, group_time=group_time, model=att
    )
