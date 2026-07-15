"""Estimator validation on synthetic data with known truth.

These tests are the project's core credibility claim: before touching real
data, the exact pipeline (CS estimator, aggregations, elasticity step) must
recover a truth we planted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.estimate.cs import CSResult, run_cs
from src.estimate.elasticity import estimate_elasticity
from src.estimate.run import cohort_dose_table, synthetic_truth_curve
from src.estimate.twfe import run_twfe


def true_avg_effect(panel: pd.DataFrame) -> float:
    """Exact estimand: mean injected effect across treated post-period cells."""
    treated_post = panel["cohort_mindex"].notna() & (panel["mindex"] >= panel["cohort_mindex"])
    return float(panel.loc[treated_post, "true_effect"].mean())


def test_event_study_recovers_truth(cs_fit: CSResult, config, panel) -> None:
    truth = synthetic_truth_curve(config, cs_fit.event_study["rel_period"])
    merged = cs_fit.event_study.merge(truth, on="rel_period")
    post = merged[merged["rel_period"] >= 0]
    assert len(post) >= 12
    err = (post["att"] - post["true_att"]).abs()
    tol = np.maximum(3 * post["se"], 0.03)
    assert (err <= tol).all(), f"event-study misses truth:\n{post[err > tol]}"


def test_no_pretrends_in_leads(cs_fit: CSResult) -> None:
    leads = cs_fit.event_study.query("rel_period < -1")
    assert (leads["att"].abs() < 0.03).all()
    covers_zero = (leads["ci_low"] <= 0) & (leads["ci_high"] >= 0)
    assert covers_zero.mean() >= 0.8  # ~alpha-level exceptions allowed


def test_cs_close_to_truth_and_twfe_biased(cs_fit: CSResult, panel) -> None:
    truth = true_avg_effect(panel)
    cs_att = float(cs_fit.simple["att"].iloc[0])
    twfe = run_twfe(panel, outcome="log_subs")
    cs_err, twfe_err = abs(cs_att - truth), abs(twfe.att - truth)
    assert cs_err < 0.03, f"CS overall {cs_att:.4f} vs truth {truth:.4f}"
    # staggered timing + heterogeneous dynamic effects: TWFE must be visibly worse —
    # this is the Goodman-Bacon story the writeup leans on
    assert twfe_err > 2 * cs_err, f"TWFE err {twfe_err:.4f} vs CS err {cs_err:.4f}"
    assert twfe_err > 0.02


def test_elasticity_recovered(cs_fit: CSResult, config) -> None:
    res = estimate_elasticity(cs_fit, config, cohort_dose_table(config))
    true_eps = float(config.synth["true_elasticity"])
    assert res.epsilon == pytest.approx(true_eps, abs=0.05)
    # every cohort's ratio should individually be near the truth (dose-response coherence)
    assert (res.by_cohort["epsilon_ratio"] - true_eps).abs().max() < 0.10


def test_followers_placebo_null(panel, config) -> None:
    cs = run_cs(panel, config, outcome="log_followers")
    att = float(cs.simple["att"].iloc[0])
    lo, hi = float(cs.simple["ci_low"].iloc[0]), float(cs.simple["ci_high"].iloc[0])
    assert abs(att) < 0.02
    assert lo <= 0 <= hi


def test_fake_treatment_on_never_treated_is_null(panel, config) -> None:
    rng = np.random.default_rng(7)
    never = panel[panel["cohort_mindex"].isna()].copy()
    channels = never["channel_id"].unique()
    fake_cohorts = {c.name: float(c.treat_mindex) for c in config.cohorts}
    # assign fake cohorts to 60% of never-treated channels, keep 40% as controls
    n_fake = int(0.6 * len(channels))
    fake_ids = rng.choice(channels, size=n_fake, replace=False)
    assignment = dict(
        zip(fake_ids, rng.choice(list(fake_cohorts.values()), size=n_fake), strict=True)
    )
    never["cohort_mindex"] = never["channel_id"].map(assignment).astype(float)
    cs = run_cs(never, config, outcome="log_subs")
    att = float(cs.simple["att"].iloc[0])
    lo, hi = float(cs.simple["ci_low"].iloc[0]), float(cs.simple["ci_high"].iloc[0])
    assert abs(att) < 0.02
    assert lo <= 0 <= hi
