"""Synthetic channel-month panel with known true elasticity.

Purpose: (1) prove the whole pipeline runs end-to-end with zero network access,
(2) validate that the Callaway–Sant'Anna estimator recovers a known truth and
that naive TWFE is biased under staggered timing + heterogeneous effects.

DGP (all in logs):
    log_subs[i,t] = alpha_i + delta_t + tau_g(t - g) + eps[i,t]

  - alpha_i: channel fixed effect (log-normal channel sizes), shifted by a
    category-mix covariate so the doubly-robust path has real work to do.
  - delta_t: COVID-era common calendar trend (fast 2020 growth, then plateau).
  - tau_g(k) = elasticity * dlogp_g * (1 - exp(-(k+1)/ramp_tau)) for k >= 0:
    dose-proportional effect ramping to full over a few months. Heterogeneous
    across cohorts (doses differ) and dynamic — exactly the regime where TWFE's
    already-treated comparisons go wrong.
  - followers get the same alpha/delta structure but NO treatment effect:
    the falsification outcome (follows are free; price shouldn't move them).

The injected effect is stored per row in `true_effect`, so tests can compare
any estimate against the exact estimand on the same cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.panel.months import from_mindex, to_mindex
from src.panel.schema import validate_panel

CONTROL_COUNTRY_LANGUAGE = ("US", "en")
# Rough language tags for synthetic cohort countries (only used for realism/grouping).
COUNTRY_LANGUAGE = {
    "TR": "tr", "MX": "es", "BR": "pt", "AR": "es", "CL": "es", "CO": "es",
    "TH": "th", "KR": "ko", "IN": "hi", "PH": "en",
    "DE": "de", "FR": "fr", "IT": "it", "PL": "pl",
}  # fmt: skip


def true_effect_curve(
    k: np.ndarray, dlogp: float, elasticity: float, ramp_tau: float
) -> np.ndarray:
    """Treatment effect in log points at event time k >= 0 (0 before treatment)."""
    full = elasticity * dlogp
    return np.where(k >= 0, full * (1.0 - np.exp(-(k + 1) / ramp_tau)), 0.0)


def generate(config: Config) -> pd.DataFrame:
    s = config.synth
    rng = np.random.default_rng(int(s["seed"]))

    start_idx, end_idx = to_mindex(config.panel_start), to_mindex(config.panel_end)
    mindexes = np.arange(start_idx, end_idx + 1)
    n_months = len(mindexes)
    months = [from_mindex(int(i)) for i in mindexes]
    years = np.array([int(m.split("-")[0]) for m in months])

    # --- channels ---------------------------------------------------------
    n = int(s["n_channels"])
    shares = s["cohort_shares"]
    names = list(shares)
    counts = np.floor(np.array([shares[c] for c in names]) * n).astype(int)
    counts[0] += n - counts.sum()  # remainder to first group
    cohort_name = np.repeat(names, counts)
    rng.shuffle(cohort_name)

    cohort_mindex = np.full(n, np.nan)
    dlogp = np.zeros(n)
    country = np.full(n, CONTROL_COUNTRY_LANGUAGE[0], dtype=object)
    for c in config.cohorts:
        mask = cohort_name == c.name
        cohort_mindex[mask] = c.treat_mindex
        dlogp[mask] = c.dlogp
        country[mask] = rng.choice(list(c.example_countries), size=int(mask.sum()))
    language = np.array(
        [COUNTRY_LANGUAGE.get(str(ctry), CONTROL_COUNTRY_LANGUAGE[1]) for ctry in country],
        dtype=object,
    )

    cat_share_games = rng.beta(2, 2, size=n)
    alpha = rng.normal(
        float(s["base_log_subs_mean"]), float(s["base_log_subs_sd"]), size=n
    ) + 0.4 * (cat_share_games - 0.5)
    # Pre-treatment size covariate: alpha measured with noise (no look-ahead by construction).
    pre_size = alpha + rng.normal(0, 0.2, size=n)

    # --- calendar trend ---------------------------------------------------
    growth = np.array([float(s["monthly_growth"][str(y)]) for y in years])
    delta = np.concatenate([[0.0], np.cumsum(growth[:-1])])

    # --- assemble long panel ---------------------------------------------
    ch = np.repeat(np.arange(n), n_months)
    t = np.tile(mindexes, n)
    k = t - np.repeat(cohort_mindex, n_months)  # event time; NaN for never-treated
    anticipation = int(s.get("anticipation_months", 0))
    k_eff = np.where(np.isnan(k), -np.inf, k + anticipation)

    tau = np.zeros(n * n_months)
    for c in config.cohorts:
        rows = np.repeat(cohort_name == c.name, n_months)
        tau[rows] = true_effect_curve(
            k_eff[rows], c.dlogp, float(s["true_elasticity"]), float(s["ramp_tau"])
        )

    noise_sd = float(s["noise_sd"])
    log_subs = (
        np.repeat(alpha, n_months) + np.tile(delta, n) + tau + rng.normal(0, noise_sd, n * n_months)
    )
    log_followers = (
        np.repeat(alpha, n_months) + 2.0 + 0.8 * np.tile(delta, n)
        + rng.normal(0, noise_sd, n * n_months)
    )  # fmt: skip

    df = pd.DataFrame(
        {
            "channel_id": np.char.add("synth_", ch.astype(str)),
            "month": np.tile(np.array(months, dtype=object), n),
            "mindex": t.astype(int),
            "cohort_mindex": np.repeat(cohort_mindex, n_months),
            "cohort_name": np.repeat(cohort_name, n_months).astype(object),
            "country": np.repeat(country, n_months),
            "language": np.repeat(language, n_months),
            "subs": np.exp(log_subs),
            "log_subs": log_subs,
            "followers": np.exp(log_followers),
            "log_followers": log_followers,
            "pre_size": np.repeat(pre_size, n_months),
            "cat_share_games": np.repeat(cat_share_games, n_months),
            "true_effect": tau,
            "dlogp": np.repeat(dlogp, n_months),
        }
    )
    return validate_panel(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/study.yaml")
    parser.add_argument("--out", default="data/processed/synthetic_panel.parquet")
    args = parser.parse_args()

    config = load_config(args.config)
    df = generate(config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    n_channels = df["channel_id"].nunique()
    print(f"wrote {out}: {len(df):,} rows, {n_channels:,} channels, "
          f"{df['month'].nunique()} months")  # fmt: skip
    print(df.groupby("cohort_name", sort=False)["channel_id"].nunique().to_string())


if __name__ == "__main__":
    main()
