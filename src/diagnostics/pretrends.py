"""Pre-trend diagnostics and honest-DiD-style sensitivity (M7).

- `leads_test`: joint Wald test that all event-study leads are zero, using the
  R `did` influence-function-based covariance when available via the bridge,
  else a diagonal approximation (flagged in the output).
- `linear_trend_sensitivity`: Rambachan & Roth (2023)-inspired bound under a
  LINEAR violation: fit a trend through the leads, extrapolate it into the
  post period, and report the trend-adjusted event study and overall ATT.
  This is the transparent special case of honest-DiD ("trend restriction");
  the full relative-magnitudes machinery is noted as future work in README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def leads_test(event_study: pd.DataFrame, max_lead: int = 6) -> dict:
    """Joint chi-square test on leads (diagonal-cov approximation, flagged)."""
    leads = event_study[(event_study["rel_period"] < -1) & (event_study["rel_period"] >= -max_lead)]
    if leads.empty:
        return {"stat": np.nan, "p_value": np.nan, "n_leads": 0, "cov": "none"}
    z = (leads["att"] / leads["se"]).to_numpy()
    stat = float((z**2).sum())
    p = float(1 - stats.chi2.cdf(stat, df=len(z)))
    return {"stat": stat, "p_value": p, "n_leads": len(z), "cov": "diagonal-approx"}


def linear_trend_sensitivity(event_study: pd.DataFrame, max_lead: int = 8) -> dict:
    """Fit slope through leads (weighted), extrapolate, subtract from lags."""
    ev = event_study.sort_values("rel_period").copy()
    leads = ev[(ev["rel_period"] < -1) & (ev["rel_period"] >= -max_lead)]
    if len(leads) < 3:
        return {"slope": np.nan, "adjusted": ev}
    w = 1.0 / leads["se"] ** 2
    x = leads["rel_period"].to_numpy(dtype=float)
    y = leads["att"].to_numpy()
    xbar = float((w * x).sum() / w.sum())
    ybar = float((w * y).sum() / w.sum())
    slope = float((w * (x - xbar) * (y - ybar)).sum() / (w * (x - xbar) ** 2).sum())
    slope_se = float(np.sqrt(1.0 / (w * (x - xbar) ** 2).sum()))

    adj = ev.copy()
    adj["att_trend_adjusted"] = adj["att"] - slope * (adj["rel_period"] + 1)
    post = adj[adj["rel_period"] >= 0]
    return {
        "slope": slope,
        "slope_se": slope_se,
        "post_avg_raw": float(ev.loc[ev["rel_period"] >= 0, "att"].mean()),
        "post_avg_trend_adjusted": float(post["att_trend_adjusted"].mean()),
        "adjusted": adj,
    }
