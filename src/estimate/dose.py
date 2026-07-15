"""Country-level dose-response elasticity on the real panel.

Each treated country c is one dose point: its CS ATT (that country's channels
vs the never-treated US pool, anticipation-adjusted) against its price dose
dlogp_c from the curated price table. Elasticity is WLS through the origin:

    ATT_c = epsilon * dlogp_c + u_c,   weights 1/se_c^2

Two ATT flavors per country: `overall` (CS 'simple' aggregation over all post
periods) and `mature` (group-time ATTs averaged over the mature event window,
post ramp-in) — the mature one is the headline estimand.

CI: cluster bootstrap over channels (treated resampled within country,
controls resampled from the US pool), rerunning the whole per-country CS +
WLS pipeline each draw. This respects the correlation induced by the shared
control pool that the analytic SE ignores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.estimate.cs import run_cs
from src.panel.prices import load_price_table


@dataclass
class DoseResponse:
    by_country: pd.DataFrame  # country, dlogp, att, se, n_channels
    epsilon: float
    se_analytic: float
    ci_boot: tuple[float, float] | None
    boot_draws: np.ndarray | None


def _country_att(
    panel: pd.DataFrame, config: Config, country: str, anticipation: int, mature: bool
) -> tuple[float, float] | None:
    sub = panel[(panel["country"] == country) | (panel["cohort_mindex"].isna())]
    if sub[sub["country"] == country]["channel_id"].nunique() == 0:
        return None
    try:
        cs = run_cs(sub, config, outcome="log_subs", anticipation=anticipation)
    except Exception:
        return None  # degenerate resample (e.g. no pre-period channels drawn)
    if mature:
        lo, hi = config.estimator.get("mature_window", [6, 12])
        gt = cs.group_time
        rel = gt["mindex"] - gt["cohort_mindex"]
        m = gt[(rel >= lo) & (rel <= hi)]
        if m.empty:
            return None
        att = float(m["att"].mean())
        se = float(np.sqrt((m["se"] ** 2).sum()) / len(m))
        return att, se
    return float(cs.simple["att"].iloc[0]), float(cs.simple["se"].iloc[0])


def _wls_origin(att: np.ndarray, se: np.ndarray, dose: np.ndarray) -> tuple[float, float]:
    w = 1.0 / se**2
    sxx = float((w * dose * dose).sum())
    eps = float((w * dose * att).sum()) / sxx
    return eps, float(np.sqrt(1.0 / sxx))


def estimate_dose_response(
    panel: pd.DataFrame,
    config: Config,
    *,
    anticipation: int = 1,
    mature: bool = True,
    n_boot: int = 200,
    seed: int = 0,
) -> DoseResponse:
    prices = load_price_table()
    dose_by_country = dict(zip(prices["country_code"], prices["dlogp"], strict=True))
    countries = sorted(c for c in panel.loc[panel["cohort_mindex"].notna(), "country"].unique())

    rows = []
    for c in countries:
        res = _country_att(panel, config, c, anticipation, mature)
        if res is None or c not in dose_by_country:
            continue
        att, se = res
        rows.append(
            {
                "country": c,
                "dlogp": float(dose_by_country[c]),
                "att": att,
                "se": se,
                "n_channels": int(panel.loc[panel["country"] == c, "channel_id"].nunique()),
            }
        )
    by = pd.DataFrame(rows)
    eps, se_analytic = _wls_origin(
        by["att"].to_numpy(), by["se"].to_numpy(), by["dlogp"].to_numpy()
    )

    draws = None
    ci = None
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        controls = panel[panel["cohort_mindex"].isna()]
        ctrl_ids = controls["channel_id"].unique()
        treated_ids = {
            c: panel.loc[panel["country"] == c, "channel_id"].unique() for c in by["country"]
        }
        by_channel = {ch: grp for ch, grp in panel.groupby("channel_id")}
        draws_list = []
        for _b in range(n_boot):
            parts = []
            for tag, ids in [("ctrl", ctrl_ids), *[(c, treated_ids[c]) for c in by["country"]]]:
                sampled = rng.choice(ids, size=len(ids), replace=True)
                for j, ch in enumerate(sampled):
                    g = by_channel[ch].copy()
                    g["channel_id"] = f"{tag}_{j}_{ch}"
                    parts.append(g)
            bpanel = pd.concat(parts, ignore_index=True)
            batt, bse, bdose = [], [], []
            for c in by["country"]:
                res = _country_att(bpanel, config, c, anticipation, mature)
                if res is None:
                    break
                batt.append(res[0])
                bse.append(res[1])
                bdose.append(float(dose_by_country[c]))
            else:
                e, _ = _wls_origin(np.array(batt), np.array(bse), np.array(bdose))
                draws_list.append(e)
        draws = np.array(draws_list)
        draws = draws[np.isfinite(draws)]  # degenerate resamples can yield NaN
        if len(draws) >= 50:
            ci = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))

    return DoseResponse(
        by_country=by, epsilon=eps, se_analytic=se_analytic, ci_boot=ci, boot_draws=draws
    )
