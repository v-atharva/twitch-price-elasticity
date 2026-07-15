"""Goodman-Bacon (2021) decomposition of the static TWFE coefficient.

The TWFE estimate is a variance-weighted average of all 2x2 DiD comparisons:
each treated cohort vs never-treated, earlier- vs later-treated (clean), and
later- vs earlier-treated ("forbidden": already-treated units serve as
controls, which biases TWFE when effects are dynamic). This makes the
TWFE-vs-CS gap interpretable rather than mysterious.

Requires a BALANCED panel (the identity only holds exactly there) — used on
the synthetic panel for the didactic figure; the real pilot panel is
unbalanced, which the writeup notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class BaconDecomposition:
    comparisons: pd.DataFrame  # type, group1, group2, weight, estimate
    twfe: float  # weighted average (= TWFE coefficient on balanced panel)

    def by_type(self) -> pd.DataFrame:
        rows = []
        for t, d in self.comparisons.groupby("type"):
            rows.append(
                {
                    "type": t,
                    "weight": float(d["weight"].sum()),
                    "avg_estimate": float(np.average(d["estimate"], weights=d["weight"])),
                }
            )
        return pd.DataFrame(rows)


def _two_by_two(
    df: pd.DataFrame, treat_g: float, ctrl_g: float | None, window: tuple[int, int], outcome: str
) -> float:
    """Simple 2x2 DiD of treat group vs control group over [pre, post) split at treat_g."""
    lo, hi = window
    ctrl = df[df["cohort_mindex"].isna()] if ctrl_g is None else df[df["cohort_mindex"] == ctrl_g]
    trt = df[df["cohort_mindex"] == treat_g]
    means = {}
    for name, grp in (("t", trt), ("c", ctrl)):
        sub = grp[(grp["mindex"] >= lo) & (grp["mindex"] < hi)]
        pre = sub.loc[sub["mindex"] < treat_g, outcome].mean()
        post = sub.loc[sub["mindex"] >= treat_g, outcome].mean()
        means[name] = post - pre
    return float(means["t"] - means["c"])


def bacon_decompose(panel: pd.DataFrame, outcome: str = "log_subs") -> BaconDecomposition:
    """Decompose static TWFE into weighted 2x2 comparisons (balanced panel)."""
    df = panel[["channel_id", "mindex", "cohort_mindex", outcome]].copy()
    t_min, t_max = int(df["mindex"].min()), int(df["mindex"].max())
    T = t_max - t_min + 1
    groups = sorted(g for g in df["cohort_mindex"].dropna().unique())
    n = df.groupby("channel_id")["cohort_mindex"].first()
    n_never = int(n.isna().sum())
    n_by_g = {g: int((n == g).sum()) for g in groups}
    n_total = len(n)

    def share(g: float) -> float:  # share of periods treated
        return (t_max + 1 - g) / T

    rows = []
    # treated vs never-treated
    for g in groups:
        if n_never == 0:
            break
        Dg = share(g)
        n_gu = (n_by_g[g] + n_never) / n_total
        w = n_gu**2 * (n_by_g[g] / (n_by_g[g] + n_never)) * (n_never / (n_by_g[g] + n_never))
        w *= Dg * (1 - Dg)
        est = _two_by_two(df, g, None, (t_min, t_max + 1), outcome)
        rows.append(
            {"type": "treated_vs_never", "group1": g, "group2": None, "w_raw": w, "estimate": est}
        )
    # timing pairs
    for gk, gl in combinations(groups, 2):  # gk earlier than gl
        Dk, Dl = share(gk), share(gl)
        n_ku = (n_by_g[gk] + n_by_g[gl]) / n_total
        nk = n_by_g[gk] / (n_by_g[gk] + n_by_g[gl])
        nl = 1 - nk
        # earlier vs later (later still untreated: window ends when later gets treated)
        # weights per Goodman-Bacon (2021), eq. (10a/10b)
        w_early = n_ku**2 * nk * nl * (Dk - Dl) * (1 - Dk) / (1 - (Dk - Dl)) ** 2
        est_early = _two_by_two(df, gk, gl, (t_min, int(gl)), outcome)
        rows.append(
            {
                "type": "earlier_vs_later",
                "group1": gk,
                "group2": gl,
                "w_raw": w_early,
                "estimate": est_early,
            }
        )
        # later vs earlier (earlier already treated: window starts after earlier's g)
        w_late = n_ku**2 * nk * nl * (Dk - Dl) * Dl / (1 - (Dk - Dl)) ** 2
        est_late = _two_by_two(df, gl, gk, (int(gk), t_max + 1), outcome)
        rows.append(
            {
                "type": "later_vs_earlier(forbidden)",
                "group1": gl,
                "group2": gk,
                "w_raw": w_late,
                "estimate": est_late,
            }
        )

    comp = pd.DataFrame(rows)
    comp["weight"] = comp["w_raw"] / comp["w_raw"].sum()
    comp = comp.drop(columns="w_raw")
    twfe = float(np.average(comp["estimate"], weights=comp["weight"]))
    return BaconDecomposition(comparisons=comp, twfe=twfe)
