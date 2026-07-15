"""End-to-end estimation CLI: panel -> CS ATTs, TWFE benchmark, elasticity, figures.

Runs identically on synthetic and real panels (same schema). On synthetic
panels the event-study figure overlays the known true effect curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.estimate.cs import run_cs
from src.estimate.elasticity import estimate_elasticity
from src.estimate.twfe import run_twfe
from src.panel.schema import validate_panel
from src.panel.synthetic import true_effect_curve
from src.report.figures import event_study_figure


def cohort_dose_table(config: Config) -> pd.DataFrame:
    return pd.DataFrame(
        [{"cohort_mindex": float(c.treat_mindex), "dlogp": c.dlogp} for c in config.cohorts]
    )


def synthetic_truth_curve(config: Config, rel_periods: pd.Series) -> pd.DataFrame:
    """Cohort-share-weighted true event-study path, for overlay on synth figures."""
    s = config.synth
    shares = {k: v for k, v in s["cohort_shares"].items() if k != "never"}
    total = sum(shares.values())
    ks = np.array(sorted(rel_periods.unique()))
    att = np.zeros_like(ks, dtype=float)
    for name, share in shares.items():
        c = config.cohort(name)
        att += (share / total) * true_effect_curve(
            ks.astype(float), c.dlogp, float(s["true_elasticity"]), float(s["ramp_tau"])
        )
    return pd.DataFrame({"rel_period": ks, "true_att": att})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/study.yaml")
    parser.add_argument("--panel", default="data/processed/synthetic_panel.parquet")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    panel = validate_panel(pd.read_parquet(args.panel))
    is_synth = "true_effect" in panel.columns
    out = Path(args.out_dir)
    (out / "estimates").mkdir(parents=True, exist_ok=True)

    # --- primary outcome ---------------------------------------------------
    cs = run_cs(panel, config, outcome="log_subs")
    cs.event_study.to_csv(out / "estimates/event_study_log_subs.csv", index=False)
    cs.cohort.to_csv(out / "estimates/cohort_atts_log_subs.csv", index=False)
    cs.group_time.to_csv(out / "estimates/group_time_atts_log_subs.csv", index=False)

    twfe = run_twfe(panel, outcome="log_subs")
    elas = estimate_elasticity(cs, config, cohort_dose_table(config))
    elas.by_cohort.to_csv(out / "estimates/elasticity_by_cohort.csv", index=False)

    # --- falsification outcome: followers are free ------------------------
    cs_follow = run_cs(panel, config, outcome="log_followers")
    cs_follow.event_study.to_csv(out / "estimates/event_study_log_followers.csv", index=False)

    summary = {
        "panel": str(args.panel),
        "n_channels": int(panel["channel_id"].nunique()),
        "n_rows": len(panel),
        "cs_overall_att": float(cs.simple["att"].iloc[0]),
        "cs_overall_se": float(cs.simple["se"].iloc[0]),
        "twfe_att": twfe.att,
        "twfe_se": twfe.se,
        "elasticity": elas.epsilon,
        "elasticity_se": elas.se,
        "elasticity_ci": [elas.ci_low, elas.ci_high],
        "followers_overall_att": float(cs_follow.simple["att"].iloc[0]),
        "followers_overall_se": float(cs_follow.simple["se"].iloc[0]),
    }
    if is_synth:
        treated_post = panel["true_effect"] != 0
        summary["true_avg_effect_treated_cells"] = float(
            panel.loc[treated_post, "true_effect"].mean()
        )
        summary["true_elasticity"] = float(config.synth["true_elasticity"])
    (out / "estimates/summary.json").write_text(json.dumps(summary, indent=2))

    # --- figures -----------------------------------------------------------
    truth = synthetic_truth_curve(config, cs.event_study["rel_period"]) if is_synth else None
    label = "synthetic panel" if is_synth else "real panel"
    event_study_figure(
        cs.event_study,
        out / "figures/event_study_log_subs.png",
        title="Cheaper subs, more subs: effect of local pricing on subscriptions",
        subtitle=f"Callaway–Sant'Anna event study, log active subs, {label}; "
        f"never-treated (US-audience) controls, doubly robust",
        truth=truth,
    )
    event_study_figure(
        cs_follow.event_study,
        out / "figures/event_study_log_followers.png",
        title="Falsification: followers are free, price should not move them",
        subtitle=f"Same design, outcome = log followers, {label}",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
