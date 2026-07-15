"""M6 driver: final estimation artifacts on the real (pilot) panel.

Produces, under outputs/:
  estimates/real_event_study.csv        CS event study, uniform (sup-t) bands
  estimates/real_cohort_atts.csv        per-cohort aggregation
  estimates/dose_response.csv           per-country ATT vs price dose
  estimates/final_summary.json          headline numbers incl. bootstrap CI
  figures/real_event_study.png          the money plot
  figures/dose_response.png             elasticity fit across countries
  figures/bacon_synthetic.csv           Goodman-Bacon decomposition (synthetic)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.estimate.bacon import bacon_decompose
from src.estimate.cs import run_cs
from src.estimate.dose import estimate_dose_response
from src.estimate.twfe import run_twfe
from src.panel.schema import validate_panel
from src.report.figures import dose_response_figure, event_study_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/study.yaml")
    parser.add_argument("--panel", default="data/processed/pilot_panel.parquet")
    parser.add_argument("--synth-panel", default="data/processed/synthetic_panel.parquet")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    panel = validate_panel(pd.read_parquet(args.panel))
    anticipation = int(config.estimator["anticipation_real"])
    out = Path(args.out_dir)
    (out / "estimates").mkdir(parents=True, exist_ok=True)

    # --- event study with uniform bands ------------------------------------
    cs = run_cs(
        panel, config, outcome="log_subs", anticipation=anticipation,
        boot_iterations=999, random_state=42,
    )  # fmt: skip
    cs.event_study.to_csv(out / "estimates/real_event_study.csv", index=False)
    cs.cohort.to_csv(out / "estimates/real_cohort_atts.csv", index=False)
    n_treat = panel.loc[panel["cohort_mindex"].notna(), "channel_id"].nunique()
    n_ctrl = panel.loc[panel["cohort_mindex"].isna(), "channel_id"].nunique()
    event_study_figure(
        cs.event_study,
        out / "figures/real_event_study.png",
        title="Cheaper subs sold more subs: local pricing and paid subscriptions",
        subtitle=f"Callaway–Sant'Anna event study, log paid subs; {n_treat} treated channels "
        f"(TR/MX/BR/ES/AR) vs {n_ctrl} US-audience controls; uniform 95% bands, anticipation=1",
    )

    # --- dose response ------------------------------------------------------
    dr = estimate_dose_response(
        panel, config, anticipation=anticipation, mature=True, n_boot=args.n_boot
    )
    dr.by_country.to_csv(out / "estimates/dose_response.csv", index=False)
    dose_response_figure(
        dr.by_country,
        dr.epsilon,
        out / "figures/dose_response.png",
        subtitle="Mature-window ATT (6-12 months post) per country vs log price change at rollout; "
        f"WLS through origin: elasticity = {dr.epsilon:.2f}",
    )

    # --- TWFE benchmark + Bacon (synthetic, balanced) -----------------------
    twfe = run_twfe(panel, outcome="log_subs")
    synth = pd.read_parquet(args.synth_panel)
    bacon = bacon_decompose(synth, outcome="log_subs")
    bacon.by_type().to_csv(out / "estimates/bacon_synthetic.csv", index=False)

    summary = {
        "n_channels": int(panel["channel_id"].nunique()),
        "n_treated": int(n_treat),
        "n_controls": int(n_ctrl),
        "anticipation": anticipation,
        "cs_overall_att": float(cs.simple["att"].iloc[0]),
        "cs_overall_se": float(cs.simple["se"].iloc[0]),
        "twfe_att": twfe.att,
        "elasticity_mature": dr.epsilon,
        "elasticity_se_analytic": dr.se_analytic,
        "elasticity_ci_boot": list(dr.ci_boot) if dr.ci_boot else None,
        "n_boot_effective": len(dr.boot_draws) if dr.boot_draws is not None else 0,
        "dose_response": dr.by_country.to_dict(orient="records"),
        "bacon_synthetic_by_type": bacon.by_type().to_dict(orient="records"),
    }
    (out / "estimates/final_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
