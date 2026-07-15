"""M7 driver: diagnostics & robustness on the real (pilot) panel.

Writes outputs/estimates/diagnostics.json, robustness.csv, and the
placebo/pretrend figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.diagnostics.placebos import fake_dates_on_controls, permutation_test, prime_placebo
from src.diagnostics.pretrends import leads_test, linear_trend_sensitivity
from src.diagnostics.robustness import run_battery
from src.estimate.cs import run_cs
from src.panel.schema import validate_panel
from src.report.figures import event_study_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/study.yaml")
    parser.add_argument("--panel", default="data/processed/pilot_panel.parquet")
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    panel = validate_panel(pd.read_parquet(args.panel))
    anticipation = int(config.estimator["anticipation_real"])
    out = Path(args.out_dir)
    (out / "estimates").mkdir(parents=True, exist_ok=True)

    cs = run_cs(panel, config, outcome="log_subs", anticipation=anticipation)

    # pre-trends
    pt = leads_test(cs.event_study)
    trend = linear_trend_sensitivity(cs.event_study)
    trend_out = {k: v for k, v in trend.items() if k != "adjusted"}

    # placebos
    prime = prime_placebo(panel, config, anticipation)
    event_study_figure(
        prime["event_study"],
        out / "figures/placebo_prime_event_study.png",
        title="Placebo: Prime subs are free — price should not move them",
        subtitle="Same design and sample, outcome = log Prime subs (price-insensitive)",
    )
    fake = fake_dates_on_controls(panel, config, anticipation)
    perm = permutation_test(panel, config, anticipation, n_perm=args.n_perm)

    robustness = run_battery(panel, config, anticipation)
    robustness.to_csv(out / "estimates/robustness.csv", index=False)

    diagnostics = {
        "pretrends_joint_test": pt,
        "linear_trend_sensitivity": trend_out,
        "placebo_prime": {"att": prime["att"], "se": prime["se"]},
        "placebo_fake_dates_on_controls": fake,
        "permutation_test": perm,
        "robustness": robustness.to_dict(orient="records"),
    }
    (out / "estimates/diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps({k: v for k, v in diagnostics.items() if k != "robustness"}, indent=2))
    print("\nrobustness battery:")
    print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()
