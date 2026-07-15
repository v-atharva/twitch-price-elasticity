"""Cross-check the Python CS implementation against R's `did` package.

R's `did` is Callaway & Sant'Anna's reference implementation. This runs the
identical spec (doubly robust, never-treated controls, varying base period,
same covariates) in both stacks on the same panel and asserts the point
estimates agree within tolerance. Wired into CI on the synthetic panel.

Rscript resolution order: $RSCRIPT_BIN, PATH, conda env `rdid`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.estimate.cs import run_cs
from src.panel.schema import validate_panel

ATT_TOL = 0.01  # log points; same estimator, same data — should agree tightly


def find_rscript() -> str:
    if env := os.environ.get("RSCRIPT_BIN"):
        return env
    if path := shutil.which("Rscript"):
        return path
    conda = Path.home() / "miniconda3/envs/rdid/bin/Rscript"
    if conda.exists():
        return str(conda)
    raise FileNotFoundError("Rscript not found (set RSCRIPT_BIN or install R)")


def export_for_r(panel: pd.DataFrame, path: Path) -> None:
    df = panel.copy()
    df["id"] = pd.factorize(df["channel_id"])[0] + 1
    df["g"] = df["cohort_mindex"].fillna(0).astype(int)  # did convention: 0 = never
    df[["id", "mindex", "g", "log_subs", "pre_size", "cat_share_games"]].to_csv(path, index=False)


def crosscheck(panel_path: str, anticipation: int) -> dict:
    config = load_config()
    panel = validate_panel(pd.read_parquet(panel_path))

    py = run_cs(panel, config, outcome="log_subs", anticipation=anticipation)
    py_att = float(py.simple["att"].iloc[0])

    with tempfile.TemporaryDirectory() as td:
        csv_in, csv_out = Path(td) / "panel.csv", Path(td) / "r_results.csv"
        export_for_r(panel, csv_in)
        subprocess.run(
            [find_rscript(), "scripts/crosscheck.R", str(csv_in), str(csv_out), str(anticipation)],
            check=True,
            capture_output=True,
            text=True,
        )
        r_res = pd.read_csv(csv_out)

    r_att = float(r_res.loc[r_res["kind"] == "simple", "att"].iloc[0])
    r_event = r_res[r_res["kind"] == "event"].set_index("rel_period")["att"]
    py_event = py.event_study.set_index("rel_period")["att"]
    common = sorted(set(r_event.index) & set(py_event.index))
    event_mad = float((py_event.loc[common] - r_event.loc[common]).abs().mean())

    return {
        "python_att": py_att,
        "r_att": r_att,
        "abs_diff": abs(py_att - r_att),
        "event_study_mean_abs_diff": event_mad,
        "n_event_points_compared": len(common),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/synthetic_panel.parquet")
    parser.add_argument("--anticipation", type=int, default=0)
    args = parser.parse_args()

    result = crosscheck(args.panel, args.anticipation)
    for k, v in result.items():
        print(f"{k}: {v}")
    if result["abs_diff"] > ATT_TOL or result["event_study_mean_abs_diff"] > 2 * ATT_TOL:
        print(f"FAIL: Python and R disagree beyond tolerance ({ATT_TOL})", file=sys.stderr)
        sys.exit(1)
    print("OK: Python `differences` matches R `did` within tolerance")


if __name__ == "__main__":
    main()
