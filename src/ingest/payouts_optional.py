"""Pluggable interface for the October 2021 leaked creator-payout dataset.

HARD-DISABLED. This dataset is only usable with the project owner's explicit
opt-in (config `sources.payouts_dataset_enabled: true` AND a local file path
the owner supplies themselves). This module never downloads anything. If
excluded, the analysis stands on sub-count data alone — by design.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def load_payouts(config: dict[str, Any]) -> pd.DataFrame:
    sources = config.get("sources", {})
    if not sources.get("payouts_dataset_enabled", False):
        raise RuntimeError(
            "The leaked payout dataset is disabled. It requires explicit opt-in from the "
            "project owner (sources.payouts_dataset_enabled) and a locally supplied file. "
            "This code will never download it."
        )
    path = sources.get("payouts_dataset_path")
    if not path:
        raise RuntimeError("payouts_dataset_enabled is true but no payouts_dataset_path given")
    raise NotImplementedError(
        "Integration is intentionally unimplemented until the owner opts in (see PLAN.md)."
    )
