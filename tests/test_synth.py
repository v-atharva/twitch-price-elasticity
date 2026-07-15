import pandas as pd
import pytest

from src.panel.months import to_mindex
from src.panel.schema import validate_panel


def test_shape_and_schema(panel: pd.DataFrame, config) -> None:
    n = int(config.synth["n_channels"])
    n_months = to_mindex(config.panel_end) - to_mindex(config.panel_start) + 1
    assert len(panel) == n * n_months
    validate_panel(panel)  # raises on violation


def test_cohort_shares(panel: pd.DataFrame, config) -> None:
    shares = config.synth["cohort_shares"]
    got = panel.groupby("cohort_name")["channel_id"].nunique() / int(config.synth["n_channels"])
    for name, share in shares.items():
        assert got[name] == pytest.approx(share, abs=0.01)


def test_no_effect_before_treatment_or_for_controls(panel: pd.DataFrame) -> None:
    never = panel["cohort_mindex"].isna()
    assert (panel.loc[never, "true_effect"] == 0).all()
    pre = ~never & (panel["mindex"] < panel["cohort_mindex"])
    assert (panel.loc[pre, "true_effect"] == 0).all()
    post = ~never & (panel["mindex"] >= panel["cohort_mindex"])
    assert (panel.loc[post, "true_effect"] > 0).all()  # price cuts raise subs


def test_effect_proportional_to_dose(panel: pd.DataFrame, config) -> None:
    # at long horizons the effect per cohort approaches elasticity * dlogp
    eps = float(config.synth["true_elasticity"])
    mature = panel[~panel["cohort_mindex"].isna()]
    mature = mature[mature["mindex"] - mature["cohort_mindex"] >= 10]
    for name, grp in mature.groupby("cohort_name"):
        full = eps * config.cohort(str(name)).dlogp
        assert grp["true_effect"].mean() == pytest.approx(full, rel=0.02)


def test_schema_catches_duplicates(panel: pd.DataFrame) -> None:
    bad = pd.concat([panel, panel.head(5)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(bad)
