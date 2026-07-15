"""Goodman-Bacon decomposition sanity checks on the synthetic (balanced) panel."""

import pytest

from src.estimate.bacon import bacon_decompose
from src.estimate.twfe import run_twfe


@pytest.fixture(scope="module")
def bacon(panel):
    return bacon_decompose(panel, outcome="log_subs")


def test_weights_sum_to_one(bacon) -> None:
    assert bacon.comparisons["weight"].sum() == pytest.approx(1.0, abs=1e-9)
    assert (bacon.comparisons["weight"] >= 0).all()


def test_forbidden_comparisons_carry_weight(bacon) -> None:
    by = bacon.by_type().set_index("type")
    assert by.loc["later_vs_earlier(forbidden)", "weight"] > 0.005


def test_decomposition_tracks_twfe(panel, bacon) -> None:
    # the identity is exact for OLS-defined 2x2s; our mean-difference 2x2s are a
    # close approximation on a balanced panel with iid noise
    twfe = run_twfe(panel, outcome="log_subs")
    assert bacon.twfe == pytest.approx(twfe.att, abs=0.05)


def test_forbidden_estimates_biased_down(bacon, config) -> None:
    # with ramping effects, already-treated controls absorb part of the effect:
    # forbidden comparisons should sit below the clean treated-vs-never ones
    by = bacon.by_type().set_index("type")
    assert (
        by.loc["later_vs_earlier(forbidden)", "avg_estimate"]
        < by.loc["treated_vs_never", "avg_estimate"]
    )
