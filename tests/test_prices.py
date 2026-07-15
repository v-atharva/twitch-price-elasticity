"""Anchor tests for the curated cohort/price table.

The anchors are externally verified facts (Twitch blog + contemporaneous
press); if an edit to the CSV breaks one, the table is wrong, not the test.
"""

import math

import pandas as pd
import pytest

from src.panel.prices import _treat_month_g, load_price_table


@pytest.fixture(scope="module")
def table():
    return load_price_table()


def row(table, code):
    return table[table["country_code"] == code].iloc[0]


def test_turkey_anchor(table) -> None:
    tr = row(table, "TR")
    assert tr["treat_date"] == "2021-05-20"
    assert tr["new_price"] == 9.90 and tr["new_currency"] == "TRY"
    # ~-76% at rollout FX (8.3679 TRY/USD): new price ~= $1.18 vs $4.99
    assert tr["new_price_usd_at_rollout"] == pytest.approx(1.18, abs=0.02)
    assert tr["pct_price_change"] == pytest.approx(-0.763, abs=0.01)
    assert tr["treat_month_g"] == "2021-06"  # May 20 -> only 12 treated days in May
    assert bool(tr["partial_first_month"])


def test_mexico_anchor(table) -> None:
    mx = row(table, "MX")
    assert mx["treat_date"] == "2021-05-20"
    assert mx["new_price_usd_at_rollout"] == pytest.approx(2.41, abs=0.02)
    assert mx["pct_price_change"] == pytest.approx(-0.517, abs=0.01)
    assert mx["treat_month_g"] == "2021-06"


def test_brazil_anchor(table) -> None:
    br = row(table, "BR")
    assert br["treat_date"] == "2021-07-27"
    assert br["treat_month_g"] == "2021-08"
    assert br["pct_price_change"] == pytest.approx(-0.695, abs=0.01)
    assert br["provenance_quality"] == "indirect"  # upgrade at M5


def test_argentina_anchor(table) -> None:
    ar = row(table, "AR")
    assert ar["treat_date"] == "2021-07-27"
    assert ar["treat_month_g"] == "2021-08"
    # Infobae 2021-07-27: USD 4.99 -> USD 1.99 ("60% mas barato")
    assert ar["pct_price_change"] == pytest.approx(-0.601, abs=0.005)


def test_us_never_treated(table) -> None:
    us = row(table, "US")
    assert pd.isna(us["treat_month_g"])
    assert us["dlogp"] == 0.0


def test_same_currency_dose_needs_no_fx(table) -> None:
    au = row(table, "AU")
    assert au["dlogp"] == pytest.approx(math.log(7.99 / 8.99))
    tw = row(table, "TW")  # no ECB TWD rate, but local-ratio dose is exact
    assert tw["dlogp"] == pytest.approx(math.log(77 / 149))
    assert pd.isna(tw["new_price_usd_at_rollout"])


def test_blog_percent_decreases_match(table) -> None:
    # % decreases printed in the 2021-08-05 Twitch blog table
    blog = {"AU": -0.11, "NZ": -0.20, "TW": -0.48, "KR": -0.24, "TH": -0.57, "SG": -0.36}
    for code, pct in blog.items():
        assert row(table, code)["pct_price_change"] == pytest.approx(pct, abs=0.01), code


def test_treat_month_rule() -> None:
    assert _treat_month_g("2021-05-20") == "2021-06"
    assert _treat_month_g("2021-07-27") == "2021-08"
    assert _treat_month_g("2021-08-05") == "2021-08"
    assert _treat_month_g("2021-08-01") == "2021-08"
    assert _treat_month_g("2021-12-25") == "2022-01"


def test_all_treated_have_negative_dose_and_month(table) -> None:
    treated = table[table["treat_date"].notna()]
    assert (treated["dlogp"] < 0).all()
    assert treated["treat_month_g"].notna().all()
