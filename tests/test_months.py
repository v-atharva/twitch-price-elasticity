import pytest

from src.panel.months import from_mindex, month_range, to_mindex


def test_roundtrip() -> None:
    for m in ["2000-01", "2020-12", "2021-05", "2022-01"]:
        assert from_mindex(to_mindex(m)) == m


def test_known_values() -> None:
    assert to_mindex("2000-01") == 0
    assert to_mindex("2021-05") == 256
    assert from_mindex(256) == "2021-05"


def test_month_range() -> None:
    r = month_range("2020-11", "2021-02")
    assert r == ["2020-11", "2020-12", "2021-01", "2021-02"]


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        to_mindex("2021-13")
    with pytest.raises(ValueError):
        month_range("2021-05", "2021-04")
