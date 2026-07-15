"""Month arithmetic: the panel's time index is an integer count of months.

Estimators need an integer time axis; humans need "2021-05". `mindex` counts
months since 2000-01 (0 = 2000-01), so it is stable across config changes to
the panel window.
"""

from __future__ import annotations

EPOCH_YEAR = 2000


def to_mindex(month: str) -> int:
    """ "YYYY-MM" -> integer months since 2000-01."""
    y, m = month.split("-")
    year, mon = int(y), int(m)
    if not 1 <= mon <= 12:
        raise ValueError(f"bad month: {month!r}")
    return (year - EPOCH_YEAR) * 12 + (mon - 1)


def from_mindex(mindex: int) -> str:
    """Integer months since 2000-01 -> "YYYY-MM"."""
    year, mon = divmod(mindex, 12)
    return f"{year + EPOCH_YEAR:04d}-{mon + 1:02d}"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of "YYYY-MM" from start to end."""
    a, b = to_mindex(start), to_mindex(end)
    if b < a:
        raise ValueError(f"end {end!r} before start {start!r}")
    return [from_mindex(i) for i in range(a, b + 1)]
