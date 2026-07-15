"""Load and validate the curated cohort/price reference tables.

`data/reference/price_table.csv` is hand-curated with per-row provenance
(source URL + note + quality flag). This module derives the analysis columns:

- `usd rate` join: FX at the country's rollout date (ECB reference rates,
  pinned in fx_rates.csv — deliberately NOT live FX, so the dose is measured
  at the moment the price changed, before e.g. the late-2021 TRY collapse).
- `new_price_usd_at_rollout`, `old_price_usd_at_rollout`
- `dlogp` = log(new/old) computed in the ORIGINAL currency of the pair when
  old and new share a currency (FX cancels), else via USD at rollout FX.
- `pct_price_change` = exp(dlogp) - 1.

Cohort month assignment (`treat_month_g`): with a monthly panel, a country
treated mid-month has a partially-treated calendar month. Rule: g = the first
month with >= 15 treated days (May 20 -> g = 2021-06; Jul 27 -> 2021-08;
Aug 5 -> 2021-08). The preceding partial month is flagged
(`partial_first_month`) and must not serve as the DiD base period — handled
by the estimator's anticipation >= 1 setting. Sensitivity to this rule
(calendar-month assignment instead) is an M7 robustness check.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.panel.months import to_mindex

PRICE_TABLE = Path("data/reference/price_table.csv")
FX_RATES = Path("data/reference/fx_rates.csv")


def _treat_month_g(treat_date: str) -> str:
    """First month with >=15 treated days."""
    y, m, d = (int(x) for x in treat_date.split("-"))
    # days treated within the rollout month (all months treated as 30 for the rule)
    if 30 - d + 1 >= 15:
        return f"{y:04d}-{m:02d}"
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}-{m:02d}"


def load_price_table(
    price_path: str | Path = PRICE_TABLE, fx_path: str | Path = FX_RATES
) -> pd.DataFrame:
    prices = pd.read_csv(price_path, dtype={"treat_date": "string"})
    fx = pd.read_csv(fx_path)
    fx_map = {(r.date, r.currency): float(str(r.units_per_usd)) for r in fx.itertuples()}

    rows = []
    for r in prices.itertuples():
        treated = isinstance(r.treat_date, str) and bool(r.treat_date)
        old_p, new_p = float(str(r.old_price)), float(str(r.new_price))

        def usd(price: float, currency: str, date: str) -> float | None:
            rate = fx_map.get((date, currency))
            return price / rate if rate else None

        if treated:
            date = str(r.treat_date)
            old_usd = usd(old_p, str(r.old_currency), date)
            new_usd = usd(new_p, str(r.new_currency), date)
            if r.old_currency == r.new_currency:
                dlogp = math.log(new_p / old_p)
            elif old_usd is not None and new_usd is not None:
                dlogp = math.log(new_usd / old_usd)
            else:
                raise ValueError(f"{r.country_code}: cannot compute dose (missing FX)")
            g = _treat_month_g(date)
            first_day = int(date.split("-")[2])
            partial = first_day > 1 and _treat_month_g(date) != date[:7]
        else:
            old_usd = new_usd = old_p
            dlogp, g, partial = 0.0, None, False

        rows.append(
            {
                "country_code": r.country_code,
                "country": r.country,
                "region": r.region,
                "treat_date": r.treat_date if treated else None,
                "treat_month_g": g,
                "treat_mindex_g": to_mindex(g) if g else None,
                "partial_first_month": partial,
                "old_price": old_p,
                "old_currency": r.old_currency,
                "new_price": new_p,
                "new_currency": r.new_currency,
                "old_price_usd_at_rollout": old_usd,
                "new_price_usd_at_rollout": new_usd,
                "dlogp": dlogp,
                "pct_price_change": math.exp(dlogp) - 1.0,
                "provenance_quality": r.provenance_quality,
                "source_url": r.source_url,
            }
        )
    out = pd.DataFrame(rows)
    _validate(out)
    return out


def _validate(df: pd.DataFrame) -> None:
    if df["country_code"].duplicated().any():
        raise ValueError("duplicate country codes in price table")
    treated = df[df["treat_date"].notna()]
    if not (treated["dlogp"] < 0).all():
        bad = treated.loc[treated["dlogp"] >= 0, "country_code"].tolist()
        raise ValueError(f"treated countries with non-negative dose: {bad}")
    never = df[df["treat_date"].isna()]
    if not (never["dlogp"] == 0).all():
        raise ValueError("never-treated rows must have zero dose")
