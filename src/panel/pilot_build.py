"""Build the pilot channel-month panel from harvested TwitchTracker chart data.

Inputs (data/interim/pilot/): pilot_dump_1.json (full-format records with
`rows` = [month, paid, prime]) and pilot_delta.json (compact records with
`s`=start month, `p`=paid array, `q`=prime array, optional `m`=explicit month
codes "YYMM" when the series has gaps).

Outcome definitions:
  paid  = Tier1 + Tier2 + Tier3 + Undefined active subs (INCLUDES gift subs —
          the tier chart does not separate them; documented threat)
  prime = Prime subs: free via Amazon Prime, price-insensitive — the
          within-channel placebo series (fills the schema's `followers` slot
          for the pilot; real follower series arrive with SullyGnome at M5).

Attrition rule: months after a channel's paid count collapses below
MIN_ACTIVE (channel left Twitch / stopped streaming) are censored, from the
first sub-threshold month onward. Zeros in log space are not "no demand",
they are exit.

Cohort assignment: country (curated map, provisional for es-language) ->
treat_month_g from the price table (>=15-treated-days rule).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.panel.months import to_mindex
from src.panel.prices import load_price_table
from src.panel.schema import validate_panel

PILOT_DIR = Path("data/interim/pilot")
COUNTRY_MAP = Path("data/reference/channel_country_map.csv")
MIN_ACTIVE = 10  # paid subs; below this a big channel has effectively exited
PANEL_START, PANEL_END = "2020-01", "2022-12"

COUNTRY_LANGUAGE = {
    "TR": "tr", "MX": "es", "BR": "pt", "ES": "es", "AR": "es", "US": "en",
}  # fmt: skip


def _mm(code: str) -> str:
    """'2103' -> '2021-03'."""
    return f"20{code[:2]}-{code[2:]}"


def load_records() -> dict[str, list[tuple[str, float | None, float | None]]]:
    """channel -> [(month, paid, prime), ...] from both dump formats."""
    out: dict[str, list[tuple[str, float | None, float | None]]] = {}
    d1 = json.loads((PILOT_DIR / "pilot_dump_1.json").read_text())
    for ch, rec in d1.items():
        if isinstance(rec, dict) and rec.get("rows"):
            out[ch] = [(m, p, q) for m, p, q in rec["rows"]]
    delta = json.loads((PILOT_DIR / "pilot_delta.json").read_text())
    scaleup = PILOT_DIR / "scaleup_delta.json"
    if scaleup.exists():
        delta.update(json.loads(scaleup.read_text()))
    for ch, rec in delta.items():
        if not isinstance(rec, dict):
            continue
        n = len(rec["p"])
        if "m" in rec:
            months = [_mm(c) for c in rec["m"]]
        else:
            start = to_mindex(rec["s"])
            from src.panel.months import from_mindex

            months = [from_mindex(start + i) for i in range(n)]
        out[ch] = list(zip(months, rec["p"], rec["q"], strict=True))
    return out


def build_panel() -> pd.DataFrame:
    records = load_records()
    cmap = pd.read_csv(COUNTRY_MAP)
    country = dict(zip(cmap.channel, cmap.country, strict=True))
    confidence = dict(zip(cmap.channel, cmap.confidence, strict=True))
    prices = load_price_table()
    g_by_country = dict(zip(prices.country_code, prices.treat_mindex_g, strict=True))

    # channels not in the map are assumed English/US-audience controls if they
    # came from the English roster; the roster file is the source of truth
    roster = pd.read_csv("data/interim/pilot_roster_v2.csv")
    roster_lang = dict(zip(roster.channel, roster.language_expected, strict=True))
    # renames + scale-up tranche channels not present in the pilot roster file
    roster_lang.setdefault("cristinini", "es")
    roster_lang.setdefault("alkapone", "es")
    roster_lang.setdefault("coringa", "pt")
    scaleup_en = (
        "forsen", "cohhcarnage", "39daph", "atrioc", "botezlive", "boxbox", "chocotaco",
        "clintstevens", "cdawg", "distortion2", "dogdog", "gothamchess", "gmhikaru",
        "itshafu", "kitboga", "lilypichu", "pokelawls", "scarra",
    )  # fmt: skip
    for ch in scaleup_en:
        roster_lang.setdefault(ch, "en")

    rows = []
    lo, hi = to_mindex(PANEL_START), to_mindex(PANEL_END)
    for ch, series in records.items():
        ctry = country.get(ch)
        conf = confidence.get(ch, "")
        lang = roster_lang.get(ch, "?")
        if ctry is None:
            if lang == "en":
                ctry = "US"  # US-audience proxy; contamination documented
            else:
                continue  # non-EN channel with unknown country: excluded
        elif conf == "low":
            continue  # ambiguous country: excluded pending M5 verification
        g = g_by_country.get(ctry)
        g = None if pd.isna(g) else int(g)

        # attrition censoring: cut at first collapse
        series = sorted(series, key=lambda r: r[0])
        kept = []
        for m, paid, prime in series:
            if paid is not None and paid < MIN_ACTIVE:
                break
            kept.append((m, paid, prime))
        for m, paid, prime in kept:
            idx = to_mindex(m)
            if not lo <= idx <= hi or paid is None:
                continue
            rows.append(
                {
                    "channel_id": ch,
                    "month": m,
                    "mindex": idx,
                    "cohort_mindex": float(g) if g is not None else np.nan,
                    "cohort_name": f"g{g}" if g is not None else "never",
                    "country": ctry,
                    "language": COUNTRY_LANGUAGE.get(ctry, lang),
                    "subs": float(paid),
                    "log_subs": float(np.log(paid)),
                    "followers": float(prime) if prime else np.nan,
                    "log_followers": float(np.log(prime)) if prime else np.nan,
                    "cat_share_games": 0.5,  # placeholder covariate until M5
                }
            )
    df = pd.DataFrame(rows)

    # pre-treatment size covariate: mean log paid before the earliest cohort month
    first_g = int(min(c for c in df["cohort_mindex"].dropna()))
    pre = df[df["mindex"] < first_g].groupby("channel_id")["log_subs"].mean()
    df["pre_size"] = df["channel_id"].map(pre)
    df = df[df["pre_size"].notna()]  # channels with zero pre-first-cohort data drop out
    return validate_panel(df.reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/processed/pilot_panel.parquet")
    args = parser.parse_args()
    df = build_panel()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df):,} rows, {df['channel_id'].nunique()} channels")
    print(df.groupby(["cohort_name", "country"])["channel_id"].nunique().to_string())


if __name__ == "__main__":
    main()
