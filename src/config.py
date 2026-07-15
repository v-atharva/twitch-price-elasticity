"""Load and lightly validate config/study.yaml — the single source of truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.panel.months import to_mindex

DEFAULT_CONFIG_PATH = Path("config/study.yaml")


@dataclass(frozen=True)
class Cohort:
    name: str
    treat_month: str  # "YYYY-MM"
    dlogp: float  # log(new_price_usd / old_price_usd), negative = price cut
    example_countries: tuple[str, ...]

    @property
    def treat_mindex(self) -> int:
        return to_mindex(self.treat_month)


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    cohorts: tuple[Cohort, ...]
    panel_start: str
    panel_end: str

    def cohort(self, name: str) -> Cohort:
        for c in self.cohorts:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def synth(self) -> dict[str, Any]:
        return self.raw["synth"]

    @property
    def estimator(self) -> dict[str, Any]:
        return self.raw["estimator"]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    cohorts = tuple(
        Cohort(
            name=name,
            treat_month=str(spec["treat_month"]),
            dlogp=float(spec["dlogp"]),
            example_countries=tuple(spec.get("example_countries", [])),
        )
        for name, spec in raw["cohorts"].items()
    )
    start, end = str(raw["panel"]["start"]), str(raw["panel"]["end"])
    for c in cohorts:
        if not to_mindex(start) < c.treat_mindex <= to_mindex(end):
            raise ValueError(f"cohort {c.name} treat_month {c.treat_month} outside panel window")
    return Config(raw=raw, cohorts=cohorts, panel_start=start, panel_end=end)
