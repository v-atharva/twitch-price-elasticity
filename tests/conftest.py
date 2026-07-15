"""Shared fixtures: one synthetic panel + one CS fit per test session (they're slow)."""

from __future__ import annotations

import pytest

from src.config import Config, load_config
from src.estimate.cs import CSResult, run_cs
from src.panel.synthetic import generate


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config("config/study.yaml")


@pytest.fixture(scope="session")
def panel(config: Config):
    return generate(config)


@pytest.fixture(scope="session")
def cs_fit(panel, config: Config) -> CSResult:
    return run_cs(panel, config, outcome="log_subs")
