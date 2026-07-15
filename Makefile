# Every artifact is regenerable from here. `make all` runs the full synthetic
# pipeline end-to-end with zero network access.

CONFIG ?= config/study.yaml
SYNTH_PANEL := data/processed/synthetic_panel.parquet

.PHONY: synth estimate figures test lint typecheck fix all clean

synth:
	uv run python -m src.panel.synthetic --config $(CONFIG) --out $(SYNTH_PANEL)

estimate:
	uv run python -m src.estimate.run --config $(CONFIG) --panel $(SYNTH_PANEL)

figures: estimate  # figures are emitted by the estimation step

test:
	uv run pytest

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

fix:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy

crosscheck:
	uv run python -m src.estimate.r_crosscheck

all: synth estimate

clean:
	rm -rf data/processed/* data/interim/* outputs/estimates/*
