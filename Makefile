# Front door for the learning-to-read-out code release.
# Targets are ordered to mirror the reproduction pipeline:
#   install -> test -> audit -> extract -> train -> analyze
# `make help` lists them. This is a code-only release (no data, no metrics shipped):
# see docs/REPRODUCE.md for the figure -> metrics map and docs/DATA.md for the external assets
# each stage needs and where regenerated metrics land.

.DEFAULT_GOAL := help
.PHONY: help install test lint audit extract train analyze clean

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

# Plain sync (may re-resolve) so developers can pick up dependency changes;
# CI uses `uv sync --locked --extra dev` to validate the committed lockfile.
install:  ## Create the environment (uv sync with dev tools; builds the vendored SAE lib)
	uv sync --extra dev

test:  ## Run the test suite (CPU only, no data, no GPU)
	uv run --extra dev pytest -q

lint:  ## Ruff lint (same check as CI)
	uv run --extra dev ruff check .

audit:  ## Validate the experiments.yaml <-> experiments/ <-> paper/ correspondence
	uv run python scripts/audit/check_layout.py
	uv run python scripts/audit/gen_reproduce_index.py --check

extract:  ## Stage 1: build W_E snapshots from HF checkpoints (needs network + disk)
	@echo "Example: uv run python scripts/extract/extract_we_pythia.py --model EleutherAI/pythia-160m"
	@echo "Snapshot schedule: readout.core.model_specs.DEFAULT_STEPS_32. See docs/DATA.md."
	@echo "(W_U snapshots are auto-extracted on cache-miss by the train target.)"

train:  ## Stage 2: train a trajectory crosscoder from snapshots (needs GPU for >160M)
	@echo "Example: uv run python scripts/train/train_crosscoder.py \\"
	@echo "           --model EleutherAI/pythia-160m --expansion-factor 32.0 \\"
	@echo "           --batch-size 1024 --lr 5e-5 --n-epochs 300 --seed 0 \\"
	@echo "           --input-preprocess center_scale --amp-dtype fp32 \\"
	@echo "           --tanh-stretch 1.0 \\"
	@echo "           --output local_snapshots/wu_cc_pythia160m_seed0.pt"
	@echo "Settings-of-record for each published run: configs/runs/*.yaml"

analyze:  ## Stage 3: compute & persist the metrics behind each figure (no figures rendered)
	@echo "Per-experiment metric commands are in docs/REPRODUCE.md (mapped from experiments.yaml)."
	@echo "Code-only release: metrics are regenerated on run (gitignored), not shipped;"
	@echo "paper figures are rendered in the paper LaTeX tree from those metrics."

clean:  ## Remove caches and build artefacts (keeps source and results)
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
