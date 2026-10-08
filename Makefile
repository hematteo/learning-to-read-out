# Front door for the learning-to-read-out code release.
# Targets are ordered to mirror the reproduction pipeline:
#   install -> test -> figures -> extract -> train -> analyze
# `make help` lists them. `make figures` redraws the paper figures from the small data files in
# paper_figures/data/; the other stages recompute those data from public checkpoints. See
# docs/REPRODUCE.md for the figure -> metrics map and docs/REPRODUCE.md for the external assets.

.DEFAULT_GOAL := help
.PHONY: help install test lint figures extract train analyze clean

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

# Plain sync (may re-resolve) so developers can pick up dependency changes;
# CI uses `uv sync --locked --extra dev` to validate the committed lockfile.
install:  ## Create the environment (uv sync with dev tools and the figure extra)
	uv sync --extra dev --extra figures

test:  ## Run the test suite (CPU only, no data, no GPU)
	uv run --extra dev pytest -q

lint:  ## Ruff lint (same check as CI)
	uv run --extra dev ruff check .

figures:  ## Redraw the paper figures into paper_figures/out/ (CPU, about 30 s)
	uv run --extra figures python paper_figures/make_all.py

extract:  ## Stage 1: build W_E snapshots from HF checkpoints (needs network + disk)
	@echo "Example: uv run python scripts/extract/extract_we_pythia.py --model EleutherAI/pythia-160m"
	@echo "Snapshot schedule: readout.core.model_specs.DEFAULT_STEPS_32. See docs/REPRODUCE.md."
	@echo "(W_U snapshots are auto-extracted on cache-miss by the train target.)"

train:  ## Stage 2: train a trajectory crosscoder from snapshots (needs GPU for >160M)
	@echo "Example: uv run python scripts/train/train_crosscoder.py \\"
	@echo "           --model EleutherAI/pythia-160m --expansion-factor 32.0 \\"
	@echo "           --batch-size 1024 --lr 5e-5 --n-epochs 300 --seed 0 \\"
	@echo "           --input-preprocess center_scale --amp-dtype fp32 \\"
	@echo "           --tanh-stretch 1.0 --l1-warmup-fraction 0.05 \\"
	@echo "           --output local_snapshots/wu_cc_pythia160m_seed0.pt"
	@echo "Settings-of-record for each published run: configs/runs/*.yaml"

analyze:  ## Stage 3: compute & persist the metrics behind each figure (no figures rendered)
	@echo "Per-result commands are in docs/REPRODUCE.md."
	@echo "Recomputed metrics land under results/ and figures/ (gitignored); paper_figures/data/"
	@echo "README.md lists which script regenerates each file that make figures reads."

clean:  ## Remove caches and build artefacts (keeps source and results)
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
