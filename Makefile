# QuantOS -- one command per thing you might want to do.
#
# `make help` lists everything. `make setup` gets a working environment from a
# clean checkout; `make check` runs exactly what CI runs, so a green local run
# means a green pipeline rather than "probably".

.DEFAULT_GOAL := help
SHELL := /bin/bash
PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help
help: ## Show this help
	@echo "QuantOS -- make targets"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "First time here?  make setup && make demo"

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --quiet --upgrade pip

.PHONY: setup
setup: $(BIN)/python ## Create the venv and install everything (one command, clean checkout to working)
	$(BIN)/pip install --quiet -e ".[dev,test]"
	@echo
	@echo "Ready. Try:  make demo   or   $(BIN)/quantos research --ticker NVDA"

.PHONY: setup-hooks
setup-hooks: setup ## Install pre-commit hooks so bad commits fail before CI does
	$(BIN)/pre-commit install

# --------------------------------------------------------------------------- #
# The checks CI runs, in the order CI runs them
# --------------------------------------------------------------------------- #
.PHONY: check
check: lint types test claims links ## Everything CI runs. Green here means green there.
	@echo
	@echo "All checks passed."

.PHONY: lint
lint: ## ruff check + format --check
	$(BIN)/ruff check src tests benchmarks scripts
	$(BIN)/ruff format --check src tests

.PHONY: format
format: ## Apply ruff formatting and autofixes
	$(BIN)/ruff check --fix src tests benchmarks scripts
	$(BIN)/ruff format src tests benchmarks scripts

.PHONY: types
types: ## mypy --strict over the package
	$(BIN)/mypy

.PHONY: test
test: ## Full suite, including every docstring example
	$(BIN)/pytest -q

.PHONY: test-fast
test-fast: ## Suite minus the slow simulation tests, for the inner loop
	$(BIN)/pytest -q -x -m "not slow" --ignore=tests/sim

.PHONY: coverage
coverage: ## Suite with a coverage report
	$(BIN)/pytest -q --cov=quantos --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

.PHONY: claims
claims: ## Re-derive every number this repository documents
	$(BIN)/python scripts/verify_claims.py

.PHONY: links
links: ## Fail on a broken local link in any markdown file
	$(BIN)/python scripts/check_links.py

.PHONY: mutation
mutation: ## Mutation testing -- are the tests load-bearing? (slow)
	$(BIN)/python scripts/mutation_test.py

# --------------------------------------------------------------------------- #
# Running it
# --------------------------------------------------------------------------- #
.PHONY: demo
demo: ## Tour every subsystem, about two minutes
	$(BIN)/quantos demo

.PHONY: serve
serve: ## Local research viewer at http://localhost:8000
	$(BIN)/quantos serve

.PHONY: site
site: ## Build the static site into ./site
	$(BIN)/python scripts/build_site.py --out site
	@echo "Open site/index.html"

.PHONY: record-demo
record-demo: ## Re-record the animated terminal demo from a real session
	$(BIN)/python scripts/record_demo.py --out docs/demo.svg

.PHONY: gallery
gallery: ## Regenerate the figure gallery from real data
	$(BIN)/python scripts/build_gallery.py

# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
.PHONY: docker-build
docker-build: ## Build the container image
	docker build -t quantos:local .

.PHONY: docker-test
docker-test: docker-build ## Run the suite inside the container
	docker run --rm quantos:local pytest -q

.PHONY: docker-serve
docker-serve: docker-build ## Serve the viewer from the container on :8000
	docker run --rm -p 8000:8000 quantos:local quantos serve --host 0.0.0.0

# --------------------------------------------------------------------------- #
.PHONY: clean
clean: ## Remove build, cache and coverage artefacts (not the venv)
	rm -rf build dist *.egg-info htmlcov .coverage coverage.xml site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean ## Also remove the virtual environment
	rm -rf $(VENV)
