PYTHON ?= python
IMAGE ?= faq-bot:latest

.PHONY: help install run token index seed sync test lint format format-check check docker-build docker-up docker-index docker-down clean

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | sed 's/:.*## / - /'

install: ## Install the editable package with dev, store, ML and PDF extras
	$(PYTHON) -m pip install -e ".[dev,stores,ml,pdf]"

run: ## Run the API on :8000
	$(PYTHON) -m chat_search serve

token: ## Mint a development patient JWT for the default tenant
	$(PYTHON) -m chat_search token patient default

index: ## Index DATA_DIR PDFs into Qdrant (separate from serving)
	$(PYTHON) -m chat_search index-documents

seed: ## Load sample providers into Elasticsearch
	$(PYTHON) -m chat_search seed-providers default

sync: ## Run one provider sync (needs ES_URL, ES_SYNC_API_KEY, PROVIDERS_*)
	$(PYTHON) -m chat_search sync-providers

test: ## Run the unit test suite
	$(PYTHON) -m pytest -q

lint: ## Lint with ruff
	$(PYTHON) -m ruff check .

format: ## Format with ruff
	$(PYTHON) -m ruff format .

format-check: ## Verify formatting without changing files
	$(PYTHON) -m ruff format --check .

check: lint format-check test ## Run everything CI runs

docker-build: ## Build the service image
	docker build -t $(IMAGE) .

docker-up: ## Start the local stack
	docker compose up --build

docker-index: ## Run the one-shot document index job in compose
	docker compose --profile indexing run --rm indexer

docker-down: ## Stop the local stack
	docker compose down

clean: ## Remove local caches and build artifacts
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
