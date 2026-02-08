.PHONY: help lint format format-check test test-quick test-cov clean build install-dev all

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint: ## Run ruff linter
	uv run ruff check src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/

format-check: ## Check code formatting
	uv run ruff format --check src/ tests/

test: ## Run tests with verbose output
	uv run pytest tests/ -v

test-quick: ## Run tests, stop on first failure
	uv run pytest tests/ -x -q

test-cov: ## Run tests with coverage report
	uv run pytest tests/ --cov=copex --cov-report=term-missing

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

build: ## Build the package
	uv run python -m build

install-dev: ## Install in development mode
	pip install -e ".[dev]"

all: lint format-check test ## Run lint, format check, and tests
