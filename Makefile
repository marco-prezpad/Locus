.PHONY: ci lint format typecheck test run \
        docker-build docker-up docker-down docker-shell \
        docker-gpu-build docker-gpu-up docker-gpu-shell

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

fix:
	@mkdir -p logs/ruff-check
	@mkdir -p logs/ruff-format
	@ts=$$(date +%Y%m%d-%H%M%S); \
	uv run ruff check --diff src tests > logs/ruff-check/ruff-check-diff-$$ts.log 2>&1 || true; \
	uv run ruff check --fix src tests; \
	uv run ruff format --diff src tests > logs/ruff-format/ruff-format-diff-$$ts.log 2>&1 || true; \
	uv run ruff format src tests

fix-unsafe:
	@mkdir -p logs/ruff-check
	@mkdir -p logs/ruff-format
	@ts=$$(date +%Y%m%d-%H%M%S); \
	uv run ruff check --diff src tests > logs/ruff-check/ruff-check-diff-$$ts.log 2>&1 || true; \
	uv run ruff check --fix --unsafe-fixes src tests; \
	uv run ruff format --diff src tests > logs/ruff-format/ruff-format-diff-$$ts.log 2>&1 || true; \
	uv run ruff format src tests

typecheck:
	uv run pyright

test:
	uv run pytest

test-fast:
	uv run pytest --no-cov -x

test-file:
	uv run pytest $(FILE)

ci: lint format typecheck test
	@echo "CI local green"

run:
	uv run python -m locus

docker-build:
	docker compose build locus-dev

docker-up:
	docker compose up -d locus-dev

docker-down:
	docker compose down

docker-shell:
	docker compose exec locus-dev bash

docker-gpu-build:
	docker compose --profile gpu build locus-dev-gpu

docker-gpu-up:
	docker compose --profile gpu up -d locus-dev-gpu

docker-gpu-shell:
	docker compose exec locus-dev-gpu bash
