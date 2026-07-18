.PHONY: test
test:
	uv run pytest

.PHONY: lint
lint:
	uv run mypy
	uv run pylint src/optisample notebooks/utils

.PHONY: format
format:
	uv run isort .
	uv run black .

.PHONY: explore
explore:
	uv run marimo edit notebooks/explore.py
