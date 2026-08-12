.PHONY: test
test:
	uv run python -m pytest -n 6

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
