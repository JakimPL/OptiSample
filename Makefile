.PHONY: test lint format

test:
	uv run pytest

lint:
	uv run mypy
	uv run pylint src/optisample

format:
	uv run isort .
	uv run black .
