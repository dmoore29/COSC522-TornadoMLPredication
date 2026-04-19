.PHONY: setup test build-extract inspect run lint

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

test:
	. .venv/bin/activate && pytest

build-extract:
	. .venv/bin/activate && python scripts/build_gsod_extract.py

inspect:
	. .venv/bin/activate && python -m tornado_ml.inspect_data --config configs/default.yaml

run:
	. .venv/bin/activate && python -m tornado_ml.main --config configs/default.yaml

lint:
	. .venv/bin/activate && ruff check .
