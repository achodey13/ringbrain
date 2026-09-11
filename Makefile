.PHONY: setup lint typecheck test check chat eval run

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	cp -n .env.example .env || true

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest -v

check: lint typecheck test

chat:
	python -m ringbrain.cli

eval:
	python -m ringbrain.eval.harness

run:
	uvicorn ringbrain.api.main:app --reload
