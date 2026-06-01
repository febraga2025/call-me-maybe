
UV_BIN = $(HOME)/.local/bin/uv
PYTHON = $(UV_BIN) run python

all: install run

install:
	@if [ ! -f $(UV_BIN) ]; then \
		echo "uv not found. Installing locally via curl..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Synchronizing project dependencies..."
	$(UV_BIN) sync

run:
	@echo "Starting the program..."
	$(PYTHON) -m src --visual

clean:
	rm -rf .venv __pycache__ src/__pycache__ .mypy_cache .pytest_cache

lint:
	$(PYTHON) -m flake8 src
	$(PYTHON) -m mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--follow-imports=skip --exclude '^llm_sdk/' \
		--disallow-untyped-defs --check-untyped-defs src


test:
	$(PYTHON) -m pytest -q


.PHONY: all install run clean lint test