# Variáveis
UV_BIN = $(HOME)/.local/bin/uv
PYTHON = $(UV_BIN) run python

all: install run

# 1. INSTALL: Sincroniza o projeto baseado no pyproject.toml
install:
	@if [ ! -f $(UV_BIN) ]; then \
		echo "uv não encontrado. Instalando localmente via curl..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Sincronizando dependências do projeto..."
	$(UV_BIN) sync

# 2. RUN: Executa o programa como módulo
run:
	@echo "Iniciando o programa..."
	$(PYTHON) -m src

# 3. CLEAN[cite: 1]
clean:
	rm -rf .venv __pycache__ src/__pycache__ .mypy_cache .pytest_cache

# 4. LINT (Requisito obrigatório)[cite: 1]
lint:
	$(PYTHON) -m flake8 src
	$(PYTHON) -m mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs src


.PHONY: all install run clean lint