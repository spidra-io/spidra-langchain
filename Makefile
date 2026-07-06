.PHONY: all install lint format tests integration_tests

all: help

help:
	@echo "Available targets:"
	@echo "  install           Install package and dev dependencies"
	@echo "  lint              Run ruff linter"
	@echo "  format            Run ruff formatter"
	@echo "  tests             Run unit tests"
	@echo "  integration_tests Run integration tests (requires SPIDRA_API_KEY)"

install:
	pip install -e ".[dev]"

lint:
	ruff check langchain_spidra tests

format:
	ruff format langchain_spidra tests

tests:
	pytest tests/unit_tests -v

integration_tests:
	pytest tests/integration_tests -v
