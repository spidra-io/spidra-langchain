.PHONY: all format lint type test tests integration_tests help

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
PYTEST_EXTRA ?=

integration_test integration_tests: TEST_FILE=tests/integration_tests/

test tests:
	pip install -q -e ".[test]" && pytest $(PYTEST_EXTRA) $(TEST_FILE)

integration_test integration_tests:
	pip install -q -e ".[test]" && pytest -v --tb=short $(PYTEST_EXTRA) $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

PYTHON_FILES=.
MYPY_CACHE=.mypy_cache

lint:
	ruff check $(PYTHON_FILES)
	ruff format $(PYTHON_FILES) --diff

format:
	ruff format $(PYTHON_FILES)
	ruff check --fix $(PYTHON_FILES)

type:
	mkdir -p $(MYPY_CACHE) && mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

check_imports: $(shell find langchain_spidra -name '*.py')
	python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports              - check imports'
	@echo 'format                     - run code formatters'
	@echo 'lint                       - run linters'
	@echo 'type                       - run type checking'
	@echo 'test                       - run unit tests'
	@echo 'tests                      - run unit tests'
	@echo 'integration_tests          - run integration tests'
	@echo 'test TEST_FILE=<file>      - run all tests in file'
