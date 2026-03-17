.PHONY: all format lint test test_unit test_integration test_e2e test_all evals eval_graph eval_multiturn eval_graph_qwen eval_graph_glm eval_multiturn_polite eval_multiturn_hacker test_watch test_watch_unit test_watch_integration test_watch_e2e test_profile extended_tests dev dev_ui test_supervisor_unit test_supervisor_integration test_supervisor_trace test_supervisor_e2e test_supervisor_all

# Default target executed when no arguments are given to make.
all: help


######################
# DEVELOPMENT
######################

dev:
	uv run langgraph dev --no-browser

dev_ui:
	uv run langgraph dev


######################
# TESTS
######################

test_supervisor_unit:
	uv run pytest tests/unit_tests/supervisor_agent -q

test_supervisor_integration:
	uv run pytest tests/integration_tests/supervisor_agent -q

test_supervisor_trace:
	uv run pytest tests/integration_tests/supervisor_agent/test_supervisor_internal_trajectory.py -q

test_supervisor_e2e:
	uv run pytest tests/e2e_tests/supervisor_agent -q


test_supervisor_all:
	uv run pytest tests/e2e_tests/supervisor_agent tests/integration_tests/supervisor_agent tests/unit_tests/supervisor_agent -q


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint:
	uv run python -m ruff check .
	uv run python -m ruff format src --diff
	uv run python -m ruff check --select I src
	uv run python -m mypy --strict src
	mkdir -p .mypy_cache && uv run python -m mypy --strict src --cache-dir .mypy_cache

lint_diff lint_package:
	uv run python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

lint_tests:
	uv run python -m ruff check tests --fix
	uv run python -m ruff format tests
	# Skip mypy for tests to allow more flexible typing

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	uv run codespell --toml pyproject.toml

spell_fix:
	uv run codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'DEVELOPMENT:'
	@echo 'dev                          - run langgraph dev without browser'
	@echo 'dev_ui                       - run langgraph dev with browser'
	@echo ''
	@echo 'TESTS:'
	@echo 'test_supervisor_unit         - run supervisor unit tests'
	@echo 'test_supervisor_integration  - run supervisor integration smoke tests'
	@echo 'test_supervisor_trace        - run supervisor internal trajectory assertions'
	@echo 'test_supervisor_e2e          - run supervisor e2e smoke tests'
	@echo 'test_supervisor_all          - run all supervisor tests (unit+integration+e2e)'
	@echo ''
	@echo 'CODE QUALITY:'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters (ruff + mypy on src/)'
	@echo 'lint_tests                   - run linters on tests (ruff only, no mypy)'
	@echo 'lint_package                 - run linters on src/ only'

