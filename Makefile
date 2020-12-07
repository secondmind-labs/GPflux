.PHONY: help install format check test check-and-test


LIB_NAME = gpflux
TESTS_NAME = tests
ARCHS_NAME = experiments
LINT_NAMES = $(LIB_NAME) $(TESTS_NAME) notebooks
TYPE_NAMES = $(LIB_NAME)

# the --per-file-ignores are to ignore "unused import" warnings in __init__.py files (F401)
# the F403 ignore in gpflux/__init__.py allows the `from .<submodule> import *`
LINT_FILE_IGNORES = "$(LIB_NAME)/__init__.py:F401,F403 \
                     $(LIB_NAME)/architectures/__init__.py:F401 \
                     $(LIB_NAME)/encoders/__init__.py:F401 \
                     $(LIB_NAME)/experiment_support/__init__.py:F401 \
                     $(LIB_NAME)/initializers/__init__.py:F401 \
                     $(LIB_NAME)/layers/__init__.py:F401 \
                     $(LIB_NAME)/models/__init__.py:F401 \
                     $(LIB_NAME)/optimization/__init__.py:F401 \
                     $(LIB_NAME)/utils/__init__.py:F401"


help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


install:  ## Install repo for developement
	@echo "\n=== pip install package with requirements =============="
	pip install -e .
	@echo "\n=== pip install dev requirements ======================"
	pip install -r requirements.txt
	@echo "\n=== pip install test requirements ======================"
	pip install -r tests_requirements.txt


format: ## Formats code with `black` and `isort`
	@echo "\n=== isort =============================================="
	isort .
	@echo "\n=== black =============================================="
	black --line-length=100 $(LINT_NAMES)


check: ## Runs all static checks such as code formatting checks, linting, mypy
	@echo "\n=== black (formatting) ================================="
	black --check --line-length=100 $(LINT_NAMES)
	@echo "\n=== flake8 (linting)===================================="
	flake8 --statistics \
		   --per-file-ignores=$(LINT_FILE_IGNORES) \
		   --exclude=.ipynb_checkpoints ./gpflux
	@echo "\n=== mypy (static type checking) ========================"
	mypy $(TYPE_NAMES)

test: ## Run unit and integration tests with pytest
	pytest --cov=$(LIB_NAME) \
	       --cov-report html:cover_html \
	       --cov-config .coveragerc \
	       --cov-report term \
	       --cov-report xml \
	       --cov-fail-under=97 \
	       --junitxml=reports/junit.xml \
	       -v --tb=short --durations=10 \
	       $(TESTS_NAME)

check-and-test: check test  ## Run pytest and static tests
