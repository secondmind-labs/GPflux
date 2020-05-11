LIB_NAME = gpflux
TESTS_NAME = tests
ARCHS_NAME = experiments
LINT_NAMES = $(LIB_NAME) $(ARCHS_NAME) $(TESTS_NAME)
TYPE_NAMES = $(LIB_NAME) $(ARCHS_NAME) $(TESTS_NAME)

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

LINT_IGNORES = "W503,E203"


build:
	tox

install:
	pip install -e .
	pip install -r tests_requirements.txt

format-check:
	black --check $(LINT_NAMES)

format:
	black $(LINT_NAMES)

lint:
	flake8 --per-file-ignores=$(LINT_FILE_IGNORES) --extend-ignore=$(LINT_IGNORES) $(LINT_NAMES)

types:
	mypy $(TYPE_NAMES)

static-tests: format-check lint types

pytest:
	pytest --cov=$(LIB_NAME) \
	       --cov-report html:cover_html \
	       --cov-config .coveragerc \
	       --cov-report term \
	       --cov-report xml \
	       --cov-fail-under=97 \
	       --junitxml=reports/junit.xml \
	       -v --tb=short \
	       $(TESTS_NAME)

.PHONY: tests
tests: static-tests pytest
