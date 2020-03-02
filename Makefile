LIB_NAME = gpflux
TESTS_NAME = tests

# the --per-file-ignores are to ignore "unused import" warnings in __init__.py files (F401)
# the F403 ignore in gpflux/__init__.py allows the `from .<submodule> import *`
LINT_FILE_IGNORES = "$(LIB_NAME)/__init__.py:F401,F403 \
                     $(LIB_NAME)/layers/__init__.py:F401 \
                     $(LIB_NAME)/models/__init__.py:F401 \
                     $(LIB_NAME)/initializers/__init__.py:F401 \
                     $(LIB_NAME)/encoders/__init__.py:F401"

LINT_IGNORES = "W503,E203"


build:
	tox

install:
	pip install -e .
	pip install -r tests_requirements.txt

format-check:
	black --check $(LIB_NAME) $(TESTS_NAME)

format:
	black $(LIB_NAME) $(TESTS_NAME)

lint:
	flake8 --per-file-ignores=$(LINT_FILE_IGNORES) --extend-ignore=$(LINT_IGNORES) $(LIB_NAME) $(TESTS_NAME)

types:
	mypy $(LIB_NAME)

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
