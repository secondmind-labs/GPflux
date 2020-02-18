LIB_NAME = gpflux
TESTS_NAME = tests

# the --per-file-ignores are to ignore "unused import" warnings in __init__.py files (F401)
# the F403 ignore in gpflux/__init__.py allows the `from .<submodule> import *`
LINT_FILE_IGNORES = "$(LIB_NAME)/__init__.py:F401,F403 \
                     $(LIB_NAME)/layers/__init__.py:F401 \
                     $(LIB_NAME)/models/__init__.py:F401 \
                     $(LIB_NAME)/initializers/__init__.py:F401"


build:
	tox
install:
	pip install -e .
	pip install -r tests_requirements.txt
format-check:
	black --check $(LIB_NAME) $(TESTS_NAME)
format:
	black $(LIB_NAME) $(TESTS_NAME)
full-test: format-check types test
lint:
	flake8 --per-file-ignores=$(LINT_FILE_IGNORES) $(LIB_NAME) $(TESTS_NAME)
types:
	mypy $(LIB_NAME)
test:
	pytest --cov=$(LIB_NAME) \
	       --cov-report html:cover_html \
	       --cov-config .coveragerc \
	       --cov-report term \
	       --cov-report xml \
	       --cov-fail-under=97 \
	       --junitxml=reports/junit.xml \
	       -v --tb=short \
	       $(TESTS_NAME)
