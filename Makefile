.PHONY: help install docs format check test check-and-test


LIB_NAME = gpflux
TESTS_NAME = tests
ARCHS_NAME = experiments
LINT_NAMES = $(LIB_NAME) $(TESTS_NAME) docs/notebooks
TYPE_NAMES = $(LIB_NAME)
SUCCESS='\033[0;32m'
UNAME_S = $(shell uname -s)

PANDOC_DEB = https://github.com/jgm/pandoc/releases/download/2.10.1/pandoc-2.10.1-1-amd64.deb

# the --per-file-ignores are to ignore "unused import" warnings in __init__.py files (F401)
# the F403 ignore in gpflux/__init__.py allows the `from .<submodule> import *`
LINT_FILE_IGNORES = "$(LIB_NAME)/__init__.py:F401,F403 \
                     $(LIB_NAME)/architectures/__init__.py:F401 \
                     $(LIB_NAME)/encoders/__init__.py:F401 \
                     $(LIB_NAME)/experiment_support/__init__.py:F401 \
                     $(LIB_NAME)/initializers/__init__.py:F401 \
                     $(LIB_NAME)/layers/__init__.py:F401 \
                     $(LIB_NAME)/layers/basis_functions/__init__.py:F401 \
                     $(LIB_NAME)/models/__init__.py:F401 \
                     $(LIB_NAME)/optimization/__init__.py:F401 \
                     $(LIB_NAME)/sampling/__init__.py:F401 \
                     $(LIB_NAME)/utils/__init__.py:F401"


help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


install:  ## Install repo for developement
	@echo "\n=== pip install package with dev requirements =============="
	pip install --upgrade --upgrade-strategy eager \
		-r notebook_requirements.txt \
		-r tests_requirements.txt \
		tensorflow${VERSION_TF} \
		keras${VERSION_KERAS} \
		tensorflow-probability${VERSION_TFP} \
		-e .

docs:  ## Build the documentation
	@echo "\n=== pip install doc requirements =============="
	pip install -r docs/docs_requirements.txt
	@echo "\n=== install pandoc =============="
ifeq ("$(UNAME_S)", "Linux")
	$(eval TEMP_DEB=$(shell mktemp))
	@echo "Checking for pandoc installation..."
	@(which pandoc) || ( echo "\nPandoc not found." \
	  && echo "Trying to install automatically...\n" \
	  && wget -O "$(TEMP_DEB)" $(PANDOC_DEB) \
	  && echo "\nInstalling pandoc using dpkg -i from $(PANDOC_DEB)" \
	  && echo "(If this step does not work, manually install pandoc, see http://pandoc.org/)\n" \
	  && sudo dpkg -i "$(TEMP_DEB)" \
	)
	@rm -f "$(TEMP_DEB)"
endif
ifeq ($(UNAME_S),Darwin)
	brew install pandoc
endif
	@echo "\n=== build docs =============="
	(cd docs ; make html)
	@echo "\n${SUCCESS}=== Docs are available at docs/_build/html/index.html ============== ${SUCCESS}"

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
	       --cov-fail-under=94 \
	       --junitxml=reports/junit.xml \
	       -v --tb=short --durations=10 \
	       $(TESTS_NAME)

quicktest:  ## Run the tests, start with the failing ones and break on first fail.
	pytest -vv -x --ff -rN -Wignore -s

check-and-test: check test  ## Run pytest and static tests
