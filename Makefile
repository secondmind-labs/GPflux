install:
	pip install -e .
	pip install -r tests_requirements.txt
lint:
	# the --per-file-ignores are to ignore "unused import" warnings in __init__.py files (F401)
	# the F403 ignore in gpflux/__init__.py allows the `from .<submodule> import *`
	flake8 --per-file-ignores='gpflux/__init__.py:F401,F403 gpflux/layers/__init__.py:F401 gpflux/models/__init__.py:F401 gpflux/initializers/__init__.py:F401' gpflux tests
types:
	mypy gpflux
#unit_test:
#	pytest --cov=gpflux tests/unit
#integration_test:
#	pytest --cov=gpflux tests/integration
full_test:
	pytest --cov=gpflux --cov-report html:cover_html -v --tb=short --junitxml=reports/junit.xml --cov-config .coveragerc --cov-report term --cov-report xml tests --cov-fail-under=97
build:
	tox
#profile:
#	python gpflux/profile.py
