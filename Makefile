install:
	pip install -r requirements.txt
	pip install -e .
lint:
	flake8 gpflux tests
types:
	mypy gpflux
basic_test:
	pytest --cov=gpflux tests --ignore=tests/test_invariant.py
full_test:
	pytest --cov=gpflux --cov-report html:cover_html -v --tb=short --junitxml=nosetests.xml --cov-config .coveragerc --cov-report term --cov-report xml tests
build:
	tox
profile:
	python gpflux/profile.py