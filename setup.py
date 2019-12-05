#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'numpy==1.17.4',
    'tensorflow==2.0.0',
    'tensorflow-probability==0.8.0',
    'scipy==1.3.2',
    'scikit-learn==0.20.2',
    'matplotlib==3.1.1',
    'flake8==3.7.7',
    'mypy==0.670',
    'tqdm==4.38.0',
    'gpflow==2.0.0rc1',
    # for tests:
    'pytest-cov==2.7.1',
    'nbformat==4.4.0',
    'nbconvert==5.4.1',
    'jupyter_client==5.2.4',
    'ipykernel==5.1.0',
    'tornado==5.1.1',  # tornado 6 is broken
]

setup(name='gpflux',
      version="alpha",
      author="PROWLER.io",
      author_email="gpflux@prowler.io",
      description="GPFlux: Deep GP library",
      keywords="Deep-Gaussian-processes",
      install_requires=requirements,
      packages=['experiments', 'gpflux2'])
