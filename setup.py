#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'numpy==1.16.4',
    'tensorflow==1.14.0',
    'tensorflow-probability==0.7.0',
    'scipy==1.3.0',
    'scikit-learn==0.20.2',
    'matplotlib==3.1.0',
    'flake8==3.7.7',
    'mypy==0.670',
    'tqdm==4.28.1',
    'keras==2.2.4',
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
      author_email="vincent@prowler.io",
      description="GPFlux: Deep GP library",
      keywords="Deep-Gaussian-processes",
      install_requires=requirements,
      packages=['experiments', 'gpflux'])
