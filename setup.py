#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'numpy==1.15.1',
    'tensorflow==1.12.0',
    'scipy==0.19.0',
    'scikit-learn==0.20.0',
    'flake8==3.6.0',
    'mypy==0.670',
    'tqdm',
    # for tests:
    'pytest-cov==2.5.1',
    'nbformat',
    'nbconvert',
    'jupyter_client',
    'ipykernel',
]

setup(name='gpflux',
      version="alpha",
      author="Prowler.io",
      author_email="vincent@prowler.io",
      description="GPFlux: Deep GP library",
      keywords="Deep-Gaussian-processes",
      install_requires=requirements)
