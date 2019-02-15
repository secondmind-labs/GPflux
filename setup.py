#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'numpy==1.15.4',
    'tensorflow>=1.12.0',  # conda offers only version 1.12.0 (15.02.2019)
    'scipy==1.1.0',
    'scikit-learn==0.20.2',
    'flake8==3.6.0',
    'mypy==0.670',
    'tqdm==4.28.1',
    # for tests:
    'pytest-cov==2.5.1',
    'nbformat==4.4.0',
    'nbconvert==5.4.0',
    'jupyter_client==5.2.4',
    'ipykernel==5.1.0',
]

setup(name='gpflux',
      version="alpha",
      author="Prowler.io",
      author_email="vincent@prowler.io",
      description="GPFlux: Deep GP library",
      keywords="Deep-Gaussian-processes",
      install_requires=requirements,
      packages=['experiments', 'gpflux'])
