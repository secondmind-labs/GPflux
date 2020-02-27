#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'numpy',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'tensorflow>=2.0.0',
    'tensorflow-probability>=0.8.0',
    'gpflow>=2.0.0rc1',
]

setup(name='gpflux',
      version='0.0.1',
      author="PROWLER.io",
      author_email="gpflux@prowler.io",
      description="GPFlux: Deep GP library",
      keywords="Deep-Gaussian-processes",
      install_requires=requirements,
      packages=['experiments', 'gpflux'])
