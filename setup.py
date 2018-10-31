#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'numpy==1.15.1',
    'tensorflow==1.12.0rc2',
    'scipy==0.19.0',
    'scikit-learn==0.20.0',
    'pytest-cov==2.5.1',
    'jupyter==1.0.0',
    'sacred==0.7.4 ',
    'pandas==0.23.4 '
]

setup(name='gpflux',
      version="alpha",
      author="Prowler.io",
      author_email="vincent@prowler.io",
      description="GPFlux: Deep GP library",
      keywords="Deep-Gaussian-processes",
      install_requires=requirements)
