#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'numpy',
    'tensorflow',
    'gpflow',
]

setup(name='gpflux',
      version="alpha",
      author="Prowler.io",
      author_email="vincent@prowler.io",
      description=("GPFlux: Deep GP library"),
      keywords="Deep-Gaussian-processes",
      install_requires=requirements)
