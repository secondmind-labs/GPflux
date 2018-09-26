#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'numpy',
    'tensorflow',
    'scipy',
    'sklearn'
    # don't import gpflow here as we need develop, add back in when we depend on the released version
]

setup(name='gpflux',
      version="alpha",
      author="Prowler.io",
      author_email="vincent@prowler.io",
      description=("GPFlux: Deep GP library"),
      keywords="Deep-Gaussian-processes",
      install_requires=requirements)
