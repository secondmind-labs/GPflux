#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_namespace_packages, setup

requirements = [
    "gpflow>=2.1",
    "numpy",
    "scipy",
    "tensorflow-probability>=0.11.0",
    "tensorflow>=2.3.0",
]

setup(
    name="gpflux",
    version="0.1",
    author="Secondmind Labs",
    author_email="gpflux@prowler.io",
    description="GPFlux: Deep GP library",
    keywords="Deep-Gaussian-processes",
    install_requires=requirements,
    packages=find_namespace_packages(include=["gpflux*"]),
)
