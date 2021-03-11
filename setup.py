#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_namespace_packages, setup

requirements = [
    "gpflow>=2.1",
    "numpy",
    "scipy",
    "tensorflow-probability>=0.12.0",
    "tensorflow>=2.4.0",
]

setup(
    name="gpflux",
    version="0.1",
    author="Secondmind Labs",
    author_email="gpflux@secondmind.ai",
    description="GPflux: Deep GP library",
    keywords="Deep-Gaussian-processes",
    install_requires=requirements,
    packages=find_namespace_packages(include=["gpflux*"]),
    package_data={"gpflux": ["py.typed"]},
)
