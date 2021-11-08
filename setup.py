#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from setuptools import find_namespace_packages, setup


requirements = [
    "deprecated",
    "gpflow>=2.1",
    "numpy",
    "scipy",
    "tensorflow>=2.5.0,<2.6.0",
    "tensorflow-probability>=0.12.0,<0.14.0",
]

with open("README.md", "r") as file:
    long_description = file.read()

with open(Path(__file__).parent / "gpflux" / "version.py", "r") as version_file:
    exec(version_file.read())

setup(
    name="gpflux",
    version=__version__,
    author="Secondmind Labs",
    author_email="gpflux@secondmind.ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="GPflux: Deep GP library",
    license="Apache License 2.0",
    keywords="Deep-Gaussian-processes",
    install_requires=requirements,
    packages=find_namespace_packages(include=["gpflux*"]),
    package_data={"gpflux": ["py.typed"]},
    project_urls={
        "Source on GitHub": "https://github.com/secondmind-labs/GPflux",
        "Documentation": "https://secondmind-labs.github.io/GPflux/",
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
