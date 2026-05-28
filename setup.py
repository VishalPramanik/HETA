#!/usr/bin/env python3
# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="heta",
    version="1.0.0",
    author="Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha",
    author_email="vishalpramanik@ufl.edu",
    description=(
        "HETA: Hessian-Enhanced Token Attribution for Interpreting "
        "Autoregressive LLMs (ICLR 2026)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VishalPramanik/HETA",
    project_urls={
        "Paper": "https://arxiv.org/abs/2604.13258",
        "Bug Tracker": "https://github.com/VishalPramanik/HETA/issues",
    },
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "eval": [
            "datasets>=2.14.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="interpretability attribution hessian transformers llm xai",
)
