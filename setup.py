"""Install the advds package for CSC 5300 Week 13."""

from setuptools import find_packages, setup

setup(
    name="advds",
    version="1.0.0",
    description="Segment trees, Fenwick trees, persistent trees, rank/select, wavelet trees, and cache-oblivious layouts.",
    author="Omar Vasquez",
    python_requires=">=3.9",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest>=7"],
    },
)
