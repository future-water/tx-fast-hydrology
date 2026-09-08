#!/usr/bin/env python

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).parent

setup(
    name="tx-fast-hydrology",
    version="0.1",
    description="Routing and data assimilation code for TxDOT FAST project.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Matt Bartos",
    author_email="mdbartos@utexas.edu",
    url="https://future-water.org",
    packages=["tx_fast_hydrology"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    include_package_data=True,
    python_requires=">=3.11,<3.13",
    install_requires=[
        "numpy>=1.22,<1.27",
        "pandas>=2.0,<3",
        "numba>=0.59,<0.60",
        "scipy>=1.10,<2",
        "xarray>=2024.1,<2025",
        "h5netcdf>=1.3,<2",
        "h5py>=3.10,<4",
    ],
    extras_require=dict(
        dev=["pytest", "pytest-cov"]
    ),
)
