#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

import uplc

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="uplc",
    version=uplc.__version__,
    description="Python implementation of untyped plutus language core",
    author=uplc.__author__,
    author_email=uplc.__author_email__,
    url=uplc.__url__,
    py_modules=["uplc"],
    packages=find_packages(),
    install_requires=[
        "frozendict==2.3.4",
        "cbor2==5.4.6",
        "frozenlist==1.3.3",
        "rply==0.7.8",
        "pyaiken==0.3.0",
        "pycardano==0.7.2",
    ],
    tests_require=[
        "hypothesis==6.62.0",
        "parameterized==0.8.1",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=uplc.__license__,
    classifiers=[
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Assemblers",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="python cardano smart contract blockchain verification haskell",
    python_requires=">=3",
    test_suite="uplc.tests",
    entry_points={
        "console_scripts": ["uplc=uplc.__main__:main"],
    },
)
