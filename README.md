Untyped Plutus Language Core 
==================================================
[![Build Status](https://app.travis-ci.com/ImperatorLang/uplc.svg?branch=master)](https://app.travis-ci.com/ImperatorLang/uplc)
 [![PyPI version](https://badge.fury.io/py/uplc.svg)](https://pypi.org/project/uplc/)
 ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/uplc.svg)
 [![PyPI - Status](https://img.shields.io/pypi/status/uplc.svg)](https://pypi.org/project/uplc/)
[![Coverage Status](https://coveralls.io/repos/github/ImperatorLang/uplc/badge.svg?branch=master)](https://coveralls.io/github/ImperatorLang/uplc?branch=master)

This is a basic library to support creating and manipulating programs written in [UPLC](https://blog.hachi.one/post/an-introduction-to-plutus-core/).

## Installation

Install anything between python 3.8 and 3.10.
Then run

```bash
pip install uplc
```

## Usage

This tool may be used to parse, reformat (/dump), evaluate or build contract artifacts from UPLC code.

```bash
# Check validity of a source file
uplc parse examples/fibonacci.uplc

# Dump a source file in either the aiken or the plutus dialect
uplc dump examples/fibonacci.uplc --dialect aiken
uplc dump examples/fibonacci.uplc --dialect plutus --unique-varnames

# Evaluate a UPLC program on UPLC input
uplc eval examples/fibonacci.uplc "(con integer 5)"

# Build smart contract artifacts from the UPLC program
uplc build examples/fibonacci.uplc
```


## Scope and Contributions
This is a side product of the development of a pythonic smart contract language for the Cardano blockchain
and hence much tailored to the needs of that development.

Contributions are very welcome.
