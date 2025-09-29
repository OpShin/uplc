Untyped Plutus Language Core 
==================================================
[![CI/CD](https://github.com/OpShin/uplc/actions/workflows/build.yml/badge.svg)](https://github.com/OpShin/uplc/actions/workflows/build.yml)
 [![PyPI version](https://badge.fury.io/py/uplc.svg)](https://pypi.org/project/uplc/)
 ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/uplc.svg)
 [![PyPI - Status](https://img.shields.io/pypi/status/uplc.svg)](https://pypi.org/project/uplc/)
[![Coverage Status](https://coveralls.io/repos/github/OpShin/uplc/badge.svg?branch=master)](https://coveralls.io/github/OpShin/uplc?branch=master)

This is a basic library to support creating and manipulating programs written in [UPLC](https://blog.hachi.one/post/an-introduction-to-plutus-core/).

## Installation

Install Python 3.9 or higher and make sure you have `pip` installed.
Then run

```bash
pip install uplc
```

### Secp256k1

If you want to use the builtin functions for verification of ECDSA and Schnorr signatures,
follow the instructions to install `libsecp256k1` with schnorr support enabled. We provide a script to do this for you on Linux and MacOS.

```bash
curl -sSL https://raw.githubusercontent.com/OpShin/uplc/refs/heads/master/install_secp256k1.sh | bash
```


This makes sure that the exact same version is used that is used in the `cardano-node`.
Then, install the python bindings for `secp256k1` with

```bash
pip install python-secp256k1-cardano
```

## Usage

This tool may be used to parse, reformat (/dump), evaluate or build contract artifacts from UPLC code.

```bash
# Check validity of a source file
uplc parse examples/fibonacci.uplc

# Dump a source file in either the official plutus or legacy aiken dialect
uplc dump examples/fibonacci.uplc --dialect plutus --unique-varnames
uplc dump examples/fibonacci.uplc --dialect legacy-aiken

# Evaluate a UPLC program on UPLC input
uplc eval examples/fibonacci.uplc "(con integer 5)"

# Build smart contract artifacts from the UPLC program
uplc build examples/fibonacci.uplc

# This package can also be used to analyze built contracts (output from any Smart Contract Language)
uplc dump build/fibonacci/script.cbor --from-cbor

# You can also apply additional parameters to a script using the build command
uplc build script.cbor --from-cbor "(con integer 5)"

# Show all options
uplc --help
```

## Running tests

To run the testsuite of UPLC, use `pytest`.
```
pytest
```


## Scope and Contributions
This is a side product of the development of a pythonic smart contract language for the Cardano blockchain
and hence much tailored to the needs of that development.

Most likely it *can* do what you would like to do but its not properly documented. Please do reach out via Discord or GitHub issue if you think this tool could be of use to you.

Contributions are very welcome.
