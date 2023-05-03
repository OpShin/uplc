Untyped Plutus Language Core 
==================================================
[![Build Status](https://app.travis-ci.com/OpShin/uplc.svg?branch=master)](https://app.travis-ci.com/OpShin/uplc)
 [![PyPI version](https://badge.fury.io/py/uplc.svg)](https://pypi.org/project/uplc/)
 ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/uplc.svg)
 [![PyPI - Status](https://img.shields.io/pypi/status/uplc.svg)](https://pypi.org/project/uplc/)
[![Coverage Status](https://coveralls.io/repos/github/OpShin/uplc/badge.svg?branch=master)](https://coveralls.io/github/OpShin/uplc?branch=master)

This is a basic library to support creating and manipulating programs written in [UPLC](https://blog.hachi.one/post/an-introduction-to-plutus-core/).

## Installation

Install anything between python 3.8 and 3.10.
Then run

```bash
pip install uplc
```

### Secp256k1

If you want to use the builtin functions for verification of ECDSA and Schnorr signatures,
follow the instructions to install `libsecp256k1` with schnorr support enabled:

https://github.com/input-output-hk/cardano-node/blob/master/doc/getting-started/install.md/#installing-secp256k1

This makes sure that the exact same version is used that is used in the `cardano-node`.

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

# This package can also be used to analyze built contracts (output from any Smart Contract Language)
uplc dump build/fibonacci/script.cbor --from-cbor

# Show all options
uplc --help
```


## Scope and Contributions
This is a side product of the development of a pythonic smart contract language for the Cardano blockchain
and hence much tailored to the needs of that development.

Contributions are very welcome.
