#!/usr/bin/env bash
set -e

echo "Creating and entering temporary directory"
mkdir -p tmp
cd tmp
git clone https://github.com/bitcoin-core/secp256k1
cd secp256k1
git checkout ac83be33
./autogen.sh
./configure --enable-module-schnorrsig --enable-experimental
make
echo "Installing package, please enter sudo credentials if necessary"
sudo make install

echo "Exiting and deleting temporary directory"
cd ..
cd ..
rm -rf tmp
