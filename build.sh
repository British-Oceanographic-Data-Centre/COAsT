#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

mkdir -p build dist
echo "Cleaning up old versions..."
./venv/bin/pip install "cython<3.0.0"
./venv/bin/pip install --no-build-isolation PyYAML==6.0
rm -rf ./build/* ./dist/* ./Example_Python_Package.egg-info
echo "Building package for distribution..."
./venv/bin/python ./setup.py sdist bdist_wheel

echo "Done!"
