#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

echo "Cleaning up any existing virtual environment..."
rm -rf ./venv
echo "Creating new virtual environment..."
python3 -m virtualenv ./venv
echo "Installing required packages..."
./venv/bin/pip install setuptools wheel twine

echo "Done!"
