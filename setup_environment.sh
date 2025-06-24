#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

echo "Cleaning up any existing virtual environment..."
rm -rf ./venv
echo "Installing required packages..."
conda activate coast
conda install conda-forge::setuptools conda-forge::wheel conda-forge::twine

echo "Done!"