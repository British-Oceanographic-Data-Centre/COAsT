#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

echo "Cleaning up any existing virtual environment..."
rm -rf ./venv
echo "Creating new virtual environment..."
python3 -m virtualenv ./venv
echo "Installing required packages..."
./venv/bin/pip install setuptools wheel twine
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda config --set verbosity 3 —env
conda env update --prune --file environment.yml 
conda activate coast

echo "Done!"
