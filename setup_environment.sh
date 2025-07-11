#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

echo "Cleaning up any existing virtual environment..."
rm -rf ./venv
echo "Installing required packages..."
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate coast
conda install -c conda-forge cython setuptools wheel twine

echo "Done!"