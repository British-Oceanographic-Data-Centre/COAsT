#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

mkdir -p build dist
echo "Cleaning up old versions..."
rm -rf ./build/* ./dist/* ./Example_Python_Package.egg-info
echo "Building package for distribution..."
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate coast
python ./setup.py sdist bdist_wheel

echo "Done!"