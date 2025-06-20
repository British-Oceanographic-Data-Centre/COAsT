#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

mkdir -p build dist
echo "Cleaning up old versions..."
rm -rf ./build/* ./dist/* ./Example_Python_Package.egg-info
echo "Building package for distribution..."
./venv/bin/python ./setup.py sdist bdist_wheel
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda create -n coast-toolbox python=3.10
conda activate coast-toolbox
conda install conda-forge::hdf5

echo "Done!"
