#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

URL=${1:-"https://upload.pypi.org/legacy/"}
TOKEN=$2

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate coast
python -m twine upload --username __token__ --password $TOKEN --repository-url "$URL" ./dist/*
