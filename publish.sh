#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

URL=${1:-"https://upload.pypi.org/legacy/"}
TOKEN=$2

./venv/bin/python -m twine upload --username __token__ --password $TOKEN --repository-url "$URL" ./dist/*
