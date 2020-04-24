#!/bin/bash

set -e  # Exit on error
cd "$(dirname "$0")"  # Set working directory

URL=${1:-"https://upload.pypi.org/legacy/"}

./venv/bin/python -m twine upload --repository-url "$URL" ./dist/*
