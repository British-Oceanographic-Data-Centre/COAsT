"""Script to turn google docstrings into markdown"""
from pathlib import Path
import subprocess
from datetime import date
import sys

# make sure we install the local whl for the docsting2md package
subprocess.check_call([sys.executable, "-m", "pip", "install", "docstring2md-0.4.1-py3-none-any.whl"])

# find all the python files within coast
root = Path(".")
coast_dir = root / "coast"
file_format = ".py"
file_paths = []
glob_prefix = "**/*"
file_paths.extend(coast_dir.glob(f"{glob_prefix}{file_format}"))

# Make sure we have a markdown folder to write to
md = root / "markdown"
md.mkdir(exist_ok=True)

# pre-formatted hugo markdown header where we can dynamical add dates and file names
markdown_header = """---
title: "{0}"
linkTitle: "{0}"
date: {1}
description: >
  Docstrings for the {0} class
---"""

day_now = date.today()
end = None  # variable holder for end of string

for python_file in file_paths:
    file_name_min_extension = python_file.name[0:-3]

    if file_name_min_extension == "__init__":
        # we might have more than one __init__.py file given the folder structure and these files shouldn't contain
        # methods, so we don't want to convert them
        continue

    process_call = f"export_docstring2md -i {str(python_file.absolute())}"
    # The first 33 char ain't needed and look mess within our site
    markdown_body = subprocess.run(process_call.split(), stdout=subprocess.PIPE).stdout.decode("utf-8")[33:end]

    # This adds the hugo required header to our markdown string.
    markdown_header_formatted = markdown_header.format(file_name_min_extension.capitalize(), day_now)
    with open(Path(f"markdown/{file_name_min_extension}.md"), "w") as md_file:
        md_file.write(markdown_header_formatted + markdown_body)
