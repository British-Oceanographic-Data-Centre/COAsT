from pathlib import Path
import subprocess
from datetime import date

root = Path(".")
coast_dir = root / "coast"
file_format = ".py"
file_paths = []
glob_prefix = "**/*"
file_paths.extend(coast_dir.glob(f"{glob_prefix}{file_format}"))
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
        continue

    process_call = f"export_docstring2md -i {str(python_file.absolute())}"
    markdown_body = subprocess.run(process_call.split(), stdout=subprocess.PIPE).stdout.decode("utf-8")[33:end]
    markdown_header_formatted = markdown_header.format(file_name_min_extension.capitalize(), day_now)
    with open(Path(f"markdown/{file_name_min_extension}.md"), 'w') as md_file:
        md_file.write(markdown_header_formatted + markdown_body)
