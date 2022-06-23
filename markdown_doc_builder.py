"""Script to turn google docstrings into markdown"""
from pathlib import Path
from datetime import date
from typing import Generator, List, Optional
from time import time
from docstring2md import DocString2MD

# Pre-formatted Hugo markdown header to which we can dynamically add dates and file names
MARKDOWN_HEADER = """---
title: "{0}"
linkTitle: "{0}"
date: {1}
description: >
  Docstrings for the {0} class
---"""


class MarkdownBuilder:
    def __init__(self, package: str, glob: str = "**/*.py"):
        self.package: str = package
        self.glob: str = glob

    @property
    def directory(self) -> Path:
        """Make sure we have a markdown folder to write to."""
        (directory := Path("markdown").resolve(strict=True)).mkdir(exist_ok=True)
        return directory

    @property
    def files(self) -> Generator[Path, None, None]:
        """Return all the Python modules within the target package."""
        return Path(self.package).resolve(strict=True).glob(self.glob)

    def generate_docs(self) -> List[Path]:
        """Generate docs for selected Python modules and output to Markdown files of the same name."""
        outputs = []
        for file in self.files:
            if (stem := file.stem) == "__init__":
                # We might have more than one __init__.py file depending on package structure and these files shouldn't
                # contain methods, so we don't want to convert them
                continue

            if not (doc := get_doc(file)):
                continue  # No docstring returned, skip this file
            doc = doc[33:]  # First 33 characters are not required for our docs

            # Write the output we've generated to a file
            (output := self.directory / f"{stem}.md").write_text(generate_header(stem) + doc)
            outputs.append(output)
        return outputs


def generate_header(name: str) -> str:
    """Generate the Hugo-required header for our Markdown string."""
    return MARKDOWN_HEADER.format(name.capitalize(), date.today())


def get_doc(file: Path) -> Optional[str]:
    module = DocString2MD(module_name=str(file.resolve(strict=True)))
    module.import_module()
    return module.get_doc()


def main() -> None:
    start = time()
    builder = MarkdownBuilder("coast")
    outputs = builder.generate_docs()
    print(f"Generated {len(outputs)} Markdown files in {round(time() - start, 2)} seconds!")


if __name__ == "__main__":
    main()
