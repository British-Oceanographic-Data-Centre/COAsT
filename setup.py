from setuptools import setup
from coast import PACKAGE


def get_package(package_path="package.json"):
    from types import SimpleNamespace
    import json

    with open(package_path, "r") as package_file:
        _package = json.load(package_file)
    return SimpleNamespace(**_package)


setup(
    name=PACKAGE.name,
    version=PACKAGE.version,
    description=PACKAGE.description,
    url=PACKAGE.url,
    download_url=PACKAGE.download_url,
    author=PACKAGE.author,
    author_email=PACKAGE.author_email,
    license=PACKAGE.license,  # TODO,
    setup_requires=PACKAGE.setup_requires,
    classifiers=PACKAGE.classifiers,
    keywords=PACKAGE.keywords,
    project_urls=PACKAGE.project_urls,
    install_requires=PACKAGE.install_requires,
    python_requires=PACKAGE.python_requires,
    packages=PACKAGE.packages,
    include_package_data=PACKAGE.include_package_data
)
