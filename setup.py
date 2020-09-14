from setuptools import setup
from sys import argv


def get_package(package_path="package.json"):
    from types import SimpleNamespace
    import json

    with open(package_path, "r") as package_file:
        pack = json.load(package_file)
    return SimpleNamespace(**pack)


def generate_conda(directory="conda"):
    import oyaml as yaml  # OYaml is used to preserve the order of the metadata dict in YAML output
    from collections import OrderedDict
    from os import path

    pack = get_package()
    package_metadata = OrderedDict({
        "package": {
            "name": pack.name.lower(),
            "version": pack.version
        },
        "source": {
            "url": f"https://pypi.io/packages/source/{pack.name[0]}/{pack.name}/{pack.name}-{pack.version}.tar.gz"
        },
        "requirements": {
            "host": pack.install_requires,
            "run": pack.install_requires
        },
        "test": {
            "imports": [
                "coast"
            ]
        },
        "about": {
            "home": pack.url,
            "license": pack.license,
            "license_family": pack.license_family,
            "license_file": "",
            "summary": pack.description,
            "doc_url": "",
            "dev_url": "",
        },
        "extra": {
            "recipe-maintainers": [
                pack.github
            ]
        }
    })

    yaml_path = path.join(directory, "meta.yaml")
    yaml.Dumper.ignore_aliases = lambda *args: True  # Dummy function to return true
    with open(yaml_path, "w") as meta_file:
        yaml.dump(package_metadata, meta_file)


if __name__ == "__main__":
    if argv[1] == "conda":
        generate_conda()
    else:
        package = get_package()
        setup(
            name=package.name,
            version=package.version,
            description=package.description,
            url=package.url,
            download_url=package.download_url,
            author=package.author,
            author_email=package.author_email,
            license=package.license,  # TODO,
            setup_requires=package.setup_requires,
            classifiers=package.classifiers,
            keywords=package.keywords,
            project_urls=package.project_urls,
            install_requires=package.install_requires,
            python_requires=package.python_requires,
            packages=package.packages,
            include_package_data=package.include_package_data
        )
