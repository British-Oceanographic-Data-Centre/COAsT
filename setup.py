from setuptools import setup
from sys import argv
from types import SimpleNamespace


PACKAGE = SimpleNamespace(**{
    "name": "COAsT",
    "version": "0.2.1a39",
    "description": "This is the Coast Ocean Assessment Tool",
    "url": "https://www.bodc.ac.uk",
    "download_url": "https://github.com/British-Oceanographic-Data-Centre/COAsT/",
    "author": "British Oceanographic Data Centre (BODC)",
    "author_email": "bodcsoft@bodc.ac.uk",
    "license": "Put something here",
    "license_family": "OTHER",
    "setup_requires": [
        "wheel"
    ],
    "classifiers": [
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
    ],
    "keywords": [
        "NEMO",
        "shallow water",
        "ocean assessment"
    ],
    "project_urls": {
        "documentation": "https://british-oceanographic-data-centre.github.io/COAsT/"
    },
    "install_requires": [
        "numpy>=1.16",
        "dask>=2",
        "dask[complete]>=2",
        "xarray>=0.1",
        "matplotlib>=3",
        "netCDF4>=1",
        "scipy>=1",
        "gsw>=3",
        "scikit-learn>=0.2",
        "scikit-image>=0.15"
    ],
    "python_requires": ">=3.7",
    "packages": [
        "coast"
    ],
    "include_package_data": True,
    "github": "British-Oceanographic-Data-Centre"
})


def generate_conda(directory="conda"):
    import oyaml as yaml  # OYaml is used to preserve the order of the metadata dict in YAML output
    from collections import OrderedDict
    from os import path

    requirements = PACKAGE.install_requires + ["python", "pip"]

    package_metadata = OrderedDict({
        "package": {
            "name": PACKAGE.name.lower(),
            "version": PACKAGE.version
        },
        "source": {
            "url": f"https://pypi.io/packages/source/{PACKAGE.name[0]}/{PACKAGE.name}/{PACKAGE.name}-{PACKAGE.version}.tar.gz"
        },
        "build": {
            "number": 0,
            "script": "{{ PYTHON }} -m pip install . --no-deps --ignore-installed -vv "
        },
        "requirements": {
            "host": requirements,
            "run": requirements
        },
        "test": {
            "imports": [
                "coast"
            ]
        },
        "about": {
            "home": PACKAGE.url,
            "license": PACKAGE.license,
            "license_family": PACKAGE.license_family,
            "license_file": "",
            "summary": PACKAGE.description,
            "doc_url": "",
            "dev_url": "",
        },
        "extra": {
            "recipe-maintainers": [
                PACKAGE.github
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
