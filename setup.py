from setuptools import setup
from sys import argv
from types import SimpleNamespace

with open("README.md", "r") as fh:
    long_description = fh.read()

PACKAGE = SimpleNamespace(
    **{
        "name": "COAsT",
        "version": "3.3.0",
        "description": "This is the Coast Ocean Assessment Tool",
        "long_description": long_description,
        "long_description_content_type": "text/markdown",
        "url": "https://www.bodc.ac.uk",
        "download_url": "https://github.com/British-Oceanographic-Data-Centre/COAsT/",
        "author": "British Oceanographic Data Centre (BODC)",
        "author_email": "bodcsoft@bodc.ac.uk",
        "license": "MIT License",
        "license_family": "OTHER",
        "setup_requires": ["wheel"],
        "classifiers": [
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Hydrology",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        "keywords": ["NEMO", "shallow water", "ocean assessment"],
        "project_urls": {"documentation": "https://british-oceanographic-data-centre.github.io/COAsT/"},
        "install_requires": [
            "PyYAML==6.0",
            "oyaml==1.0",
            "pytest==7.1.1",
            "pytest-mock==3.7.0",
            "numpy>=1.22.3",
            "dask>=2022.3.0",
            "dask[complete]>=2022.3.0",
            "xarray>=2022.3.0",
            "matplotlib>=3.5.3",
            "netCDF4>=1.5.8",
            "scipy>=1.8.0",
            "gsw>=3.6.17",
            "utide>=0.3.0",
            "scikit-learn>=1.0.2",
            "scikit-image>=0.19.2",
            "statsmodels>=0.13.2",
            "pydap>=3.2.2",
            "lxml>=4.9.0",  # Required for pydap CAS parsing,
            "requests>=2.27.1",
            "tqdm>=4.66.1",
            "pyproj>=3.5.0"
            # "xesmf>=0.3.0",  # Optional. Not part of main package
            # "esmpy>=8.0.0",  # Optional. Not part of main package
        ],
        "python_requires": ">=3.8,<3.11",
        "packages": ["coast", "coast.data", "coast._utils", "coast.diagnostics"],
        "include_package_data": True,
        "github": "British-Oceanographic-Data-Centre",
    }
)


def generate_conda(directory="conda"):
    import oyaml as yaml  # OYaml is used to preserve the order of the metadata dict in YAML output
    from collections import OrderedDict
    from os import path

    requirements = PACKAGE.install_requires + ["python", "pip"]

    package_metadata = OrderedDict(
        {
            "package": {"name": PACKAGE.name.lower(), "version": PACKAGE.version},
            "source": {
                "url": f"https://pypi.io/packages/source/{PACKAGE.name[0]}/{PACKAGE.name}/{PACKAGE.name}-{PACKAGE.version}.tar.gz"
            },
            "build": {"number": 0, "script": "{{ PYTHON }} -m pip install . --no-deps --ignore-installed -vv "},
            "requirements": {"host": requirements, "run": requirements},
            "test": {"imports": ["coast"]},
            "about": {
                "home": PACKAGE.url,
                "license": PACKAGE.license,
                "license_family": PACKAGE.license_family,
                "license_file": "",
                "summary": PACKAGE.description,
                "doc_url": "",
                "dev_url": "",
            },
            "extra": {"recipe-maintainers": [PACKAGE.github]},
        }
    )

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
            long_description=PACKAGE.long_description,
            long_description_content_type=PACKAGE.long_description_content_type,
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
            include_package_data=PACKAGE.include_package_data,
        )
