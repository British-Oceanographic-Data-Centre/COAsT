from types import SimpleNamespace


PACKAGE = SimpleNamespace(**{
    "name": "COAsT",
    "version": "0.2.1a14",
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
        "dask[complete]",
        "xarray",
        "numpy",
        "matplotlib",
        "netCDF4"
    ],
    "python_requires": ">=3",
    "packages": [
        "coast"
    ],
    "include_package_data": True,
    "github": "British-Oceanographic-Data-Centre"
})
