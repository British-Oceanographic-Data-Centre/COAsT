from distutils.core import setup

setup(
    name="COAsT",
    version="0.1.1",
    description="This is the Coastal Ocean Assessment Tool",
    url="https://www.bodc.ac.uk",
    download_url = 'https://github.com/British-Oceanographic-Data-Centre/COAsT/archive/0.1.1.tar.gz',
    author="British Oceanographic Data Centre (BODC)",
    author_email="bodcsoft@bodc.ac.uk",
    license="Put something here",  # TODO,

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    keywords=["NEMO", "shallow water", "ocean assessment"],
    project_urls={"documentation":"https://british-oceanographic-data-centre.github.io/COAsT/"},
    install_requires=[
        'dask',
        'xarray',
    ],
    python_requires=">=3",
    packages=["COAsT"],
    include_package_data=True
)
