from setuptools import setup, find_packages


setup(
    name="Coastal Ocean Assessment Tool",
    version="0.0.1.dev1",
    description="Put something here",  # TODO
    url="https://www.bodc.ac.uk",  # TODO
    author="British Oceanographic Data Centre (BODC)",
    license="Put something here",  # TODO,

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="Put something here",  # TODO
    project_urls="Put something here",  # TODO
    install_requires=[],  # TODO
    python_requires=">=3",
    packages=find_packages("coast"),
    include_package_data=True
)
