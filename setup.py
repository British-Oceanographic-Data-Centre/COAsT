from setuptools import setup
from metadata import get_package

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
