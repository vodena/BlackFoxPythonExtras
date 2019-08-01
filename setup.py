# coding: utf-8

"""
    BlackFoxExtras

    BlackFoxExtras

"""


from setuptools import setup, find_packages  # noqa: H301

NAME = "blackfox_extras"
VERSION = "0.0.1"

REQUIRES = ["blackfox >= 0.0.8", "numpy >= 1.16.2", "scikit-learn >= 0.21.2"]

setup(
    name=NAME,
    version=VERSION,
    description="BlackFox Extras",
    author="Tomislav Mrdja",
    author_email="",
    url="https://github.com/vodena/BlackFoxPythonExtras",
    keywords=["BlackFox"],
    install_requires=REQUIRES,
    packages=find_packages(exclude=["test.*", "test"]),
    include_package_data=True,
    long_description="""\
        BlackFox Extras
    """
)
