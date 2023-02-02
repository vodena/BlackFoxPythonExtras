# coding: utf-8

"""
    BlackFoxExtras

    BlackFoxExtras

"""


from setuptools import setup, find_packages  # noqa: H301

NAME = "blackfox_extras"
VERSION = "1.0.0"

REQUIRES = [
    "blackfox >= 6.1.0",
    "numpy >= 1.19.1",
    "scikit-learn >= 1.0.0",
    "pandas >= 0.25.3",
    "category_encoders >= 2.2.2",
    "tensorflow >= 2.6.1",
    "xgboost == 1.0.2"
    ]

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
