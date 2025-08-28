from setuptools import setup, find_packages

setup(
    name="hypnose",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "ipympl",
        "harp-python",
        "swc-aeon",
    ],
)