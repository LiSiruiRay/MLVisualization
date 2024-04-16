# Author: ray
# Date: 4/16/24
# Description:

from setuptools import setup, find_packages

setup(
    name="FEDformerInstallable",  # Replace with your package name
    version="0.1.0",
    author="Ray",  # Replace with your name or your organization's name
    description="A Python package for FEDformer incorporating time series forecasting models",
    long_description=open('README.md').read(),  # Ensure you have a README.md in your package directory
    long_description_content_type="text/markdown",
    url="https://github.com/LiSiruiRay/FEDformer",  # Replace with the URL to your fork of the repository
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').read().splitlines(),  # Reads requirements from requirements.txt
    python_requires='>=3.6',
)
