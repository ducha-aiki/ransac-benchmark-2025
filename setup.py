import os
from setuptools import setup, find_packages

setup(
    name="ransac_benchmark_imc",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    author="Dmytro Mishkin",
    author_email="ducha.aiki@gmail.com",
    description="A Python package for RANSAC benchmarking",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

