from setuptools import find_packages, setup

setup(
    name="setup",
    version="1.0.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
