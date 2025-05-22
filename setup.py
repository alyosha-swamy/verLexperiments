from setuptools import setup, find_packages

setup(
    name="verLexperiments_repo",
    version="0.1.0",
    packages=find_packages(), # This will find verl_spectrum_patch
    description="Main package for verLexperiments to enable top-level sitecustomize via editable install.",
) 