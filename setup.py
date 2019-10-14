from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command import build_ext

cheby_ext = Extension('daepy.cheby.cheby', ['daepy/cheby/cheby.c'])

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="daepy",
    version="1.0",
    author="Alastair Flynn",
    author_email="contact@alastairflynn.com",
    description="A Python library for solving boundary value problems of differential algebraic equations with advanced and retarded (forward and backward) delays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://lcvmwww.epfl.ch/software/daepy",
    project_urls={'Documentation':'https://lcvmwww.epfl.ch/software/daepy',
                  'Source code':'https://github.com/alastairflynn/daepy'},
    packages=find_packages(),
    ext_modules=[cheby_ext],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.5',
    install_requires=['numpy>=1.4.0', 'scipy>=1.0.0', 'matplotlib>=3.0', 'dill'],
)
