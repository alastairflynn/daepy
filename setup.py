from setuptools import setup
from setuptools.extension import Extension
from setuptools.command import build_ext

cheby_ext = Extension('cheby', ['daepy/cheby.c'])

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="daepy",
    version="0.0.0",
    author="Alastair Flynn",
    author_email="alastair.flynn@epfl.ch",
    description="A collocation solver for boundary value problems of differential algebraic equations with forward and backward deviations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://lcvmwww.epfl.ch/software/daepy",
    project_urls={'Documentation':'https://lcvmwww.epfl.ch/software/daepy',
                  'Source code':'https://github.com/alastairflynn/daepy'},
    packages=setuptools.find_packages(),
    libraries=[cheby_ext],
    cmdclass={'build_ext':build_ext},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.5',
    install_requires=['numpy>=1.4.0', 'scipy>=1.0.0', 'dill'],
)
