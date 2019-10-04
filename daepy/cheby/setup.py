from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

cheby_ext = Extension('cheby', ['cheby.pyx'])
setup(name='cheby', ext_modules=cythonize(cheby_ext))
