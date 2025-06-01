#### Test building cython codes


from setuptools import setup
import cython
from cython.build import cythonize

print(cython.__version__)

setup(ext_modules = cythonize("vector.pyx"))