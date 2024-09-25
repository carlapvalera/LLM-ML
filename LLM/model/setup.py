# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("context_manager.pyx"),
    zip_safe=False,
)