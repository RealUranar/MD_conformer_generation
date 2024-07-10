from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(Extension("rmsd", ["rmsd.pyx"]), annotate=True,),
    include_dirs=[numpy.get_include()]
)