from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "patch_inner_product",
        ["patch_inner_product.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    name='patch-inner-product',
    ext_modules=cythonize(ext_modules, annotate=True)
)

