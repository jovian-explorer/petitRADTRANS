"""
Setup file for package `petitRADTRANS`.
"""
from setuptools import find_packages
from numpy.distutils.core import Extension, setup
import os

use_compiler_flags = True

if use_compiler_flags:
    extra_compile_args = [
        "-O3",
        "-funroll-loops",
        "-ftree-vectorize",
        "-msse",
        "-msse2",
        "-m3dnow"
    ]
    extra_compile_args_debug = [
        "-mcmodel=large",
        "-std=gnu",
        "-Wall",
        "-pedantic",
        "-fimplicit-none",
        "-fcheck-array-temporaries",
        "-fbacktrace",
        "-fcheck=all",
        "-ffpe-trap=zero,invalid",
        "-g",
        "-Og"
    ]
else:
    extra_compile_args = None


fort_spec = Extension(
    name='petitRADTRANS.fort_spec',
    sources=['petitRADTRANS/fort_spec.f90'],
    extra_compile_args=extra_compile_args
)

fort_input = Extension(
    name='petitRADTRANS.fort_input',
    sources=['petitRADTRANS/fort_input.f90'],
    extra_compile_args=extra_compile_args
)

fort_rebin = Extension(
    name='petitRADTRANS.fort_rebin',
    sources=['petitRADTRANS/fort_rebin.f90'],
    extra_compile_args=extra_compile_args
)
    
rebin_give_width = Extension(
    name='petitRADTRANS.retrieval.rebin_give_width',
    sources=['petitRADTRANS/retrieval/rebin_give_width.f90'],
    extra_compile_args=extra_compile_args
)

poor_mans = Extension(
    name='petitRADTRANS.poor_mans_nonequ_chem.chem_fortran_util.chem_fortran_util',
    sources=['petitRADTRANS/poor_mans_nonequ_chem/chem_fortran_util/chem_fortran_util.f90'],
    extra_compile_args=extra_compile_args
)

extensions = [fort_spec, fort_input, fort_rebin, rebin_give_width, poor_mans]


def setup_function():
    # from petitRADTRANS.version import version  # future

    setup(
        name='petitRADTRANS',
        version='2.4.4',
        description='Exoplanet spectral synthesis tool for retrievals',
        long_description=open(os.path.join(
          os.path.dirname(__file__), 'README.rst')).read(),
        # long_description_content_type='test/x-rst',
        url='https://gitlab.com/mauricemolli/petitRADTRANS',
        author='Paul Mollière',
        author_email='molliere@mpia.de',
        license='MIT License',
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'h5py',
            'corner',
            'astropy',
            'exo_k',
            'molmass',
            'seaborn',
            'pymultinest',
            'corner'
        ],
        zip_safe=False,
        ext_modules=extensions,
    )


setup_function()
