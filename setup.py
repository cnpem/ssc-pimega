#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join as pjoin
import warnings
import glob
from setuptools import setup, Extension, find_packages
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import sys
import shutil

##########################################################

# Set Python package requirements for installation.   

compile_cuda = 0

if '--cuda' in sys.argv:

    compile_cuda = 1
    sys.argv.remove('--cuda')

    setup_requires = [
        'pkgconfig',
    ]

install_requires = [
    'numpy>=1.7.0',
    'scipy>=0.12.0',
    'matplotlib',
    'h5py',
    'scikit-image',
    'zipp',
    'six',
]


def locate_cuda():
    """Locate the CUDA environment on the system
        
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
        
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """
        
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            print ('The nvcc binary could not be '
                   'located in your $PATH. Either add it to your path, or set $CUDAHOME')
            return None
        home = os.path.dirname(os.path.dirname(nvcc))
   
    check = pjoin(home, 'lib')

    if not os.path.exists(check):
        cudaconfig = {'home':home, 'nvcc':nvcc,
        	      'include': pjoin(home, 'include'),
                      'lib': pjoin(home, 'lib64')}
    else:
        cudaconfig = {'home':home, 'nvcc':nvcc,
        	      'include': pjoin(home, 'include'),
                      'lib': pjoin(home, 'lib')}
    return cudaconfig


def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works. """
   
    CUDA = locate_cuda()

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')
    
    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


def customize_compiler(self):
    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


def CudaCompile():

    CUDA = locate_cuda()
    
    # ------------------
    # Compile CUDA codes
        
    pwd = os.getcwd()

    ssc_pipeline_codes   = set( glob.glob('cuda/ssc-pipeline/src/codes/*.cu') )
    ssc_pipeline_include = pwd + '/cuda/ssc-pipeline/src/'
     
    pimega = set( glob.glob('cuda/src/*.cu') )
    pimega_include = pwd + '/cuda/'
    
    ssc_pimega_codes = list(ssc_pipeline_codes) + list(pimega)
 
    mpi_cflags = pkgconfig.cflags("mpi")
    mpi_libs = pkgconfig.libs("mpi")
    #print(mpi_cflags)
    #print(mpi_libs)
    #sys.exit()
    
    ssc_pimega_includes = [ CUDA['include'], pimega_include, ssc_pipeline_include, pwd +  '/cuda/common/common10/' ]

    ext_pimega = Extension(name                 = 'sscPimega.lib.libssc_pimega',
                           sources              = ssc_pimega_codes,
                           library_dirs         = [ CUDA['lib'] ],
                           runtime_library_dirs = [ CUDA['lib'] ],
                           extra_compile_args   = {'nvcc': ['-Xcompiler',
                                                            #mpi_cflags,
                                                            '-use_fast_math',
                                                            '--ptxas-options=-v',
                                                            '-c',
                                                            '--compiler-options',
                                                            '-fPIC'
                                                            ]},
                           extra_link_args      = [ #mpi_libs,
                                                    '-std=c++14',
                                                   '-lm',
                                                   '-lcudart',
                                                   '-lpthread',
                                                   '-lcufft',
                                                   '-lcublas',
                                                   '-lhdf5'
                                                   ],
                                                   
                           include_dirs =  ssc_pimega_includes )

    return ext_pimega


###########################
# Main setup configuration.


if compile_cuda:

    import pkgconfig

    ext_pimega = CudaCompile()

    # run the customize_compiler
    class custom_build_ext(build_ext):
        def build_extensions(self):
            customize_compiler_for_nvcc(self.compiler)
            build_ext.build_extensions(self)
                
    setup(
        name='sscPimega',
        version = open('VERSION').read().strip(),
        
        packages = find_packages(),
        include_package_data = True,
        
        ext_modules=[ ext_pimega ],
        cmdclass={'build_ext': custom_build_ext},
        
        # since the package has c code, the egg cannot be zipped
        zip_safe=False,    
        
        author='Eduardo X. Miqueles',
        author_email='eduardo.miqueles@lnls.br',
        
        description='PIMEGA Routines',
        keywords=['detector', 'restoration', 'imaging'],
        url='http://www.',
        download_url='',
        
        license='LGPL',
        platforms='Any',
        install_requires = install_requires,
        
        classifiers=['Development Status :: 4 - Beta',
                     'License :: LPGL License',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: Education',
                     'Intended Audience :: Developers',
                     'Natural Language :: English',
                     'Operating System :: OS Independent',
                     'Programming Language :: Python',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3.0',
                     'Programming Language :: C',
                     'Programming Language :: C++']       
    )

else:

    # run the customize_compiler
    class custom_build_ext2(build_ext):
        def build_extensions(self):
            customize_compiler(self.compiler)
            build_ext.build_extensions(self)

    setup(
        name='sscPimega',
        version = open('VERSION').read().strip(),
        
        packages = find_packages(),
        include_package_data = True,
        
        cmdclass={'build_ext': custom_build_ext2},
        
        # since the package has c code, the egg cannot be zipped
        zip_safe=False,    
        
        author='Eduardo X. Miqueles',
        author_email='eduardo.miqueles@lnls.br',
        
        description='PIMEGA Routines',
        keywords=['detector', 'restoration', 'imaging'],
        url='http://www.',
        download_url='',
        
        license='LGPL',
        platforms='Any',
        install_requires = install_requires,
        
        classifiers=['Development Status :: 4 - Beta',
                     'License :: LPGL License',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: Education',
                     'Intended Audience :: Developers',
                     'Natural Language :: English',
                     'Operating System :: OS Independent',
                     'Programming Language :: Python',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3.0',
                     'Programming Language :: C',
                     'Programming Language :: C++']
        
    )

    



