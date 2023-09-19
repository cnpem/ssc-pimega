#We use the one encoding: utf8
import ctypes
from ctypes import *
import ctypes.util
import multiprocessing
import math
import os
import sys
import numpy
import time
import glob


nthreads = multiprocessing.cpu_count()

libstdcpp = ctypes.CDLL( ctypes.util.find_library( "stdc++" ), mode=ctypes.RTLD_GLOBAL )
#libcufft  = ctypes.CDLL( ctypes.util.find_library( "cufft" ), mode=ctypes.RTLD_GLOBAL )
#libcublas  = ctypes.CDLL( ctypes.util.find_library( "cublas" ), mode=ctypes.RTLD_GLOBAL )

#################
#|  ssc-pimega |#
#|  import     |#
#################

_lib = "lib/libssc_pimega"

def load_library(lib,ext):

    try:
        _path = glob.glob( os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib + "*.so")    
        if isinstance(_path, list):
            _path = _path[0]

        library = ctypes.CDLL(_path)

    except:
        library = None
    
    return library

libssc_pimega = load_library( _lib, '.so' )

def getPointer(darray,dtype=numpy.float32):
    if darray.dtype != dtype:
        return numpy.ascontiguousarray(darray.astype(dtype))
    elif darray.flags['C_CONTIGUOUS'] == False:
        return numpy.ascontiguousarray(darray)
    else:
        return darray

#########################
#|   ssc-pimega        |#
#| Function prototypes |#
#########################

try:

    libssc_pimega.ssc_pimega_pi540D_backward_pipeline.argtypes =  [ ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.c_char_p,
                                                                    ctypes.c_char_p,
                                                                    ctypes.c_char_p,
                                                                    ctypes.c_char_p,
                                                                    ctypes.c_char_p,
                                                                    ctypes.c_int,
                                                                    ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.c_int,
                                                                    ctypes.c_int,
                                                                    ctypes.c_int,
                                                                    ctypes.c_int,
                                                                    ctypes.c_int,
                                                                    ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.POINTER(ctypes.c_int),
                                                                    ctypes.c_int,
                                                                    ctypes.POINTER(ctypes.c_float),
                                                                    ctypes.POINTER(ctypes.c_float),
                                                                    ctypes.POINTER(ctypes.c_float),
                                                                    ctypes.POINTER(ctypes.c_float),
                                                                    ctypes.c_float,
                                                                    ctypes.c_int,
                                                                    ctypes.c_char_p,
                                                                    ctypes.POINTER(ctypes.c_float),
                                                                    ctypes.c_int]
    
    libssc_pimega.ssc_pimega_pi540D_backward_pipeline.restype  = ctypes.c_int
    
except:
    #print('ssc-pimega error: no CUDA functions available!')
    pass



##############
#|          |#
##############

if __name__ == "__main__":
   pass

