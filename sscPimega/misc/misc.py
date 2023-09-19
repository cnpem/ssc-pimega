import os
import sys
import ctypes
import numpy
from ..pimegatypes import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

import multiprocessing
import multiprocessing.sharedctypes

import functools
from functools import partial

import gc
import subprocess
import SharedArray as sa
import uuid
import h5py


def imshow(img, size, savename=None, show=False):

    """ Function to display a given image, segmenting colours wrt to a log-scale.
    
    Args:
        img: (ndarray) image 
        size: tuple indicating the aspect ratio
        savename: path to save the image (optional)
        show: plt.show() (optional)  

    Returns:
        bounds: (list) array with the segmented intervals 
    
    """
    
    colors = [ 'white', '#FFC0CB', '#0000FF' , '#00FFFF', 'green', 'gold', 'orange', 'red', '#C20078', 'maroon', 'black' ]
    
    cmap = mpl.colors.ListedColormap(colors)
    
    maxv    = img.max()
    epsilon = -0.1 
    
    bounds =  numpy.zeros([12,])
    bounds[0] = -10
    bounds[1] = epsilon
    bounds[2] = 0
    for k in range(3,11):
        bounds[k] = 10**( (k-2) * numpy.log(maxv) / (9 * numpy.log(10) ))
        bounds[11] = maxv
        
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=size)
    handle = plt.imshow(img, interpolation='nearest', cmap = cmap, norm=norm)
    plt.colorbar(handle, boundaries=bounds)

    plt.axis('off')
    
    if savename != None:
        plt.savefig(savename)

    if show:
        plt.show()
    
    return bounds

def show():
    plt.show()


def rings(N=1536, cx=512, cy=512, n=6):

    """ Function to generate an image with rings, suitable for simulation. 
    
    Args:
        N: image size
        cx: x-center coordinate (pixels)
        cy: y-center coordinate (pixels)
        n: number of rings

    Returns:
        (dict): (ndarray) image
    
    """
    
    t  =numpy.linspace(-1,1,N);
    x,y=numpy.meshgrid(t,t);

    r = numpy.linspace(0.02, 1.5, n)

    e = (r[1] - r[0])/20.  #ring width

    ring = lambda rad: (( ( x-t[cx])**2 + (y-t[cy])**2 < (rad+e)**2) & ((x-t[cx])**2 + (y-t[cy])**2 > (rad-e)**2) ).astype(numpy.double)

    img = ring(r[0])

    for i in range(1,len(r)):
        img += ring(r[i])

    return img



######################################################
#
#
#
# RESTORE IMAGE PACKAGES
#
# (with multiprocessing + SharedArray)
#
#
#####################################################

def _get_size_from_shape(shape):
    return functools.reduce(lambda x, y: x * y, shape)
 
def _create_np_shared_array(shape, dtype, ctype):
    # Feel free to create a map from usual dtypes to ctypes. Or suggest a more elegant way
    size = _get_size_from_shape(shape)
    shared_mem_chunck = multiprocessing.sharedctypes.RawArray(ctype, size)
    numpy_array_view = numpy.frombuffer(shared_mem_chunck, dtype).reshape(shape)
    return numpy_array_view

def _worker_batch_(params, idx_start,idx_end, elapsed):
    
    V        = params[0]
    measure  = params[2]
    output   = params[3]
    shape    = params[4]
    myfun    = params[5]
    args     = params[6]
    
    for k in range(idx_start, idx_end):
        
        start = time.time()

        image = myfun( measure[k,:,:], args)        

        elapsed0 = time.time() - start
        
        elapsed[k, 0] = elapsed0
        
        output[k, :, :] = image 
        

def _batch_(params):

    V = params[0]
    t = params[1]
    b = int( numpy.ceil(V/t) ) 

    elapsed = _create_np_shared_array([V,3], numpy.float32, ctypes.c_float)
    
    processes = []
    for k in range(t):
        begin_ = k*b
        end_   = min( (k+1)*b, V)
        p = multiprocessing.Process(target=_worker_batch_, args=(params, begin_, end_, elapsed))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
    return elapsed


####

def batch( measure, threads, oshape, myfun, args ):

    """ Function to parallelize (with multiprocessing) the application of a given function to a sequence of images
     
    Args:
        measure: digital NxSxS measured volume.
        threads: number of processes
        shape: output shape for the image return at <myfun>
        myfun: function to be applied in batch
        args: extra arguments for <myfun>

    Returns:
        (ndarray): restored volume
    
    """
    
    npoints = measure.shape[0]
    
    start = time.time()

    name = str( uuid.uuid4() )
    
    try:
        sa.delete(name)
    except:
        print('ssc-pimega: creating {}x{}x{} shared array (SharedArray)'.format(npoints, oshape[0], oshape[1]) )
        
    output = sa.create(name, [npoints, oshape[0], oshape[1]], dtype=numpy.float32)

    satime = time.time() - start
    
    start = time.time()
    
    params = (npoints, threads, measure, output, oshape, myfun, args)
    
    etimes = _batch_(params)
    
    elapsed = time.time() - start

    sa.delete(name)

    print('ssc-pimega: {} Images corrected within {} sec'.format(npoints, elapsed))
    print('               Shared Array creation {} sec'.format(satime))
    
    return output, etimes




