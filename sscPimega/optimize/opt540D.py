import os
import sys
import ctypes
import numpy
import time
import gc
import h5py

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob

import multiprocessing
import multiprocessing.sharedctypes

from threading import Thread

import functools
from functools import partial

from scipy import ndimage
from PIL import Image
import gc
import subprocess
import SharedArray as sa
import uuid

#from ..pi135D import *
from ..pi540D import *
from ..pimegatypes import *

from skimage.measure import EllipseModel, LineModelND
from scipy.optimize import minimize

from scipy import ndimage
from skimage.feature import canny
from skimage.morphology import convex_hull_image
from scipy.ndimage import gaussian_filter



import pickle

EUCLIDEAN  = 2
HORIZONTAL = 1
VERTICAL   = 0


def MACRO_ANNOTATION( s ):
    if s=='e':
        return EUCLIDEAN
    elif s=='x':
        return HORIZONTAL
    elif s=='y':
        return VERTICAL
    else:
        return -1


####################################################
#
#
# GLOBAL OPTIMIZATION SCHEME FOR ALIGNMENT
#
#
###################################################

def optimize_540D ( piannofiles, dettype ):

    if dettype == 'planar':
                    
        x, params, annotation = optimize_540D_planar ( piannofiles )
        
    elif dettype == 'nonplanar':
        
        x, params, annotation = optimize_540D_nonplanar ( piannofiles )

    else:
        print('ssc-pimega: Error! Wrong pimega detector type!')
        return False
        
    return x, params, annotation


def see_annotations_540D( img, piannofiles, dettype ):

    if dettype == 'planar':
                    
        imga, imgo = see_annotations_540D_planar ( img, piannofiles )
        
    elif dettype == 'nonplanar':
        
        imga, imgo = see_annotations_540D_nonplanar( img, piannofiles )
        
    else:
        print('ssc-pimega: Error! Wrong pimega detector type!')
        return False
        
    return imga, imgo


def get_project_bounds_geometry( var, *args ):

    def get_standard_tolerance():
        
        class outopt:
            a      = []
            rx     = []
            ry     = []
            rz     = []
            offset = []
            ox     = []
            oy     = []
            normal = []
            center = []
            z      = []

        out = outopt()
        
        out.a  = [1,1]
        out.rx = [0.5, 0.5]      
        out.ry = [0.5, 0.5] 
        out.rz = [0.5, 0.5]   
        out.offset = [50, 50]
        out.ox = [ 2 * MEDIPIX, 2 * MEDIPIX ]
        out.oy = [ 3 * MEDIPIX, 3 * MEDIPIX ]
        out.normal = None
        out.center = None  
        out.z      = [0.5 , 1]

        return out

    #
    
    if not args:
        x = get_project_values_geometry("project") 
        tolerance = get_standard_tolerance()
    else:
        
        if len(args[0]) > 1:
            
            params = args[0]

            x   = params[0]
            dic = params[1]
            
            tolerance = get_standard_tolerance()

            tolerance.a      = [ dic['a'][0] if 'a' in dic else 1,  dic['a'][1] if 'a' in dic else 1] 
            tolerance.rx     = [ dic['rx'][0] if 'rx' in dic else 0.5,  dic['rx'][1] if 'rx' in dic else 0.5]
            tolerance.ry     = [ dic['ry'][0] if 'ry' in dic else 0.5,  dic['ry'][1] if 'ry' in dic else 0.5]
            tolerance.rz     = [ dic['rz'][0] if 'rz' in dic else 0.5,  dic['rz'][1] if 'rz' in dic else 0.5]
            tolerance.offset = [ dic['offset'][0] if 'offset' in dic else 50,  dic['offset'][1] if 'offset' in dic else 50]
            tolerance.ox     = [ dic['ox'][0] if 'ox' in dic else 2*MEDIPIX,  dic['ox'][1] if 'ox' in dic else 2*MEDIPIX]
            tolerance.oy     = [ dic['oy'][0] if 'oy' in dic else 3*MEDIPIX,  dic['oy'][1] if 'oy' in dic else 3*MEDIPIX]
            tolerance.normal = None
            tolerance.center = None  
            tolerance.z      = [ dic['z'][0] if 'z' in dic else 0.5,  dic['z'][1] if 'z' in dic else 1]

        else:
            params    = args[0]
            x         = params[0]
            tolerance = get_standard_tolerance()
    
            
    a = x[0:4]
    r = x[4:76]
    L = x[76:100]
    o = x[100:148]
    v = x[148:151]
    center = x[151:153] 
    z = x[153]
    
    #
    
    bnds = []
    #a
    for k in range(4):
        bnds.append( ( a[k] - tolerance.a[0], a[k] + tolerance.a[1] ) )

    #rx
    for k in range(24):
        bnds.append( ( r[k] - tolerance.rx[0] , r[k] + tolerance.rx[1] ) )

    #ry
    for k in range(24, 24*2):
        bnds.append( ( r[k] - tolerance.ry[0], r[k] + tolerance.ry[1] ) )

    #rz
    for k in range(24*2, 24*3):
        bnds.append( ( r[k] - tolerance.rz[0], r[k] + tolerance.rz[1] ) )
        
    #L
    for k in range(24):
        bnds.append( ( L[k] - tolerance.offset[0], L[k] + tolerance.offset[1] ) )

    #ox
    for k in range(24):
        bnds.append( ( o[k] - tolerance.ox[0], o[k] + tolerance.ox[1] ) ) 
    
    #oy
    for k in range(24, 24*2):
        bnds.append( ( o[k] - tolerance.oy[0], o[k] + tolerance.oy[1] ) )
    
    #v
    for k in range(3):
        bnds.append( (None, None) )

    #center
    bnds.append( (None,None) )
    bnds.append( (None,None) )

    #zoom
    bnds.append((z - tolerance.z[0], z + tolerance.z[1] ))
    
    out = []
    for k in range(len(var)):
        if var[k]>0:
            out.append(  bnds[k]  )
        
    return out


def optimization_variables( variables ):
 
    ovar           = numpy.zeros( [4 + 24*6 + 3 + 2 + 1 + 24,])
    ovar[0:4]      = variables['a'][:]
    ovar[4:28]     = variables['rx'][:]
    ovar[28:52]    = variables['ry'][:]
    ovar[52:76]    = variables['rz'][:]    
    ovar[76:100]   = variables['offset'][:]
    ovar[100:124]  = variables['ox'][:]
    ovar[124:148]  = variables['oy'][:]
    ovar[148:151]  = variables['normal'][:]
    ovar[151:153]  = variables['center'][:]  
    ovar[153]      = variables['z']
    ovar[154:178]  = variables['gaps']
    
    return ovar


def set_optimization_variable(x):

    out = {
        'a': x[0:4],
        'rx': x[4:28],        #= variables['rx'][:]
        'ry': x[28:52],       #= variables['ry'][:]
        'rz': x[52:76],       #= variables['rz'][:]    
        'offset': x[76:100],  #= variables['offset'][:]
        'ox': x[100:124],     #= variables['ox'][:]
        'oy': x[124:148],     #= variables['oy'][:]
        'normal': x[148:151], #= variables['normal'][:]
        'center': x[151:153], #= variables['center'][:]  
        'z': x[153],          #= variables['z'][:]
        'gaps': x[154:178]    #= variables['gaps'][:]
    }
    
    return out


def get_optimization_variable( out ):

    x = 178 * [0]
    
    x[0:4] = out['a']
    x[4:28] = out['rx']        #= variables['rx'][:]
    x[28:52] = out['ry']       #= variables['ry'][:]
    x[52:76] = out['rz']       #= variables['rz'][:]    
    x[76:100] = out['offset']  #= variables['offset'][:]
    x[100:124] = out['ox']     #= variables['ox'][:]
    x[124:148] = out['oy']     #= variables['oy'][:]
    x[148:151] = out['normal'] #= variables['normal'][:]
    x[151:153] = out['center'] #= variables['center'][:]  
    x[153] = out['z']          #= variables['z']
    x[154:178] = out['gaps']   #= variables['gaps']
    
    return numpy.array(x)


def _worker_annotation_image( img ):

    def _rotation2D_( img, a ):
        if a == 90:
            o = rotate( img, numpy.pi/2.)
        elif a == 180:
            o = rotate( img, numpy.pi)  
        elif a == 270:
            o = rotate( img, 3*numpy.pi/2.) 
        else:
            o = a
        return o
    
    '''
    def _rotation2D_( img, a ):
        if a == 90:
            o = numpy.transpose( numpy.flipud( img)  )
            return o
        elif a == 180:
            o = numpy.transpose( numpy.flipud( img)  )
            o = numpy.transpose( numpy.flipud( o )  )
            return o
        elif a == 270:
            o = numpy.transpose( numpy.flipud( img )  )
            o = numpy.transpose( numpy.flipud( o )  )
            o = numpy.transpose( numpy.flipud( o )  )
            return o
        else:
            return img
    '''
    
    flist = [ ( img[0:1536,0:1536]), ( img[0:1536,1536:3072]), (img[1536:3072,1536:3072]), (img[1536:3072,0:1536]) ]
    top = numpy.hstack( (flist[0], _rotation2D_(flist[1], 90) ) )
    bottom = numpy.hstack(( _rotation2D_(flist[3], 270), _rotation2D_(flist[2], 180)))
    return numpy.vstack((top, bottom))


def annotation_image( img ):
    
    plt.imshow( _worker_annotation_image(img) )
    plt.show()
    

def annotation_points( annotation ):

    if len(annotation)==0:
        return [], [] 
    
    def findStripe(idx):
        return 5 - (idx%1536) // 256 , 255 - idx%256

    def get_module( col, row ):

        if col >= 1536 and row <= 1536:
            col_ = col - 1535 
            row_ = 1535 - row 
            module = 1
        elif col >= 1536 and row >= 1536:
            col_ = col - 1535
            row_ = 1535 - row # - 1536  
            module = 2
        elif col <= 1536 and row >= 1536:
            col_ = col
            row_ = 1535 - row# - 1536 
            module = 3                          
        else:
            col_ = col
            row_ = row 
            module = 0
            
        return module, col_, row_
    
    #dumb mesh
    t = numpy.linspace(-1,1,1536)
    
    #TL = numpy.zeros([1536, 1536]) 
    #TR = numpy.zeros([1536, 1536]) 
    #BL = numpy.zeros([1536, 1536]) 
    #BR = numpy.zeros([1536, 1536]) 

    match = numpy.zeros( [len( annotation ), len( annotation[0] ) + 8] )

    angle = - numpy.array( [0, numpy.pi/2., numpy.pi, 3*numpy.pi/2.0 ] )
    
    for k in range(len( annotation) ):

        col = int(annotation[k][0]) #x 
        row = int(annotation[k][1]) #y
        
        module, col_, row_ = get_module( col, row)

        x = t[col_]
        y = t[row_]
        
        X =   x * numpy.cos(angle[module]) + y * numpy.sin(angle[module])
        Y = - x * numpy.sin(angle[module]) + y * numpy.cos(angle[module])
        
        ix = int((X - t[0])/(t[1]-t[0])) 
        if module > 0:
            iy = 1535 - int((Y - t[0])/(t[1]-t[0])) 
        else:
            iy = int((Y - t[0])/(t[1]-t[0]))

        stripe, ystripe = findStripe(iy)

        #--------------
        # update points
        match[k][0] = ix
        match[k][1] = iy    
        match[k][2] = module
        match[k][3] = stripe
        
        if ystripe==256:
            match[k][4] = ystripe-1
        else:
            match[k][4] = ystripe
        
        #
        # next point from input table
        #

        col = int(annotation[k][2]) #x 
        row = int(annotation[k][3]) #y

        module, col_, row_ = get_module( col, row)
        
        x = t[col_]
        y = t[row_]
        X =   x * numpy.cos(angle[module]) + y * numpy.sin(angle[module])
        Y = - x * numpy.sin(angle[module]) + y * numpy.cos(angle[module])
        ix = int((X - t[0])/(t[1]-t[0])) 

        if module > 0:
            iy = 1535 - int((Y - t[0])/(t[1]-t[0])) 
        else:
            iy = int((Y - t[0])/(t[1]-t[0]))
        
        stripe, ystripe = findStripe(iy)
        
        # update points
        match[k][5] = ix
        match[k][6] = iy    
        match[k][7] = module
        match[k][8] = stripe
        match[k][9] = ystripe

        # update points
        match[k][10]  = MACRO_ANNOTATION( annotation[k][4] )
        match[k][11] = annotation[k][5]
    
    #filter list
    
    from_ = match[:,0:5]
    to_   = match[:,5:10]
    other = match[:,10:14]
    
    c = []
    b = []
    
    for k in range(match.shape[0]):
        
        if (match[k][2]==0 and match[k][7]==1) or (match[k][2]==1 and match[k][7]==0):
            bndry = numpy.zeros([match.shape[1],], dtype=match.dtype )
            #b.append(list(match[k]))
            if match[k][2]==1:
                bndry[0:5]  = to_[k,0:5]
                bndry[5:10] = from_[k,0:5]
                bndry[10:15] = other[k,:]
                c.append(list(bndry))
            else:
                bndry[0:5]  = from_[k,0:5]
                bndry[5:10] = to_[k,0:5]
                bndry[10:15] = other[k,:]
                c.append(list(bndry))
                
        if (match[k][2]==1 and match[k][7]==2) or (match[k][2]==2 and match[k][7]==1):
            #b.append(list(match[k]))
            bndry = numpy.zeros([match.shape[1],], dtype=match.dtype )
                                    
            if match[k][2]==2:
                bndry[0:5]  = to_[k,0:5]
                bndry[5:10] = from_[k,0:5]
                bndry[10:15] = other[k,:]
                c.append(list(bndry))
            else:
                bndry[0:5]  = from_[k,0:5]
                bndry[5:10] = to_[k,0:5]
                bndry[10:15] = other[k,:]
                c.append(list(bndry))

        if (match[k][2]==2 and match[k][7]==3) or (match[k][2]==3 and match[k][7]==2):
            #b.append(list(match[k]))
            bndry = numpy.zeros([match.shape[1],], dtype=match.dtype )
            if match[k][2]==3:
                bndry[0:5]  = to_[k,0:5]
                bndry[5:10] = from_[k,0:5]
                bndry[10:15] = other[k,:]
                c.append(list(bndry))
            else:
                bndry[0:5]  = from_[k,0:5]
                bndry[5:10] = to_[k,0:5]
                bndry[10:15] = other[k,:]
                c.append(list(bndry))

        if (match[k][2]==0 and match[k][7]==3) or (match[k][2]==3 and match[k][7]==0):
            #b.append(list(match[k]))
            bndry = numpy.zeros([match.shape[1],], dtype=match.dtype )
            if match[k][2]==0:
                bndry[0:5]  = to_[k,0:5]
                bndry[5:10] = from_[k,0:5]
                bndry[10:15] = other[k,:]
                c.append(list(bndry))
            else:
                bndry[0:5]  = from_[k,0:5]
                bndry[5:10] = to_[k,0:5]
                bndry[10:15] = other[k,:]
                c.append(list(bndry))
                

    bndry = numpy.array(c) 
    
    # --------
    # Output Pattern:
    #
    # List of information regarding points P1,P2 (index) at the detector
    #
    # match = [ix_P1, iy_P1, module_P1, stripe_P1, ystripe_P1,
    #          ix_P2, iy_P2, module_P2, stripe_P2, ystripe_P2, distance_type, distance,
    #          stripe_idx_P1, stripe_idx_P2]
    #

    match[:,12] = set_module_strip( match[:,2], match[:,3] )
    match[:,13] = set_module_strip( match[:,7], match[:,8] )
    
    return match, bndry


def annotation_points_standard( annotation ):

    if len(annotation)==0:
        return [] 
    
    def findStripe(idx):
        return 5 - (idx%1536) // 256 , 255 - idx%256

    def get_module( col, row ):

        if col >= 1536 and row <= 1536:
            col_ = col - 1535 
            row_ = 1535 - row 
            module = 1
        elif col >= 1536 and row >= 1536:
            col_ = col - 1535
            row_ = 1535 - row # - 1536  
            module = 2
        elif col <= 1536 and row >= 1536:
            col_ = col
            row_ = 1535 - row# - 1536 
            module = 3                          
        else:
            col_ = col
            row_ = row 
            module = 0
            
        return module, col_, row_
    
    #dumb mesh
    t = numpy.linspace(-1,1,1536)
    
    match = numpy.zeros( [len( annotation ), 5] )

    angle = - numpy.array( [0, numpy.pi/2., numpy.pi, 3*numpy.pi/2.0 ] )
    
    for k in range(len( annotation) ):

        col = int(annotation[k,0]) #x 
        row = int(annotation[k,1]) #y
        
        module, col_, row_ = get_module( col, row)

        x = t[col_]
        y = t[row_]
        
        X =   x * numpy.cos(angle[module]) + y * numpy.sin(angle[module])
        Y = - x * numpy.sin(angle[module]) + y * numpy.cos(angle[module])
        
        ix = int((X - t[0])/(t[1]-t[0])) 
        if module > 0:
            iy = 1535 - int((Y - t[0])/(t[1]-t[0])) 
        else:
            iy = int((Y - t[0])/(t[1]-t[0]))

        stripe, ystripe = findStripe(iy)

        #--------------
        # update points
        match[k,0] = ix
        match[k,1] = iy    
        match[k,2] = module
        match[k,3] = stripe
        match[k,4] = ystripe

    # --------
    # Output Pattern:
    #
    # List of information regarding points (their index) at the detector
    #
    # match = [ix_P, iy_P, module_P, stripe_P, ystripe_P]
    #
    
    return match
    
def annotation_image_points( match ):
    #
    # match = [ix_P1, iy_P1, module_P1, stripe_P1, ystripe_P1, ix_P2, iy_P2, module_P2, stripe_P2, ystripe_P2, distance_type, distance ]

    Z1 = numpy.zeros([1536,1536])
    Z2 = numpy.zeros([1536,1536])
    Z3 = numpy.zeros([1536,1536])
    Z4 = numpy.zeros([1536,1536])
    
    images = [  Z1, Z2, Z3, Z4 ]

    for k in range( match.shape[0] ):

        module = int(match[k][2])
        stripe = int(match[k][3])
        iy     = int(match[k][1])
        ix     = int(match[k][0])

        if iy == 1536:
            iy = iy - 1
        if ix == 1536:
            ix = ix - 1
            
        images[module][ iy, ix ] += 1

        module = int(match[k][7])
        stripe = int(match[k][8])
        iy     = int(match[k][6])
        ix     = int(match[k][5])

        if iy == 1536:
            iy = iy-1
        if ix == 1536:
            ix = ix-1
        
        images[module][ iy, ix ] += 1
        

    top    = numpy.hstack( (images[0], images[1]))
    bottom = numpy.hstack( (images[3], images[2])) 

    return numpy.vstack((top, bottom))


####################################################
#
#
# OPTIMIZATION SCHEME FOR 540D ALIGNMENT (NONPLANAR)
#
#
###################################################

def get_tracking_points(normal, RX, RY, RZ, L, dL, a, ox, oy, z, match, boxinfo, D):

    translate  = [ [-1,0], [0,+1], [+1, 0], [0,-1]]

    offset = numpy.abs(numpy.array( [ boxinfo['center'][0], boxinfo['center'][1], boxinfo['center'][0], boxinfo['center'][1] ] ) )
    _mref_ = [ numpy.array([1,1,-1,-1])*offset,  numpy.array([-1,1,1,-1])*offset,
               numpy.array([-1,-1,1,1])*offset,  numpy.array([1,-1,-1,1])*offset ]    
    
    _mod_, _, _ = find_center_position_at_stripe( boxinfo['center'] )

    J          = 256
    P          = 6
    M          = 6
    NOSTRIPES  = 24
    VL         = 2 * J * M
    HL         = 2 * J * P    

    dL           = numpy.array( dL ).flatten() * 1000 # (microns) 
    e3           = numpy.array( build_normal( normal ) ).reshape([3,1]) 
    L            = (numpy.array( 4 * [L] ) ) * 1000 #(microns)    
    angles       = [0, -numpy.pi/2, -numpy.pi, -3*numpy.pi/2.]
    delta        = D['delta']
    mesh         = build_mesh_540D( D['m_'], D['t_'] )
    #-----
    dx = ((float(boxinfo['xbox'][1]) - float(boxinfo['xbox'][0]))) / (HL - 1 )
    dy = ((float(boxinfo['ybox'][1]) - float(boxinfo['ybox'][0]))) / (VL - 1 )
    step = max (dx, dy)
    SCALE_FACTOR = z #MEDIPIX / min(dx, dy)    

    x0 = boxinfo['xbox'][0] * SCALE_FACTOR
    y0 = boxinfo['ybox'][0] * SCALE_FACTOR
    dx = SCALE_FACTOR * step
    dy = SCALE_FACTOR * step
    #-----

    numpoints = match.shape[0]
    vmatch    = numpy.zeros([numpoints, 11] ) #, dtype=numpy.int)
    vmatchd   = numpy.zeros([numpoints, 11] )
    
    z = numpy.array([boxinfo['ccenter'][0], boxinfo['ccenter'][1],0]).reshape([3,1]) 

    ccenter0 = boxinfo['ccenter'][0]
    ccenter1 = boxinfo['ccenter'][1]

    elapsed = []
    
    for k in range( numpoints ):

        start = time.time()
        
        #======================
        #reference point #1 (q)
        
        module0 = int(match[k][2])
        j       = int(match[k][3])
        
        m_module = mesh[module0][0]
        t_module = mesh[module0][1]
                
        ang = angles[module0] - a[module0] * numpy.pi/180.

        sin = numpy.sin(ang)
        cos = numpy.cos(ang)
        RZ_module_array = numpy.array([[cos, -1*sin, 0],[sin,cos,0],[0,0,1]])

        # using (0,0,1) as reference
        n = numpy.array([0,0,1]).reshape([3,1])
        #n = numpy.copy(e3)
        
        RX_ = ( (RX[module0][j] ) *numpy.pi/180.)
        sin = numpy.sin(RX_)
        cos = numpy.cos(RX_)
        RX_array = numpy.array([[1,0,0],[0,cos, -1*sin],[0,sin,cos]])
        
        RZ_ = ( (RZ[module0][j] ) * numpy.pi/180.)
        sin = numpy.sin(RZ_)
        cos = numpy.cos(RZ_)
        RZ_array = numpy.array([[cos, -1*sin, 0],[sin,cos, 0],[0,0,1]])
        
        RY_ = ( (RY[module0][j] ) * numpy.pi/180.)
        sin = numpy.sin(RY_)
        cos = numpy.cos(RY_)
        RY_array = numpy.array([[cos, 0, sin],[0,1,0],[-sin,0,cos]])
        
        n = numpy.dot(RX_array,n)
        n = numpy.dot(RY_array,n)
         
        nperp  = numpy.array([[n[2,0]],[0],[-n[0,0]]])
        ncross = numpy.array([[ -n[1,0]*n[0,0]],[ n[2,0]**2+n[0,0]**2],[-n[1,0]*n[2,0]]])

        n      = numpy.dot(RZ_array,n)
        ncross = numpy.dot(RZ_array,ncross)
        nperp  = numpy.dot(RZ_array,nperp)   
        
        #module rotation

        n = numpy.dot(RZ_module_array,n)
        nperp = numpy.dot(RZ_module_array,nperp)
        ncross = numpy.dot(RZ_module_array,ncross)
   
        Lj = ( L[module0] * numpy.cos( numpy.abs(RX_) ) - j * delta ) + dL[ set_module_strip (module0, j) ]
        
        dpnz = n[0] * z[0] + n[1] * z[1] + n[2] * z[2]  
        Nz = numpy.array([ n[0] * dpnz , n[1] * dpnz , n[2] * dpnz ]).reshape([3,1])

        ##
        sx    = ccenter0
        sy    = ccenter1
        Gz    = [ (1-n[0]*n[0])*z[0] - n[0]*n[1]*z[1] - n[0]*n[2]*z[2], -n[1]*n[0]*z[0] + (1-n[1]*n[1])*z[1] - n[1]*n[2]*z[2] ] 
        px    = Gz[0] - Lj * n[0]
        py    = Gz[1] - Lj * n[1]
        det   = ncross[0] * nperp[1] - ncross[1] * nperp[0] 
        tref  = 1*(( nperp[1] * ( sx - px )  - nperp[0] * (sy - py) ) / det)
        mref  = 1*((-ncross[1] * ( sx - px ) + ncross[0] * (sy - py) ) / det)

        mref = _mref_[_mod_][module0] * MEDIPIX
        
        m = m_module + mref + ox[module0][j]
        t = t_module[j*J: min( J*P,(j+1)*J )] + tref + oy[module0][j]

        x1, y1, _ = projection_virtual_det( m[int(match[k][0])],t[int(match[k][4])],n,nperp,ncross,Lj,L[module0],e3,Nz)

        x1 = float(x1)
        y1 = float(y1)

        x1 += translate[ module0 ][0] * mref
        y1 += translate[ module0 ][1] * mref
               
        ix1 = int((x1 - x0)/(dx) ) 
        iy1 = VL -  int((y1 - y0) /(dy))  
        
        #======================
        #reference point #2 (w)
        module1 = int(match[k][7])
        j       = int(match[k][8])
        
        m_module = mesh[module1][0]
        t_module = mesh[module1][1]
        
        ang = angles[module1] - a[module1] * numpy.pi/180.
        
        sin = numpy.sin(ang)
        cos = numpy.cos(ang)
        RZ_module_array = numpy.array([[cos, -1*sin, 0],[sin,cos,0],[0,0,1]])

        #using (0,0,1) as reference
        beta = numpy.array([0,0,1]).reshape([3,1])
        #beta = numpy.copy(e3)
        
        RX_ = ( (RX[module1][j] ) * numpy.pi/180.)
        sin = numpy.sin(RX_)
        cos = numpy.cos(RX_)
        RX_array = numpy.array([[1,0,0],[0,cos, -1*sin],[0,sin,cos]])
        
        RZ_ = ( (RZ[module1][j] ) * numpy.pi/180.)
        sin = numpy.sin(RZ_)
        cos = numpy.cos(RZ_)
        RZ_array = numpy.array([[cos, -1*sin, 0],[sin,cos, 0],[0,0,1]])
        
        RY_ = ( (RY[module1][j] ) * numpy.pi/180.)
        sin = numpy.sin(RY_)
        cos = numpy.cos(RY_)
        RY_array = numpy.array([[cos, 0, sin],[0,1,0],[-sin,0,cos]])
        
        beta = numpy.dot(RX_array,beta)
        beta = numpy.dot(RY_array,beta)
        
        betaperp  = numpy.array([[beta[2,0]],[0],[-beta[0,0]]])
        betacross = numpy.array([[ -beta[1,0]*beta[0,0]],[ beta[2,0]**2+beta[0,0]**2],[-beta[1,0]*beta[2,0]]])

        beta      = numpy.dot(RZ_array,beta)    
        betacross = numpy.dot(RZ_array,betacross)    
        betaperp  = numpy.dot(RZ_array,betaperp)    
        
        #module rotation    
        beta = numpy.dot(RZ_module_array,beta)
        betaperp = numpy.dot(RZ_module_array,betaperp)
        betacross = numpy.dot(RZ_module_array,betacross)
        
        dpnz = beta[0] * z[0] + beta[1] * z[1] + beta[2] * z[2]  
        Nz = numpy.array([ beta[0] * dpnz , beta[1] * dpnz , beta[2] * dpnz ]).reshape([3,1])
        
        Ls = ( L[module1] * numpy.cos( numpy.abs(RX_) ) - j * delta ) + dL[ set_module_strip (module1, j) ]
        
        sx    = ccenter0 # boxinfo['ccenter'][0]
        sy    = ccenter1 # boxinfo['ccenter'][1]
        Gz    = [ (1-beta[0]*beta[0] )*z[0] - beta[0]*beta[1]*z[1] - beta[0]*beta[2]*z[2], -beta[1]*beta[0]*z[0] + (1-beta[1]*beta[1])*z[1] - beta[1]*beta[2]*z[2] ] 
        px    = Gz[0] - Ls * beta[0]
        py    = Gz[1] - Ls * beta[1]
        det   = betacross[0] * betaperp[1] - betacross[1] * betaperp[0] 
        tref  = 1*(( betaperp[1] * ( sx - px )  - betaperp[0] * (sy - py) ) / det)
        mref  = 1*((-betacross[1] * ( sx - px ) + betacross[0] * (sy - py) ) / det)
        
        mref = _mref_[_mod_][module1] * MEDIPIX

        m = m_module + mref + ox[module1][j]
        t = t_module[j*J: min( J*P,(j+1)*J )] + tref + oy[module1][j] 
                
        x2, y2, _ = projection_virtual_det( m[ int(match[k][5]) ],t[int(match[k][9])], beta,betaperp,betacross,Ls,L[module1],e3,Nz)
        x2 = float(x2)
        y2 = float(y2)
        
        x2 += translate[ module1 ][0] * mref
        y2 += translate[ module1 ][1] * mref
        
        ix2 =  int( (x2 - x0) /(dx) ) 
        iy2 = VL -  int((y2 - y0) /(dy))    

        euclidean = numpy.abs( numpy.sqrt( (x2-x1)**2 + (y2-y1)**2 ) - match[k][11] ) / numpy.sqrt(dx**2 + dy**2)

        vmatch[k] = [ix1, iy1, ix2, iy2, match[k][10], match[k][11], dx, dy,
                     numpy.abs(ix1-ix2), numpy.abs(iy1-iy2), euclidean ]

        vmatchd[k] = [x1, y1, x2, y2, match[k][10], match[k][11], dx, dy,
                      numpy.abs(x1-x2)/dx, numpy.abs(y1-y2)/dy, euclidean ]

        #elapsed.append( time.time() - start )

    #print('E:',elapsed)
        
    return vmatch, vmatchd


def get_tracking_points_vec(normal, RX, RY, RZ, L, dL, a, ox, oy, z, match, boxinfo, D, outputType ):
    
    translate  = [ [-1,0], [0,+1], [+1, 0], [0,-1]]

    offset = numpy.abs(numpy.array( [ boxinfo['center'][0], boxinfo['center'][1], boxinfo['center'][0], boxinfo['center'][1] ] ) )
    _mref_ = [ numpy.array([1,1,-1,-1])*offset,  numpy.array([-1,1,1,-1])*offset,
               numpy.array([-1,-1,1,1])*offset,  numpy.array([1,-1,-1,1])*offset ]    
    
    _mod_, _, _ = find_center_position_at_stripe( boxinfo['center'] )

    J          = 256
    P          = 6
    M          = 6
    NOSTRIPES  = 24
    VL         = 2 * J * M
    HL         = 2 * J * P    

    dL           = numpy.array(dL).flatten() * 1000 #(microns)
    e3           = build_normal( normal ) 
    L            = (numpy.array( 4 * [L] )) * 1000 #(microns)
    a            = numpy.array(a)
    angles       = numpy.array( [0, -numpy.pi/2, -numpy.pi, -3*numpy.pi/2.] )
    delta        = D['delta']
    mesh         = numpy.array( build_mesh_540D( D['m_'], D['t_'] ) )
    
    dx = ((float(boxinfo['xbox'][1]) - float(boxinfo['xbox'][0]))) / (HL - 1 )
    dy = ((float(boxinfo['ybox'][1]) - float(boxinfo['ybox'][0]))) / (VL - 1 )
    step = max (dx, dy)
    SCALE_FACTOR = z # MEDIPIX / min(dx, dy)    

    x0 = boxinfo['xbox'][0] * SCALE_FACTOR
    y0 = boxinfo['ybox'][0] * SCALE_FACTOR
    dx = SCALE_FACTOR * step
    dy = SCALE_FACTOR * step
    #-----

    numpoints = match.shape[0]
    
    z = numpy.array([boxinfo['ccenter'][0], boxinfo['ccenter'][1],0]).reshape([3,1]) 

    ccenter0 = boxinfo['ccenter'][0]
    ccenter1 = boxinfo['ccenter'][1]

    elapsed = []

    # assuming reference to (0,0,1)
    nvec0 = lambda x,y: numpy.sin(y) * numpy.cos(x)
    nvec1 = lambda x,y: - numpy.sin(x)
    nvec2 = lambda x,y: numpy.cos(y)*numpy.cos(x)

    def rotz( n, rz):
        n0_ = n[0] * numpy.cos(rz) - n[1] * numpy.sin(rz)
        n1_ = n[0] * numpy.sin(rz) + n[1] * numpy.cos(rz)
        return n0_, n1_, n[2]
        
    a2r = numpy.pi/180.0
    RX = ( RX.reshape([24,]) ) * a2r
    RY = ( RY.reshape([24,]) ) * a2r
    RZ = ( RZ.reshape([24,]) ) * a2r
    ox_ = ox.reshape([24,])
    oy_ = oy.reshape([24,])

    N = match.shape[0]
    
    #======================
    #reference point #1 (q)
    module0 = (match[:,2]).astype(int)
    j       = (match[:,3]).astype(int)
    stripe  = set_module_strip(module0, j)

    #slicing given angles
    RX_ = RX[stripe] 
    RY_ = RY[stripe] 
    RZ_ = RZ[stripe] 
    ang = angles[ module0 ] - a[ module0 ] * a2r 
    L_module0 = L[module0]
    dL_module0 = dL[stripe]
    mref      = _mref_[_mod_][module0 ] * MEDIPIX  
    
    n0 = nvec0(RX_,RY_)
    n1 = nvec1(RX_,RY_)
    n2 = nvec2(RX_,RY_)
    
    nperp0 = n2
    nperp1 = 0 * n2
    nperp2 = -n0
    
    ncross0 = -n1*n0
    ncross1 = n2**2 + n0**2
    ncross2 = -n1*n2

    n0, n1, n2 = rotz( [n0, n1, n2], RZ_)
    nperp0, nperp1, nperp2 = rotz( [nperp0, nperp1, nperp2], RZ_)
    ncross0, ncross1, ncross2 = rotz( [ncross0, ncross1, ncross2], RZ_)
    
    #
    
    n0, n1, n2 = rotz( [n0, n1, n2], ang)
    nperp0, nperp1, nperp2 = rotz( [nperp0, nperp1, nperp2], ang)
    ncross0, ncross1, ncross2 = rotz( [ncross0, ncross1, ncross2], ang)

    n = [n0,n1,n2]
    nperp = [nperp0,nperp1,nperp2]
    ncross = [ncross0,ncross1,ncross2]
    
    Lj =  ( L_module0 * numpy.cos(RX_) - j * delta ) + dL_module0
    
    dpnz = n0 * z[0] + n1 * z[1] + n2 * z[2]  
    Nz0 = n0 * dpnz
    Nz1 = n1 * dpnz
    Nz2 = n2 * dpnz
    
    Nz = [Nz0, Nz1, Nz2]
    
    ##
    sx    = ccenter0
    sy    = ccenter1
    Gz = [ (1-n0*n0)*z[0] - n0*n1*z[1] - n0*n2*z[2], -n1*n0*z[0] + (1-n1*n1)*z[1] - n1*n2*z[2] ] 
    px    =  Gz[0] - Lj * n0
    py    =  Gz[1] - Lj * n1
    
    det   =  ncross0 * nperp1 - ncross1 * nperp0 
    tref  =  1*(( nperp1 * ( sx - px) - nperp0 * (sy - py) ) / det )

    mm = match[:,0].astype(int)
    tt = match[:,4].astype(int)

    m_t = numpy.array( [ [ ( mesh[module0[k]][0] + mref[k] + ox_[stripe[k]])[mm[k]], ( mesh[module0[k]][1][j[k]*J: min( J*P,(j[k]+1)*J )])[tt[k]] + tref[k] + oy_[stripe[k]],
                           translate[module0[k]][0]*mref[k], translate[module0[k]][1]*mref[k] ] for k in range(N) ] )
    
    meval = m_t[:,0]
    teval = m_t[:,1]
    transx= m_t[:,2]
    transy= m_t[:,3]

    y1 = - L_module0 *  ( Nz1  - Lj * n1 + teval * ncross1 + meval * nperp1 ) / ( (Nz0*e3[0]+Nz1*e3[1]+Nz2*e3[2]) - Lj*(n0*e3[0]+n1*e3[1]+n2*e3[2])+ teval * (ncross0*e3[0]+ncross1*e3[1]+ncross2*e3[2]) +  meval * (nperp0*e3[0]+nperp1*e3[1]+nperp2*e3[2]) )  + transy 
    
    x1 = - L_module0 * ( Nz0 - Lj * n0 + teval * ncross0 + meval * nperp0 ) / ( (Nz0*e3[0]+Nz1*e3[1]+Nz2*e3[2]) - Lj*(n0*e3[0]+n1*e3[1]+n2*e3[2]) + teval * (ncross0*e3[0]+ncross1*e3[1]+ncross2*e3[2]) +  meval * (nperp0*e3[0]+nperp1*e3[1]+nperp2*e3[2]) ) + transx


    #======================
    #reference point #2 (w)
    module1 = (match[:,7]).astype(int)
    j       = (match[:,8]).astype(int)
    stripe  = set_module_strip(module1, j)    

    #slicing given angles
    RX_ = RX[stripe] 
    RY_ = RY[stripe] 
    RZ_ = RZ[stripe] 
    ang = angles[ module1 ] - a[ module1 ] * a2r
    L_module1 = L[module1]
    dL_module1 = dL[stripe]
    mref      = _mref_[_mod_][module1 ] * MEDIPIX
       
    n0 = nvec0(RX_,RY_)
    n1 = nvec1(RX_,RY_)
    n2 = nvec2(RX_,RY_)
    
    nperp0 = n2
    nperp1 = 0 * n2
    nperp2 = -n0
    
    ncross0 = -n1*n0
    ncross1 = n2**2 + n0**2
    ncross2 = -n1*n2

    n0, n1, n2 = rotz( [n0, n1, n2], RZ_)
    nperp0, nperp1, nperp2 = rotz( [nperp0, nperp1, nperp2], RZ_)
    ncross0, ncross1, ncross2 = rotz( [ncross0, ncross1, ncross2], RZ_)

    #
    
    n0, n1, n2 = rotz( [n0, n1, n2], ang)
    nperp0, nperp1, nperp2 = rotz( [nperp0, nperp1, nperp2], ang)
    ncross0, ncross1, ncross2 = rotz( [ncross0, ncross1, ncross2], ang)

    n = [n0,n1,n2]
    nperp = [nperp0,nperp1,nperp2]
    ncross = [ncross0,ncross1,ncross2]

    Lj = (L_module1 * numpy.cos(RX_) - j * delta) + dL_module1
    
    dpnz = n0 * z[0] + n1 * z[1] + n2 * z[2]  
    Nz0 = n0 * dpnz
    Nz1 = n1 * dpnz
    Nz2 = n2 * dpnz
    
    Nz = [Nz0, Nz1, Nz2]
    
    ##
    sx    = ccenter0
    sy    = ccenter1
    Gz = [ (1-n0*n0)*z[0] - n0*n1*z[1] - n0*n2*z[2], -n1*n0*z[0] + (1-n1*n1)*z[1] - n1*n2*z[2] ] 
    px    =  Gz[0] - Lj * n0
    py    =  Gz[1] - Lj * n1
    det   =  ncross0 * nperp1 - ncross1 * nperp0 
    tref  =  1*(( nperp1 * ( sx - px) - nperp0 * (sy - py) ) / det ) 

    mm = match[:,5].astype(int)
    tt = match[:,9].astype(int)

    m_t = numpy.array( [ [ ( mesh[module1[k]][0] + mref[k] + ox_[stripe[k]])[mm[k]], ( mesh[module1[k]][1][j[k]*J: min( J*P,(j[k]+1)*J )])[tt[k]] + tref[k] + oy_[stripe[k]],
                           translate[module1[k]][0]*mref[k], translate[module1[k]][1]*mref[k] ] for k in range(N) ] )
    
    meval = m_t[:,0]
    teval = m_t[:,1]
    transx= m_t[:,2]
    transy= m_t[:,3]
    
    y2 = - L_module1 *  ( Nz1  - Lj * n1 + teval * ncross1 + meval * nperp1 ) / ( (Nz0*e3[0]+Nz1*e3[1]+Nz2*e3[2]) - Lj*(n0*e3[0]+n1*e3[1]+n2*e3[2])+ teval * (ncross0*e3[0]+ncross1*e3[1]+ncross2*e3[2]) +  meval * (nperp0*e3[0]+nperp1*e3[1]+nperp2*e3[2]) )  + transy 
    
    x2 = - L_module1 * ( Nz0 - Lj * n0 + teval * ncross0 + meval * nperp0 ) / ( (Nz0*e3[0]+Nz1*e3[1]+Nz2*e3[2]) - Lj*(n0*e3[0]+n1*e3[1]+n2*e3[2]) + teval * (ncross0*e3[0]+ncross1*e3[1]+ncross2*e3[2]) +  meval * (nperp0*e3[0]+nperp1*e3[1]+nperp2*e3[2]) ) + transx
    
    ###

    hor = ( match[:,10] == HORIZONTAL )
    ver = ( match[:,10] == VERTICAL )
    euc = ( match[:,10] == EUCLIDEAN )
    
    x1_h = x1[ hor ]
    x2_h = x2[ hor ]

    y1_v = y1[ ver ]
    y2_v = y2[ ver ]

    x1_e = x1[ euc ]
    x2_e = x2[ euc ]

    y1_e = y1[ euc ]
    y2_e = y2[ euc ]

    dist = match[ euc ]
    
    fe = numpy.abs( numpy.sqrt( (x2_e - x1_e)**2 + (y2_e - y1_e)**2 ) - dist[:,11] ) / numpy.sqrt(dx**2 + dy**2)
    fx = numpy.abs(x2_h - x1_h ) / dx
    fy = numpy.abs(y2_v - y1_v ) / dy

    if fe.shape[0] == 0:
        fe = - numpy.ones([1,])
    if fx.shape[0] == 0:
        fx = - numpy.ones([1,])
    if fy.shape[0] == 0:
        fy = - numpy.ones([1,])


        
    vec = numpy.hstack( ( fe , fx, fy ) )
    
    if outputType== "norm":
        fobj = numpy.abs(vec).sum() #(vec**2).sum()
    else:
        fobj = vec[ ~ numpy.all(vec == -1, axis=0) ].flatten()

    return fobj


def get_tracking_points_vec_standard(normal, RX, RY, RZ, L, dL, a, ox, oy, z, match, boxinfo, D):

    translate  = [ [-1,0], [0,+1], [+1, 0], [0,-1]]

    offset = numpy.abs(numpy.array( [ boxinfo['center'][0], boxinfo['center'][1], boxinfo['center'][0], boxinfo['center'][1] ] ) )
    _mref_ = [ numpy.array([1,1,-1,-1])*offset,  numpy.array([-1,1,1,-1])*offset,
               numpy.array([-1,-1,1,1])*offset,  numpy.array([1,-1,-1,1])*offset ]    
    
    _mod_, _, _ = find_center_position_at_stripe( boxinfo['center'] )

    J          = 256
    P          = 6
    M          = 6
    NOSTRIPES  = 24
    VL         = 2 * J * M
    HL         = 2 * J * P    

    dL           = numpy.array( dL ).flatten() * 1000 #(microns)
    e3           = build_normal ( normal ) 
    L            = (numpy.array( 4 * [L] )) * 1000 #(microns)
    a            = numpy.array(a)
    angles       = numpy.array( [0, -numpy.pi/2, -numpy.pi, -3*numpy.pi/2.] )
    delta        = D['delta']
    mesh         = numpy.array( build_mesh_540D( D['m_'], D['t_'] ) )
    
    dx = ((float(boxinfo['xbox'][1]) - float(boxinfo['xbox'][0]))) / (HL - 1 )
    dy = ((float(boxinfo['ybox'][1]) - float(boxinfo['ybox'][0]))) / (VL - 1 )
    step = max (dx, dy)

    print('---->',z)
    
    SCALE_FACTOR = z * MEDIPIX / min(dx, dy)    

    x0 = boxinfo['xbox'][0] * SCALE_FACTOR
    y0 = boxinfo['ybox'][0] * SCALE_FACTOR
    dx = SCALE_FACTOR * step
    dy = SCALE_FACTOR * step
    #-----

    numpoints = match.shape[0]
    
    z = numpy.array([boxinfo['ccenter'][0], boxinfo['ccenter'][1],0]).reshape([3,1]) 

    ccenter0 = boxinfo['ccenter'][0]
    ccenter1 = boxinfo['ccenter'][1]

    elapsed = []

    # assuming referece to (0,0,1)
    nvec0 = lambda x,y: numpy.sin(y)*numpy.cos(x)
    nvec1 = lambda x,y: - numpy.sin(x)
    nvec2 = lambda x,y: numpy.cos(y)*numpy.cos(x)
    
    def rotz( n, rz):
        n0_ = n[0] * numpy.cos(rz) - n[1] * numpy.sin(rz)
        n1_ = n[0] * numpy.sin(rz) + n[1] * numpy.cos(rz)
        return n0_, n1_, n[2]
        
    a2r = numpy.pi/180.0
    RX = ( RX.reshape([24,]) ) * a2r
    RY = ( RY.reshape([24,]) ) * a2r
    RZ = ( RZ.reshape([24,]) ) * a2r
    ox_ = ox.reshape([24,])
    oy_ = oy.reshape([24,])

    N = match.shape[0]
    
    #======================
    #reference point #1 (q)
    module0 = (match[:,2]).astype(int)
    j       = (match[:,3]).astype(int)
    stripe  = set_module_strip(module0, j)

    #slicing given angles
    RX_ = RX[stripe] 
    RY_ = RY[stripe] 
    RZ_ = RZ[stripe] 
    ang = angles[ module0 ] - a[ module0 ] * a2r 
    L_module0 = L[module0]
    dL_module0 = dL[stripe]
    mref      = _mref_[_mod_][module0 ] * MEDIPIX  
    
    n0 = nvec0(RX_,RY_)
    n1 = nvec1(RX_,RY_)
    n2 = nvec2(RX_,RY_)
    
    nperp0 = n2
    nperp1 = 0 * n2
    nperp2 = -n0
    
    ncross0 = -n1*n0
    ncross1 = n2**2 + n0**2
    ncross2 = -n1*n2

    n0, n1, n2 = rotz( [n0, n1, n2], RZ_)
    nperp0, nperp1, nperp2 = rotz( [nperp0, nperp1, nperp2], RZ_)
    ncross0, ncross1, ncross2 = rotz( [ncross0, ncross1, ncross2], RZ_)
    
    #
    
    n0, n1, n2 = rotz( [n0, n1, n2], ang)
    nperp0, nperp1, nperp2 = rotz( [nperp0, nperp1, nperp2], ang)
    ncross0, ncross1, ncross2 = rotz( [ncross0, ncross1, ncross2], ang)

    n = [n0,n1,n2]
    nperp = [nperp0,nperp1,nperp2]
    ncross = [ncross0,ncross1,ncross2]
    
    Lj =  ( L_module0 * numpy.cos(RX_) - j * delta ) + dL_module0
    
    dpnz = n0 * z[0] + n1 * z[1] + n2 * z[2]  
    Nz0 = n0 * dpnz
    Nz1 = n1 * dpnz
    Nz2 = n2 * dpnz
    
    Nz = [Nz0, Nz1, Nz2]
    
    ##
    sx    = ccenter0
    sy    = ccenter1
    Gz = [ (1-n0*n0)*z[0] - n0*n1*z[1] - n0*n2*z[2], -n1*n0*z[0] + (1-n1*n1)*z[1] - n1*n2*z[2] ] 
    px    =  Gz[0] - Lj * n0
    py    =  Gz[1] - Lj * n1
    
    det   =  ncross0 * nperp1 - ncross1 * nperp0 
    tref  =  1*(( nperp1 * ( sx - px) - nperp0 * (sy - py) ) / det )

    mm = match[:,0].astype(int)
    tt = match[:,4].astype(int)
    
    m_t = numpy.array( [ [ ( mesh[module0[k]][0] + mref[k] + ox_[stripe[k]])[mm[k]], ( mesh[module0[k]][1][j[k]*J: min( J*P,(j[k]+1)*J )])[tt[k]] + tref[k] + oy_[stripe[k]],
                           translate[module0[k]][0]*mref[k], translate[module0[k]][1]*mref[k] ] for k in range(N) ] )
    
    meval = m_t[:,0]
    teval = m_t[:,1]
    transx= m_t[:,2]
    transy= m_t[:,3]

    output = numpy.zeros([N,4])

    #y1
    y1 = - L_module0 *  ( Nz1  - Lj * n1 + teval * ncross1 + meval * nperp1 ) / ( (Nz0*e3[0]+Nz1*e3[1]+Nz2*e3[2]) - Lj*(n0*e3[0]+n1*e3[1]+n2*e3[2])+ teval * (ncross0*e3[0]+ncross1*e3[1]+ncross2*e3[2]) +  meval * (nperp0*e3[0]+nperp1*e3[1]+nperp2*e3[2]) )  + transy 
    
    #x1
    x1 = - L_module0 * ( Nz0 - Lj * n0 + teval * ncross0 + meval * nperp0 ) / ( (Nz0*e3[0]+Nz1*e3[1]+Nz2*e3[2]) - Lj*(n0*e3[0]+n1*e3[1]+n2*e3[2]) + teval * (ncross0*e3[0]+ncross1*e3[1]+ncross2*e3[2]) +  meval * (nperp0*e3[0]+nperp1*e3[1]+nperp2*e3[2]) ) + transx

    output[:,1] = y1
    output[:,0] = x1

    output[:,2] = ( (x1 - x0) /(dx) ).astype(int) 
    output[:,3] = VL - ((y1 - y0) /(dy)).astype(int)
    
    return output
    
    
def tracking540D( params, match, *args):
   
    J = 256
    P = 6
    M = 6
    VL = 2 * J * M
    HL = 2 * J * P    
     
    L = params['distance']
    
    if 's' in params.keys():
        susp = params['s']
    else:
        susp = [0,0]
    
    v =  params['normal']
    a  = params['a']
    z  = params['z']
    offsetx = params['ox']
    offsety = params['oy']
    center = params['center']
    typedet = params['typedet']

    RxM = numpy.array( [ params['rotx'][0], params['rotx'][1], params['rotx'][2], params['rotx'][3] ] )
    RyM = numpy.array( [ params['roty'][0], params['roty'][1], params['roty'][2], params['roty'][3] ] )
    RzM = numpy.array( [ params['rotz'][0], params['rotz'][1], params['rotz'][2], params['rotz'][3] ] )
    
    Ox = numpy.array( [ params['ox'][0], params['ox'][1], params['ox'][2], params['ox'][3] ] )
    Oy = numpy.array( [ params['oy'][0], params['oy'][1], params['oy'][2], params['oy'][3] ] )
    
    dL = numpy.array( [ params['offset'][0], params['offset'][1], params['offset'][2], params['offset'][3] ] )
   
    if not args:
        boxinfo = get_bounding_box( v, RxM, RyM, RzM, L, dL, a, center, typedet )  
        D       = parameters(J, P, M, False, typedet)
    else:
        params  = args[0]
        boxinfo = params[0]
        D       = params[1]
        
    vmatch = get_tracking_points(v, RxM, RyM, RzM, L, dL, a, Ox, Oy, z, match, boxinfo, D)
    
    return vmatch


def tracking540D_vec(params, match, *args):
   
    J = 256
    P = 6
    M = 6
    VL = 2 * J * M
    HL = 2 * J * P    
        
    L = params['distance']
    
    if 's' in params.keys():
        susp = params['s']
    else:
        susp = [0,0]
    
    v =  params['normal'] 
    a  = params['a']
    z  = params['z']
    offsetx = params['ox']
    offsety = params['oy']
    center = params['center']
    typedet = params['typedet']

    RxM = numpy.array( [ params['rotx'][0], params['rotx'][1], params['rotx'][2], params['rotx'][3] ] )
    RyM = numpy.array( [ params['roty'][0], params['roty'][1], params['roty'][2], params['roty'][3] ] )
    RzM = numpy.array( [ params['rotz'][0], params['rotz'][1], params['rotz'][2], params['rotz'][3] ] )
    Ox = numpy.array( [ params['ox'][0], params['ox'][1], params['ox'][2], params['ox'][3] ] )
    Oy = numpy.array( [ params['oy'][0], params['oy'][1], params['oy'][2], params['oy'][3] ] )

    dL = numpy.array( [ params['offset'][0], params['offset'][1], params['offset'][2], params['offset'][3] ] )
        
    if not args:
        boxinfo = get_bounding_box( v, RxM, RyM, RzM, L, dL, a, center, typedet )  
        D       = parameters(J, P, M, False, typedet)
    else:
        params  = args[0]
        boxinfo = params[0]
        D       = params[1]
        if len(params)==2:
            outputType  = "norm"
        else:
            outputType  = params[2]
        
    criteria = get_tracking_points_vec(v, RxM, RyM, RzM, L, dL, a, Ox, Oy, z, match, boxinfo, D, outputType )
    
    return criteria


def tracking540D_vec_standard(params, match, *args):
   
    J = 256
    P = 6
    M = 6
    VL = 2 * J * M
    HL = 2 * J * P    
        
    L = params['distance']
    
    if 's' in params.keys():
        susp = params['s']
    else:
        susp = [0,0]
    
    v = params['normal']
    a  = params['a']
    z  = params['z']
    offsetx = params['ox']
    offsety = params['oy']
    center = params['center']
    typedet = params['typedet']

    RxM = numpy.array( [ params['rotx'][0], params['rotx'][1], params['rotx'][2], params['rotx'][3] ] )
    RyM = numpy.array( [ params['roty'][0], params['roty'][1], params['roty'][2], params['roty'][3] ] )
    RzM = numpy.array( [ params['rotz'][0], params['rotz'][1], params['rotz'][2], params['rotz'][3] ] )
    Ox = numpy.array( [ params['ox'][0], params['ox'][1], params['ox'][2], params['ox'][3] ] )
    Oy = numpy.array( [ params['oy'][0], params['oy'][1], params['oy'][2], params['oy'][3] ] )
        
    dL = numpy.array( [ params['offset'][0], params['offset'][1], params['offset'][2], params['offset'][3] ] )
    
    if not args:
        boxinfo = get_bounding_box( v, RxM, RyM, RzM, L, dL, a, center, typedet )  
        D       = parameters(J, P, M, False, typedet)
    else:
        params  = args[0]
        boxinfo = params[0]
        D       = params[1]
        
    criteria = get_tracking_points_vec_standard(v, RxM, RyM, RzM, L, dL, a, Ox, Oy, z, match, boxinfo, D)
    
    return criteria


def _criteria_offset_540D_nonplanar_( x, *args ):

    startEvalC = time.time()
    
    params = args[0]
    
    L0        = params[0]
    var       = params[1]
    x0        = params[2]
    tracking  = params[3]
    boxinfo   = params[4]
    D         = params[5]
    
    y          = x0
    y[var > 0] = x    
    ydic       = set_optimization_variable ( y )
    
    paramsp  = {'geo':'nonplanar','opt':False, 'mode':'virtual', 'hexa': range(24), 'x': ydic }
    project = get_detector_dictionary( L0, paramsp )
    
    crit = tracking540D_vec(project, tracking, (boxinfo, D, "vec" ))
        
    #print('--> objective function:', crit)
    #print('--> elapsed time for objective function: {}'.format(time.time() - startEvalC ))
    #print((crit**2).sum())

    return (crit**2).sum()


def _worker_optimize_offset_540D_nonplanar_(xstart, variables, tracking, L0, *args):    

    #-------------
    #optimization variables: ovar
    #
    ovar = optimization_variables ( variables ) 

    x0 = get_optimization_variable(xstart)
    
    #
    #boxinfo computed with project values (remain constat for optimization procedures)
    boxinfo = get_bbox540D( get_detector_dictionary( L0  ) )
    J       = 256
    P       = 6
    M       = 6
    D       = parameters(J, P, M, False, "nonplanar")
        
    params = (L0, ovar, x0, tracking, boxinfo, D) 
    
    x0_  = x0[ovar > 0]

    if len(args) > 0:
        tolerance = args[0]
        bnds = get_project_bounds_geometry( ovar, (x0, tolerance) )
    else:
        bnds = get_project_bounds_geometry( ovar, (x0, ) )
        
    #res = minimize( _criteria_offset_540D_nonplanar_, x0_, args=(params,),  options={'maxiter':4000}, bounds=bnds, method="SLSQP", tol=1e-8)

    res = minimize( _criteria_offset_540D_nonplanar_, x0_, args=(params,),  options={'maxiter':4000}, method="COBYLA")
 
    x = x0
    x[ ovar > 0] = res.x

    out = set_optimization_variable ( x )
    
    return out


def see_annotations_540D_nonplanar( img, piannofiles ):

    matchDet = numpy.array([]).reshape([0,14])
    annot = []
    for m in range(len(piannofiles)):

        with open(piannofiles[m], 'rb') as f:
            pianno = pickle.load(f)

        L0 = pianno['detector_distance']
            
        x = {k:v for k,v in pianno.items() if type(k) is not str}

        print(x)
        
        _annot_ = []
        for k in range(len(x)):
            _a_ = 6*[0]
            if k in x.keys():
                if x[k][3] != 'euclidean':
                    _a_[0:2] = numpy.flipud( x[k][0][1:3] ).astype(int)
                    _a_[2:4] = numpy.flipud( x[k][1][1:3] ).astype(int)
                    if x[k][3]=='vertical':
                        _a_[4] = 'y'
                        _a_[5] = -1
                    elif x[k][3]=='horizontal':
                        _a_[4] = 'x'
                        _a_[5] = -1
                    #else:
                    #    
                    #    _a_[4] = 'e'
                    #    _a_[5] =  numpy.array( x[k][2] ).astype(int)
                    _annot_.append(_a_)

        #points matching at the detector space    
        _matchDet_, _ = annotation_points( _annot_ )
     
        annot.append( _annot_ )
        matchDet = numpy.vstack( (matchDet, _matchDet_) )
    
    params  = {'geo':'nonplanar', 'opt':False, 'mode':'virtual', 'hexa': range(24), 'x': x }

    #tracking of all annotated points on the restored image
    track = annotation_image_points( matchDet )

    annotation = {'annotation': annot, 'matchdet': matchDet, 'track': track}

    ## image with annotations
    
    annot = annotation['annotation']
    track = annotation['track']
    
    imga = _worker_annotation_image(img)

    imgo = numpy.clip( numpy.copy(img), 0, img.max())
    imga = numpy.clip( imga, 0, img.max() )

    A = numpy.zeros(imgo.shape)
    for m in range(len(annot)):
        for k in range(len(annot[m])):
            A[annot[m][k][1], annot[m][k][0] ] = 1 #2* img.max()
            A[annot[m][k][3], annot[m][k][2] ] = 1 #2* img.max()

    A = gaussian_filter(A, sigma=2) 

    track = gaussian_filter(track, sigma=2) 
    
    imga[ A > 0 ] = 2 * img.max()
    imgo[ track > 0 ] = 2 * img.max()
    
    return imga, imgo


def optimize_540D_nonplanar( piannofiles ):

    #optimization variable!
    variables = {
        'a': 4*[0],
        'rx': 24*[0], 
        'ry': 24*[0],
        'rz': 24*[0],
        'offset': 24*[0],
        'ox': 24*[1], #[1,0,0,0,0,0, 1,0,0,0,0,0, 1,0,0,0,0,0, 1,0,0,0,0,0 ], #24*[1],
        'oy': 24*[1], #[1,0,0,0,0,0, 1,0,0,0,0,0, 1,0,0,0,0,0, 1,0,0,0,0,0 ], #24*[1],
        'normal': 3*[0],
        'center': 2*[0],
        'gaps': 24*[0],
        'z': 0
    }

    matchDet = numpy.array([]).reshape([0,14])
    annot = []
    for m in range(len(piannofiles)):

        with open(piannofiles[m], 'rb') as f:
            pianno = pickle.load(f)

        L0 = pianno['detector_distance']
            
        x = {k:v for k,v in pianno.items() if type(k) is not str}
        
        _annot_ = []
        for k in range(len(x)):
            _a_ = 6*[0]
            if k in x.keys():
                if x[k][3] != 'euclidean':
                    _a_[0:2] = numpy.flipud( x[k][0][1:3] ).astype(int)
                    _a_[2:4] = numpy.flipud( x[k][1][1:3] ).astype(int)
                    if x[k][3]=='vertical':
                        _a_[4] = 'y'
                        _a_[5] = -1
                    elif x[k][3]=='horizontal':
                        _a_[4] = 'x'
                        _a_[5] = -1
                    #else:
                    #    #_a_[4] = 'e'
                    #    #_a_[5] = numpy.array( x[k][2] ).astype(int)
                    _annot_.append(_a_)

        
        #points matching at the detector space    
        _matchDet_, _ = annotation_points( _annot_ )
     
        annot.append( _annot_ )
        matchDet = numpy.vstack( (matchDet, _matchDet_) )

    
    xstart = set_optimization_variable( get_project_values_geometry( {'geo':'nonplanar','opt':False, 'mode':'virtual', 'hexa': range(24)}) )

    x = _worker_optimize_offset_540D_nonplanar_( xstart, variables, matchDet, L0 )
 
    #print('--> ox:', x['ox'])
    #print('--> oy:', x['oy'])
    #print('--> rz:', x['rz'])
    
    params  = {'geo':'nonplanar', 'opt':False, 'mode':'virtual', 'hexa': range(24), 'x': x }

    annotation = {'annotation': annot, 'matchdet': matchDet, 'track': False}
    
    return x, params, annotation


####################################################
#
#
# OPTIMIZATION SCHEME FOR 540D ALIGNMENT (PLANAR)
#
#
###################################################

def get_tracking_points_540D_planar(match, Ox, Oy, Gaps):

    #
    # match = [ix_P1, iy_P1, module_P1, stripe_P1, ystripe_P1, ix_P2, iy_P2, module_P2, stripe_P2, ystripe_P2, distance_type, distance ]
    # 

    H              = 1536
    J              = 256
    ASIC_BUMP_BOND = 3

    def get_index_withgap(start, end ):
        gapchip = ASIC_BUMP_BOND
        arr = []
        for k in range(6):
            start_ = end - 256 * (6-k) - (5 - k) * gapchip
            chip   = numpy.arange(start_,start_ + 256)
            arr.append( chip )    
        return numpy.array(arr).flatten()


    def get_shape_pi540D_planar(N,ox, oy, g):

        cr = N//2
        cc = N//2
        
        _row_ = []
        _col_ = []
    
        for k in range(6):
            start_0 = cr - (k+1)*J - g[0][0:k].sum() - oy[0][k]
            end_0   = start_0 + J
            
            end_1   = cc + (k+1)*J + g[1][0:k].sum() + oy[1][k]
            start_1 = end_1 - J
            
            end_2   = cr + (k+1)*J + g[2][0:k].sum() + oy[2][k]
            start_2 = end_2 - J
            
            start_3 = cc - (k+1)*J - g[3][0:k].sum() - oy[3][k]
            end_3   = start_3 + J
    
            if k==5:
                _row_.append([start_0,end_2])
                _col_.append([end_1,start_3])

        _row_ = numpy.array(_row_).flatten()
        _col_ = numpy.array(_col_).flatten()
    
        nrows = max(_row_) - min(_row_)
        ncols = max(_col_) - min(_col_)

        rowv = [ min(_row_), max(_row_) ]
        colv = [ min(_col_), max(_col_) ]
    
        return (nrows,ncols), rowv, colv

    N = 4000 #adjust: TODO
    
    shape, rowv, colv = get_shape_pi540D_planar(N, Ox, Oy, Gaps)

    cr = N//2
    cc = N//2

    numpoints = match.shape[0]
    vmatch    = numpy.zeros([numpoints, 11] ) #, dtype=numpy.int)
    vmatchd   = numpy.zeros([numpoints, 11] )

    def get_ix_iy( module, k, x, y, cr, cc, g, ox, oy):
        if module == 0:
            start_0 = cr - (k+1)*J - g[0][0:k].sum() - oy[0][k]
            end_0   = start_0 + J
            _ix_     = get_index_withgap(cc + ox[0][k] - H, cc + ox[0][k])
            _iy_     = numpy.arange( start_0, end_0)
            ix, iy   = numpy.meshgrid(_ix_, _iy_)
            ix       = ix[J - 1 - y,x] 
            iy       = iy[J - 1 - y,x]
            
        elif module==1:
            end_1   = cc + (k+1)*J + g[1][0:k].sum() + oy[1][k]
            start_1 = end_1 - J
            _ix_    = numpy.arange( start_1, end_1)
            _iy_    = get_index_withgap(cr + ox[1][k] - H, cr + ox[1][k])
            iy,ix   = numpy.meshgrid(_iy_, _ix_)
            iy      = numpy.flipud( iy )[J - 1 - y, x]
            ix      = numpy.flipud( ix )[J - 1 - y, x] 
            
        elif module==2:
            end_2   = cr + (k+1)*J + g[2][0:k].sum() + oy[2][k]
            start_2 = end_2 - J
            _ix_    = get_index_withgap(cc - ox[2][k], cc - ox[2][k] + H)
            _iy_    = numpy.arange( start_2, end_2 )
            ix,iy   = numpy.meshgrid(_ix_, _iy_)
            iy      = numpy.fliplr( numpy.flipud(iy))[J - 1 - y,x]
            ix      = numpy.fliplr( numpy.flipud(ix))[J - 1 - y,x]                 
        else:
            start_3 = cc - (k+1)*J - g[3][0:k].sum() - oy[3][k]
            end_3   = start_3 + J
            _ix_    = numpy.arange( start_3, end_3 ) 
            _iy_    = get_index_withgap( cc - ox[3][k], cc - ox[3][k] + H)
            iy,ix   = numpy.meshgrid(_iy_, _ix_)
            iy      = numpy.flipud( numpy.fliplr( numpy.flipud(iy)))[J - 1 - y,x]
            ix      = numpy.flipud( numpy.fliplr( numpy.flipud(ix)))[J - 1 - y,x] 

        ix = ix - colv[0]
        iy = iy - rowv[0]
            
        return ix, iy
            

    dx = 1
    dy = 1
    
    for p in range(numpoints):

        module0 = int(match[p][2])
        k0      = int(match[p][3])
        xm0     = int(match[p][0])
        ym0     = int(match[p][4])

        module1 = int(match[p][7])
        k1      = int(match[p][8])
        xm1     = int(match[p][5])
        ym1     = int(match[p][9])
        
        ix0, iy0 = get_ix_iy( module0, k0, xm0, ym0, cr, cc, Gaps, Ox, Oy)

        ix1, iy1 = get_ix_iy( module1, k1, xm1, ym1, cr, cc, Gaps, Ox, Oy)  
            
        euclidean = numpy.abs( numpy.sqrt( (ix1-ix0)**2 + (iy1-iy0)**2 ) - match[p][11] )

        vmatch[p] = [ix0, iy0, ix1, iy1, match[p][10], match[p][11], dx, dy,
                     numpy.abs(ix0-ix1), numpy.abs(iy0-iy1), euclidean ]


    print(shape)

    hor = ( match[:,10] == HORIZONTAL )
    ver = ( match[:,10] == VERTICAL )
    euc = ( match[:,10] == EUCLIDEAN )

    fe = vmatch[:, 10][ euc ]
    fx = vmatch[:, 8][ ver ]
    fy = vmatch[:, 9][ hor ]

    
    x1 = vmatch[:, 0]
    x2 = vmatch[:, 2]
    y1 = vmatch[:, 1]
    y2 = vmatch[:, 3]
    
    x1_h = x1[ hor ]
    x2_h = x2[ hor ]

    y1_v = y1[ ver ]
    y2_v = y2[ ver ]

    x1_e = x1[ euc ]
    x2_e = x2[ euc ]

    y1_e = y1[ euc ]
    y2_e = y2[ euc ]

    dist = match[ euc ]
    
    
    '''
    print('--- x1_h')
    print(x1_h)

    print('--- x2_h')
    print(x2_h)

    print('--- y1_v')
    print(y1_v)

    print('--- y2_v')
    print(y2_v)

    print('--- x1_e')
    print(x1_e)

    print('--- y1_e')
    print(y1_e)

    print('--- x2_e')
    print(x2_e)

    print('--- y2_e')
    print(y2_e)
    '''


    fe = numpy.abs( numpy.sqrt( (x2_e - x1_e)**2 + (y2_e - y1_e)**2 ) -  dist[:,11] ) #/ numpy.sqrt(dx**2 + dy**2)
    fx = numpy.abs(x2_h - x1_h) #/ dx
    fy = numpy.abs(y2_v - y1_v) #/ dy

    #print('--- fe')
    #print(fe)
    #print('--- fx')
    #print(fx)
    #print('--- fy')
    #print(fy)
    
    if fe.shape[0] == 0:
        fe = - numpy.ones([1,])
    if fx.shape[0] == 0:
        fx = - numpy.ones([1,])
    if fy.shape[0] == 0:
        fy = - numpy.ones([1,])
    
    vec = numpy.hstack( (fe, fx, fy ) )
    
    #print(fx)
    #print(fy)
    #print(fe)
    
    fobj = vec[ ~ numpy.all(vec == -1, axis=0) ].flatten()

    #print(x2_h)
    #print(x1_h)
    #print(fx)
    #print(fobj)
    
    return fobj, vmatch 
 
 
def tracking540D_planar( params, match, *args):
   
    Ox = numpy.array( [ params['ox'][0], params['ox'][1], params['ox'][2], params['ox'][3] ] )
    Oy = numpy.array( [ params['oy'][0], params['oy'][1], params['oy'][2], params['oy'][3] ] )
    Gaps  = numpy.array( [ params['gaps'][0], params['gaps'][1], params['gaps'][2], params['gaps'][3] ]  )
    
    fobj, vmatch = get_tracking_points_540D_planar(match , Ox, Oy, Gaps)
    
    return fobj, vmatch


def _criteria_offset_540D_planar_( x, *args ):

    startEvalC = time.time()
    
    params = args[0]
    
    var       = params[0]
    x0        = params[1]
    tracking  = params[2]
    
    y          = x0
    y[var > 0] = x
    
    ydic = set_optimization_variable ( y )
    
    dic = { 'gaps': ydic['gaps'],
            'oy':   ydic['oy'],
            'ox':   ydic['ox'],
            'rz':   ydic['rz']
          }

    #print(dic['ox'])
    
    paramsp  = {'geo':'planar','opt':True, 'mode':'real', 'hexa': range(24), 'x': dic }
    project = get_detector_dictionary( 0, paramsp )
    
    crit, vmatch = tracking540D_planar( project, tracking )

    objf = 0.5 * (crit**2).sum()
    
    #print('--> objective function:', crit)
    print('--> objective function:', objf )
    #print('y:', y)

    return objf

def _worker_optimize_offset_540D_planar(xstart, variables, tracking):    

    ovar = optimization_variables ( variables ) 
    
    x0 = get_optimization_variable(xstart)
    
    params = (ovar, x0, tracking, ) 
    
    x0_  = x0[ovar > 0]

    #if len(args) > 0:
    #    tolerance = args[0]
    #    bnds = get_project_bounds_geometry( ovar, (x0, tolerance) )
    #else:
    #    bnds = get_project_bounds_geometry( ovar, (x0, ) )
        
    res = minimize( _criteria_offset_540D_planar_, x0_, args=(params,), method='Nelder-Mead') # options={'maxiter':4000}, method="Nelder-Mead", tol=1e-8)
        
    x = x0
    x[ ovar > 0] = res.x

    out = set_optimization_variable ( x )
    
    return out

def optimize_540D_planar( piannofiles ):

    #optimization variable!
    variables = {
        'a': 4*[0],
        'rx': 24*[0], 
        'ry': 24*[0],
        'rz': 24*[0],
        'offset': 24*[0],
        'ox': 24*[1], 
        'oy':  24*[1],
        'normal': 3*[0],
        'center': 2*[0],
        'gaps': 24*[1],
        'z': 0
    }

    matchDet = numpy.array([]).reshape([0,14])
    annot = []
    for m in range(len(piannofiles)):

        with open(piannofiles[m], 'rb') as f:
            pianno = pickle.load(f)

        x = {k:v for k,v in pianno.items() if type(k) is not str}

        _annot_ = []
        for k in range(len(x)):
            _a_ = 6*[0]
            if k in x.keys():
                _a_[0:2] = numpy.flipud( x[k][0][1:3] )
                _a_[2:4] = numpy.flipud( x[k][1][1:3] )
                if x[k][3]=='vertical':
                    _a_[4] = 'y'
                    _a_[5] = -1
                elif x[k][3]=='horizontal':
                    _a_[4] = 'x'
                    _a_[5] = -1
                else:
                    _a_[4] = 'e'
                    _a_[5] = x[k][2]
                _annot_.append(_a_)

        #points matching at the detector space    
        _matchDet_, _ = annotation_points( _annot_ )
     
        annot.append( _annot_ )
        matchDet = numpy.vstack( (matchDet, _matchDet_) )

    #tracking of all annotated points on the restored image
    track = annotation_image_points( matchDet )
     
    xstart = set_optimization_variable( get_project_values_geometry( {'geo':'planar','opt':False, 'mode':'real', 'hexa': range(24)}) )

    x = _worker_optimize_offset_540D_planar( xstart, variables, matchDet )
 
    #print('--> ox:', x['ox'])
    #print('--> oy:', x['oy'])
    #print('--> rz:', x['rz'])

    dic = { 'gaps': x['gaps'],
            'oy':   x['oy'],
            'ox':   x['ox'],
            'rz':   x['rz']
           }
    
    params  = {'geo':'planar','opt':False, 'mode':'real', 'hexa': range(24), 'x': dic }

    annotation = {'annotation': annot, 'matchdet': matchDet, 'track': track}

    return x, params, annotation


def see_annotations_540D_planar( img, piannofiles ):

    #optimization variable!
    variables = {
        'a': 4*[0],
        'rx': 24*[0], 
        'ry': 24*[0],
        'rz': 24*[0],
        'offset': 24*[0],
        'ox': 24*[1], 
        'oy':  24*[1],
        'normal': 3*[0],
        'center': 2*[0],
        'gaps': 24*[1],
        'z': 0
    }

    matchDet = numpy.array([]).reshape([0,14])
    annot = []
    for m in range(len(piannofiles)):

        with open(piannofiles[m], 'rb') as f:
            pianno = pickle.load(f)

        x = {k:v for k,v in pianno.items() if type(k) is not str}

        _annot_ = []
        for k in range(len(x)):
            _a_ = 6*[0]
            if k in x.keys():
                _a_[0:2] = numpy.flipud( x[k][0][1:3] )
                _a_[2:4] = numpy.flipud( x[k][1][1:3] )
                if x[k][3]=='vertical':
                    _a_[4] = 'y'
                    _a_[5] = -1
                elif x[k][3]=='horizontal':
                    _a_[4] = 'x'
                    _a_[5] = -1
                else:
                    _a_[4] = 'e'
                    _a_[5] = x[k][2]
                _annot_.append(_a_)

        #points matching at the detector space    
        _matchDet_, _ = annotation_points( _annot_ )
     
        annot.append( _annot_ )
        matchDet = numpy.vstack( (matchDet, _matchDet_) )

    #tracking of all annotated points on the restored image
    track = annotation_image_points( matchDet )
        
    params  = {'geo':'planar','opt':False, 'mode':'real', 'hexa': range(24), 'x': dic }

    annotation = {'annotation': annot, 'matchdet': matchDet, 'track': track}

    ## image with annotations
    
    annot = annotation['annotation']
    track = annotation['track']
    
    imga = _worker_annotation_image(img)

    imgo = numpy.clip( numpy.copy(img),0, img.max())
    imga = numpy.clip(imga,0,img.max())

    A = numpy.zeros(imgo.shape)
    for m in range(len(annot)):
        for k in range(len(annot[m])):
            A[annot[m][k][1], annot[m][k][0] ] = 1 # 2* img.max()
            A[annot[m][k][3], annot[m][k][2] ] = 1 # 2* img.max()

    A = gaussian_filter(A, sigma=2) 
    track = gaussian_filter(track, sigma=2) 
    
    imga[ A > 0 ] = 2 * img.max()
    imgo[ track > 0 ] = 2 * img.max()
         
    return imga, imgo



    
######################################################
#
#
#
#
# MAPPING FUNCTIONS
#
#
#
#
#####################################################
    

def mapping540D(x,y, project ):

    """ Function to map a given pixel lying at the pimega/pi540D to the
        virtual imaging plane
    
    Args:
        x: (int) column
        y: (int) row
        project: pi540D input parameters

    Returns:
        rx: (int) restored x pixel
        ry: (int) restored y pixel
    """

    
    _annot_ = numpy.array([ [x, y] ])
    tracking = annotation_points_standard ( _annot_ )
    tracking = tracking540D_vec_standard ( project, tracking ) 

    rx = int( tracking[0][2] )
    ry = int( tracking[0][3] )

    return rx,ry

