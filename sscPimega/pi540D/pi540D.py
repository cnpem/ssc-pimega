import os
import sys
import ctypes
import numpy
import time
import gc
import h5py
import math

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

from ..pi135D import *
from ..pimegatypes import *


from scipy.optimize import minimize
from scipy import interpolate
from skimage.feature import canny
from skimage.morphology import convex_hull_image

import pickle

MEDIPIX  = 55.0

############


def interpolate_missing_pixels(
        image: numpy.ndarray,
        mask: numpy.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    #:param image: a 2D image
    #:param mask: a 2D boolean image, True indicates missing values
    #:param method: interpolation method, one of
    #    'nearest', 'linear', 'cubic'.
    #:param fill_value: which value to use for filling up data outside the
    #    convex hull of known pixel values.
    #    Default is 0, Has no effect for 'nearest'.
    #:return: the image with missing values interpolated
    
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = numpy.meshgrid(numpy.arange(w), numpy.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def computeMI(x, y, bins):

    def mutual_information(hgram):
        """ Mutual information for joint histogram"""
        # Convert bins counts to probability values
        pxy = hgram / float(numpy.sum(hgram))
        px = numpy.sum(pxy, axis=1) # marginal for x over y
        py = numpy.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return numpy.sum(pxy[nzs] * numpy.log(pxy[nzs] / px_py[nzs]))

    hist_2d, x_edges, y_edges = numpy.histogram2d( x.ravel(), y.ravel(), bins = bins )
    
    return mutual_information ( hist_2d )

    
def find_center_position_at_stripe( center ):
    #center = [x,y] : pixel unities starting at (0,0) from the center
    J = 256
    
    if center[0] * center[1] > 0:
        
        if center[0] > 0:
            mod = 1
            stripe = abs(center[0]) //  J
            index = abs(center[0]) % J
        else:
            mod = 3
            stripe = abs(center[0]) //  J
            index = abs(center[0]) %  J
    else:
        if center[0] > 0:
            mod = 2
            stripe = abs(center[1]) //  J
            index = abs(center[1]) %  J
        else:
            mod = 0
            stripe = abs(center[1]) //  J
            index  = abs(center[1]) %  J
            
    return mod,stripe,index


def build_mesh_540D( m, t):

    mesh = []
    
    #module 0
    mm = m - m[1535]  
    tt = t 
    mesh.append( [ mm, tt ] )

    #module 1
    mm = m - m[1535] 
    tt = t 
    mesh.append( [ mm, tt ] )

    #module 2
    mm = m - m[1535]
    mesh.append( [ mm, tt ] )

    #module 3
    mm = m - m[1535] 
    tt = t
    mesh.append( [ mm, tt ] )

    return mesh
    

def get_module_strip (s):

    return s // 6,  s % 6
    
def set_module_strip (module, j):

    return j + 6 * module

def get_image_stripe_detector( img, m, s):
    
    s = 5 - s
    
    if m==0:
        out = img[s*256:(s+1)*256,0:1536]
    elif m==1:
        out= img[s*256:(s+1)*256,1536:3072]
    elif m==2:
        out = img[s*256+1536:(s+1)*256+1536,1536:3072]
    elif m==3:
        out = img[s*256+1536:(s+1)*256+1536,0:1536]        

    return out

def set_image_stripe_detector( out, img, m, s):
    
    s = 5 - s
    
    if m==0:
        out[s*256:(s+1)*256,0:1536] = img
    elif m==1:
        out[s*256:(s+1)*256,1536:3072] = img
    elif m==2:
        out[s*256+1536:(s+1)*256+1536,1536:3072] = img
    elif m==3:
        out[s*256+1536:(s+1)*256+1536,0:1536] = img       

    return out


def set_suspicious_pixels_540D( frame, epsilon):

    SUSP540D = -10
    
    modules = [ frame[0:1536,0:1536],
                frame[0:1536,1536:3072],
                frame[1536:3072,1536:3072],
                frame[1536:3072,0:1536]]
    
    modules[0] = set_suspicious_pixels( modules[0], 256, 6, 6, epsilon)
    modules[1] = set_suspicious_pixels( modules[1], 256, 6, 6, epsilon)
    modules[2] = set_suspicious_pixels( modules[2], 256, 6, 6, epsilon)
    modules[3] = set_suspicious_pixels( modules[3], 256, 6, 6, epsilon)
    
    top    = numpy.hstack(( modules[0], modules[1] ))
    bottom = numpy.hstack(( modules[3], modules[2] )) 
    new    = numpy.vstack(( top, bottom ))
    
    new[ new < 0 ] = SUSP540D
        
    return new
    

def set_suspicious_pixels_540D_block( frame, epsilon):

    SUSP540D = -10
    
    modules = [ frame[:,0:1536,0:1536],
                frame[:,0:1536,1536:3072],
                frame[:,1536:3072,1536:3072],
                frame[:,1536:3072,0:1536]]
    
    modules[0] = set_suspicious_pixels_block( modules[0], 256, 6, 6, epsilon)
    modules[1] = set_suspicious_pixels_block( modules[1], 256, 6, 6, epsilon)
    modules[2] = set_suspicious_pixels_block( modules[2], 256, 6, 6, epsilon)
    modules[3] = set_suspicious_pixels_block( modules[3], 256, 6, 6, epsilon)
    
    top    = numpy.hstack(( modules[0], modules[1] ))
    bottom = numpy.hstack(( modules[3], modules[2] )) 
    new    = numpy.vstack(( top, bottom ))
    
    new[ new < 0 ] = SUSP540D
        
    return new



def projection_virtual_det(m,t,n,nperp,ncross,Lj,L,e3,Nz):
    #point over stripe
    p0 = Nz[0] - Lj * n[0] + t * ncross[0] + m * nperp[0]
    p1 = Nz[1] - Lj * n[1] + t * ncross[1] + m * nperp[1]
    p2 = Nz[1] - Lj * n[2] + t * ncross[2] + m * nperp[2]

    den =  ( (Nz * e3).sum() - Lj * ((n * e3).sum()) + t * ((ncross*e3).sum()) + m * ((nperp*e3).sum()) )

    #projected points: grid 
    const = (-L/den)
    x = p0 * const
    y = p1 * const
    z = p2 * const
    
    return x,y,z

def pointcloud_det(m,t,n,nperp,ncross,Lj,L, e3,Nz):
    #point over stripe
    
    p0 = Nz[0] - Lj * n[0] + t * ncross[0] + m * nperp[0]
    p1 = Nz[1] - Lj * n[1] + t * ncross[1] + m * nperp[1]
    p2 = Nz[2] - Lj * n[2] + t * ncross[2] + m * nperp[2]

    p0_ = p0.flatten()
    p1_ = p1.flatten()
    p2_ = p2.flatten()

    return p0_,p1_,p2_

    #return x.flatten(), y.flatten(), z.flatten()

def get_bounding_box(normal, RX, RY, RZ, L, dL, a, center, typedet):

    J          = 256
    P          = 6
    M          = 6
    NOSTRIPES  = 24
    VL         = 2 * J * M
    HL         = 2 * J * P
    
    xpoints_bb = []
    ypoints_bb = []

    dL           = numpy.array( dL ).flatten() * 1000 #(microns)
    e3           = numpy.array( build_normal( normal )  ).reshape([3,1]) 
    L            = (numpy.array( 4 * [L] ) ) * 1000 #(microns)
    D            = parameters(J, P, M, False, typedet)
    mesh         = build_mesh_540D( D['m_'], D['t_'] )
    angles       = [0, -numpy.pi/2, -numpy.pi, -3*numpy.pi/2.]
    delta        = D['delta']
    
    z = numpy.zeros([3,])
    
    _mod_, _stripe_, _istripe_ = find_center_position_at_stripe(center)
    
    ccenter = []

    bbox = numpy.zeros([NOSTRIPES,8])
    
    stripes_to_compute_bounding_box = set_module_strip( numpy.array([0,1,2,3]), numpy.array([5,5,5,5]) ).astype(int)
    for s in stripes_to_compute_bounding_box:
    #for s in range(NOSTRIPES):
        
        module, j = get_module_strip(s)
        
        m_module = mesh[module][0]
        t_module = mesh[module][1]
        
        stripes = []
        
        ang = angles[module] - a[module] * numpy.pi/180.
        
        RZ_module_array = numpy.array([[numpy.cos(ang), -1*numpy.sin(ang), 0],[numpy.sin(ang),numpy.cos(ang),0],[0,0,1]])

        # using (0,0,1) as reference
        n = numpy.array([0,0,1]).reshape([3,1])
        #n = numpy.copy(e3)
        
        RX_ = ( (RX[module][j] ) * numpy.pi/180.)
        RX_array = numpy.array([[1,0,0],[0,numpy.cos(RX_), -1*numpy.sin(RX_)],[0,numpy.sin(RX_),numpy.cos(RX_)]])
        
        RZ_ = ( (RZ[module][j] ) * numpy.pi/180.)
        RZ_array = numpy.array([[numpy.cos(RZ_), -1*numpy.sin(RZ_), 0],[numpy.sin(RZ_),numpy.cos(RZ_), 0],[0,0,1]])
        
        RY_ = ( (RY[module][j] ) * numpy.pi/180.)
        RY_array = numpy.array([[numpy.cos(RY_), 0, numpy.sin(RY_)],[0,1,0],[-numpy.sin(RY_),0,numpy.cos(RY_)]])
        
        n = numpy.dot(RX_array,n)
        n = numpy.dot(RY_array,n)
    
        nperp  = numpy.array([[n[2,0]],[0],[-n[0,0]]])
        ncross = numpy.array([[ -n[1,0]*n[0,0]],[ n[2,0]**2+n[0,0]**2],[-n[1,0]*n[2,0]]])

        n      = numpy.dot(RZ_array,n)    
        ncross = numpy.dot(RZ_array,ncross)    
        nperp  = numpy.dot(RZ_array,nperp)    

        #module rotation        
        start = time.time()
        
        n = numpy.dot(RZ_module_array,n)
        nperp = numpy.dot(RZ_module_array,nperp)
        ncross = numpy.dot(RZ_module_array,ncross)
        
        Lj = ( L[module]  * numpy.cos( numpy.abs(RX_) ) - j * delta ) + dL[s]
        
        dpnz = n[0] * z[0] + n[1] * z[1] + n[2] * z[2]  
        Nz = numpy.array([ n[0] * dpnz , n[1] * dpnz , n[2] * dpnz ]).reshape([3,1])

        ##
        #there is no module translation - bounding box according to origin (image center)
        sx    = z[0]
        sy    = z[1]
        Gz    = [ (1-n[0]*n[0])*z[0] - n[0]*n[1]*z[1] - n[0]*n[2]*z[2], -n[1]*n[0]*z[0] + (1-n[1]*n[1])*z[1] - n[1]*n[2]*z[2] ] 
        px    = Gz[0] - Lj * n[0]
        py    = Gz[1] - Lj * n[1]
        det   = ncross[0] * nperp[1] - ncross[1] * nperp[0] 
        tref  = 1*(( nperp[1] * ( sx - px )  - nperp[0] * (sy - py) ) / det)
        mref  = 1*((-ncross[1] * ( sx - px ) + ncross[0] * (sy - py) ) / det)
        
        m = m_module + mref
        t = t_module[j*J: min( J*P,(j+1)*J )] + tref
        
        #corner: top/left ======
        x_tl, y_tl, _ = projection_virtual_det( m[0],t[len(t)-1],n,nperp,ncross,Lj,L[module],e3,Nz)
        xpoints_bb.append(x_tl)
        ypoints_bb.append(y_tl)
        
        #corner: bottom/left ======
        x_bl, y_bl, _ = projection_virtual_det( m[0],t[0],n,nperp,ncross,Lj,L[module],e3,Nz)
        xpoints_bb.append(x_bl)
        ypoints_bb.append(y_bl)
        
        #corner: bottom/right =====
        x_br, y_br, _ = projection_virtual_det( m[len(m)-1],t[0],n,nperp,ncross,Lj,L[module],e3,Nz)
        xpoints_bb.append(x_br)
        ypoints_bb.append(y_br)
        
        #corner: top/right =====
        x_tr, y_tr, _ = projection_virtual_det( m[len(m)-1],t[len(t)-1],n,nperp,ncross,Lj,L[module],e3,Nz)
        xpoints_bb.append(x_tr)
        ypoints_bb.append(y_tr)

        bbox[s,:] = numpy.array( [ x_tl, x_bl, x_tr, x_br, y_tl, y_bl, y_tr, y_br ] ).reshape([8,])  
        

    #------------------------
    #find global bounding box
        
    minx = min( xpoints_bb ) 
    maxx = max( xpoints_bb ) 

    miny = min( ypoints_bb ) 
    maxy = max( ypoints_bb ) 
    
    xbox0 = [ float(minx), float(maxx), float(maxx)-float(minx) ] 
    ybox0 = [ float(miny), float(maxy), float(maxy)-float(miny) ] 

    ##
    
    mx = (xbox0[0]+xbox0[1])/2.0
    my = (ybox0[0]+ybox0[1])/2.0
    sx = mx #theoretical center at box
    sy = my #theoretical center at box
    
    ## bounding boxes per module
    #bbox_  =  4 * [ [ xmin, xmax, ymin, ymax ]  ] 
   
    bbox_ = [ [ min(min(bbox[0][0:2]), min(bbox[5][0:2])), max(max(bbox[0][2:4]), max(bbox[5][2:4])),  min(min(bbox[0][4:6]), min(bbox[5][4:6])), max(max(bbox[0][6:8]), max(bbox[5][6:8])), None ],
              [ min(min(bbox[6][0:2]), min(bbox[11][0:2])), max(max(bbox[6][2:4]), max(bbox[11][2:4])),  min(min(bbox[6][4:6]), min(bbox[11][4:6])), max(max(bbox[6][6:8]), max(bbox[11][6:8])), None ],
              [ min(min(bbox[12][0:2]), min(bbox[17][0:2])), max(max(bbox[12][2:4]), max(bbox[17][2:4])),  min(min(bbox[12][4:6]), min(bbox[17][4:6])), max(max(bbox[12][6:8]), max(bbox[17][6:8])), None ],
              [ min(min(bbox[18][0:2]), min(bbox[23][0:2])), max(max(bbox[18][2:4]), max(bbox[23][2:4])),  min(min(bbox[18][4:6]), min(bbox[23][4:6])), max(max(bbox[18][6:8]), max(bbox[23][6:8])), None ]
             ]

    for o in range(4):
        bbox_[o] = [ sorted( bbox_[o][0:2] ), sorted(bbox_[o][2:4]), None ]

     
    bbox_[0][2] = [ bbox_[0][0][0] ]
    bbox_[1][2] = [ bbox_[1][1][1] ]
    bbox_[2][2] = [ bbox_[2][0][1] ]
    bbox_[3][2] = [ bbox_[3][1][0] ]

    ##
    
    dx = ( float(xbox0[1]) - float(xbox0[0]) ) / (HL - 1 )
    dy = ( float(ybox0[1]) - float(ybox0[0]) ) / (VL - 1 )
    
    x0 = xbox0[0]
    y0 = ybox0[0] 

    ix = ( numpy.floor( (sx - x0) /(dx) )).astype(int) 
    iy = VL - ( numpy.floor( (sy - y0) /(dy) )).astype(int)   
    
    return {'boxes': bbox_, 'center': center, 'ccenter': [sx, sy], 'xbox': xbox0, 'ybox': ybox0 , 'icenter': [ix,iy] }


def get_bbox540D( params ):
    
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

    ####
    
    boxinfo = get_bounding_box( v, RxM, RyM, RzM, L, dL, a, center, typedet)
    
    return boxinfo

    
def build_mesh_module_540D( s, params ):

    module, j = get_module_strip(s)

    translate = [ [-1,0], [0,+1], [+1, 0], [0,-1]]
    
    #------------------------------------
    #paramsModule = (D, mesh, angles, e3, RX, RY, RZ, L, plot)
    
    D      = params[0]
    mesh   = params[1]
    angles = params[2]
    normal = params[3]
    RX     = params[4]
    RY     = params[5]
    RZ     = params[6]
    L      = params[7]
    plot   = params[8]
    symbol = params[9]
    a      = params[10]
    ox     = params[11] #* 1000 #mm -> microns
    oy     = params[12] #* 1000 #mm -> microns
    boxinfo= params[13]
    dL     = params[14]
    susp   = params[15]
    
    J       = D['J']
    P       = D['P']
    M       = D['M']
    delta   = D['delta']
    colours = D['colours']

    VL = 2 * J * M
    HL = 2 * J * P    

    e3 = numpy.array( build_normal( normal ) ).reshape([3,1])
    m_module = mesh[module][0]
    t_module = mesh[module][1]
    
    bottomleft = numpy.zeros([P,2])
    topright   = numpy.zeros([P,2])
    bottomright= numpy.zeros([P,2])
    topleft    = numpy.zeros([P,2])
    
    stripes = []
    
    ang = angles[module] - a[module] * numpy.pi/180.
    
    RZ_module_array = numpy.array([[numpy.cos(ang), -1*numpy.sin(ang), 0],[numpy.sin(ang),numpy.cos(ang),0],[0,0,1]])
    
    z = numpy.array([ boxinfo['ccenter'][0], boxinfo['ccenter'][1], 0]).reshape([3,1])

    #--
    # using (0,0,1) as reference
    n = numpy.array([0,0,1]).reshape([3,1])
    #n = numpy.copy(e3)
    
    RX_ = ( (RX[module][j] ) * numpy.pi/180.)
    RX_array = numpy.array([[1,0,0],[0,numpy.cos(RX_), -1*numpy.sin(RX_)],[0,numpy.sin(RX_),numpy.cos(RX_)]])
    
    RZ_ = ( (RZ[module][j] ) * numpy.pi/180.)
    RZ_array = numpy.array([[numpy.cos(RZ_), -1*numpy.sin(RZ_), 0],[numpy.sin(RZ_),numpy.cos(RZ_), 0],[0,0,1]])
    
    RY_ = ( (RY[module][j] ) * numpy.pi/180.)
    RY_array = numpy.array([[numpy.cos(RY_), 0, numpy.sin(RY_)],[0,1,0],[-numpy.sin(RY_),0,numpy.cos(RY_)]])
    
    n = numpy.dot(RX_array,n)
    n = numpy.dot(RY_array,n)

    
    nperp  = numpy.array([[n[2,0]],[0],[-n[0,0]]])
    ncross = numpy.array([[ -n[1,0]*n[0,0]],[ n[2,0]**2+n[0,0]**2],[-n[1,0]*n[2,0]]])

    n      = numpy.dot(RZ_array, n)    
    nperp  = numpy.dot(RZ_array, nperp)
    ncross = numpy.dot(RZ_array, ncross)
    
    #module rotation        
    n = numpy.dot(RZ_module_array,n)
    nperp = numpy.dot(RZ_module_array,nperp)
    ncross = numpy.dot(RZ_module_array,ncross)
    
    dpnz = n[0,0] * z[0,0] + n[1,0] * z[1,0] + n[2,0] * z[2,0]  
    Nz   = numpy.array([ n[0,0] * dpnz , n[1,0] * dpnz , n[2,0] * dpnz ]).reshape([3,1])
    
    ###########
                
    Lj = ( L[module]  * numpy.cos( numpy.abs(RX_) ) - j * delta ) + dL[s]

    #discriminant = (L[module]*numpy.sin(2*RX_)+1)**2 - 4*numpy.sin(RX_)*(numpy.sin(RX_)*(L[module]**2) + L[module]*numpy.cos(RX_))
    #print('-----> disc[{}]: {}'.format(j, discriminant))
    
    sx    = boxinfo['ccenter'][0] 
    sy    = boxinfo['ccenter'][1] 
    Gz    = [ (1 - n[0,0]*n[0,0])*z[0,0] - n[0,0]*n[1,0]*z[1,0] - n[0,0]*n[2,0]*z[2,0], -n[1,0]*n[0,0]*z[0,0] + (1-n[1,0]*n[1,0])*z[1,0] - n[1,0]*n[2,0]*z[2,0] ]
    px    = Gz[0] - Lj * n[0,0]
    py    = Gz[1] - Lj * n[1,0]
    det   = ncross[0,0] * nperp[1,0] - ncross[1,0] * nperp[0,0] 
    tref  = 1*(( nperp[1,0] * ( sx - px )  - nperp[0,0] * (sy - py) ) / det)
    mref  = 1*((-ncross[1,0] * ( sx - px ) + ncross[0,0] * (sy - py) ) / det)

    _mod_, _, _ = find_center_position_at_stripe( boxinfo['center'] )
    
    
    _mref_ = [ numpy.array([1,1,-1,-1]) * numpy.abs( numpy.array( [ boxinfo['center'][0], boxinfo['center'][1], boxinfo['center'][0], boxinfo['center'][1] ] ) ), 
               numpy.array([-1,1,1,-1]) * numpy.abs( numpy.array( [ boxinfo['center'][0], boxinfo['center'][1], boxinfo['center'][0], boxinfo['center'][1] ] ) ),
               numpy.array([-1,-1,1,1]) * numpy.abs( numpy.array( [ boxinfo['center'][0], boxinfo['center'][1], boxinfo['center'][0], boxinfo['center'][1] ] ) ),
               numpy.array([1,-1,-1,1]) * numpy.abs( numpy.array( [ boxinfo['center'][0], boxinfo['center'][1], boxinfo['center'][0], boxinfo['center'][1] ] ) )
              ]          
    mref = _mref_[_mod_][module] * MEDIPIX
    
    m = m_module + mref + ox[module][j]
    t = t_module[j*J: min( J*P,(j+1)*J )] + tref + oy[module][j]
     
    M_,T_ = numpy.meshgrid( m, t )   
    
    #grid points over the virtual detector
    xMesh, yMesh, zMesh = projection_virtual_det(M_,T_,n,nperp,ncross,Lj, L[module],e3, Nz)

    #stripe boundary meshes
    M_ht,T_ht = numpy.meshgrid( m, numpy.array([ t[0] ]) )  #horizontal/top
    M_hb,T_hb = numpy.meshgrid( m, numpy.array([ t[len(t)-1] ]) ) #horizontal/bottom

    M_vl, T_vl = numpy.meshgrid( m[0:-1:256],  t ) #vertical/left
    M_vr, T_vr = numpy.meshgrid( m[255:-1:256], t ) #vertical/right

    #gaps

    mgap = []
    for x in range(M-1):
        left = (x+1)*J - 1
        right = min( (x+1)*J, J*M )

        _from_ = left-susp+1
        _to_   = left+1

        for k in range(_from_, _to_):
            mgap.append( m[k] )
        
        _from_ = right
        _to_   = right+susp
    
        for k in range(_from_, _to_):
            mgap.append( m[k] )

    mgap = numpy.array(mgap)

    tgap = []
    for j in range(susp):
        tgap.append( float( t[j] ) )
    
    for j in range(susp):
        tgap.append( float( t[ J-1 - susp + 1 + j ]) )

    tgap = numpy.array( tgap )
    
    M_gap1, T_gap1 = numpy.meshgrid(mgap, t)
    
    M_gap2, T_gap2 = numpy.meshgrid(m, tgap)
    
    # Meshes

    xMesh_ht, yMesh_ht, _ = projection_virtual_det(M_ht,T_ht,n,nperp,ncross,Lj,L[module],e3,Nz)
    xMesh_hb, yMesh_hb, _ = projection_virtual_det(M_hb,T_hb,n,nperp,ncross,Lj,L[module],e3,Nz)
    
    xMesh_vl, yMesh_vl, _ = projection_virtual_det(M_vl,T_vl,n,nperp,ncross,Lj,L[module],e3,Nz)
    xMesh_vr, yMesh_vr, _ = projection_virtual_det(M_vr,T_vr,n,nperp,ncross,Lj,L[module],e3,Nz)

    xMesh_gap1, yMesh_gap1, _ = projection_virtual_det(M_gap1,T_gap1,n,nperp,ncross,Lj,L[module],e3,Nz)
    xMesh_gap2, yMesh_gap2, _ = projection_virtual_det(M_gap2,T_gap2,n,nperp,ncross,Lj,L[module],e3,Nz)

    
    xMesh    += translate[ module ][0] * mref
    yMesh    += translate[ module ][1] * mref
    
    xMesh_ht += translate[ module ][0] * mref
    yMesh_ht += translate[ module ][1] * mref
    
    xMesh_hb += translate[ module ][0] * mref
    yMesh_hb += translate[ module ][1] * mref
    
    xMesh_vl += translate[ module ][0] * mref
    yMesh_vl += translate[ module ][1] * mref

    xMesh_vr += translate[ module ][0] * mref
    yMesh_vr += translate[ module ][1] * mref
    
    ##
    
    if plot:
        #corner: top/left
        x, y, _ = projection_virtual_det( m[0],t[len(t)-1],n,nperp,ncross,Lj,L[module],e3,Nz)
        x += translate[ module ][0] * mref
        y += translate[ module ][1] * mref
        topleft[j,:] = [x,y]
        
        #corner: bottom/left
        x, y, _ = projection_virtual_det( m[0],t[0],n,nperp,ncross,Lj,L[module],e3,Nz)
        x += translate[ module ][0] * mref
        y += translate[ module ][1] * mref
        bottomleft[j,:] = [x,y]
        
        #corner: bottom/right
        x, y, _ = projection_virtual_det( m[len(m)-1],t[0],n,nperp,ncross,Lj,L[module],e3,Nz)
        x += translate[ module ][0] * mref
        y += translate[ module ][1] * mref
        bottomright[j,:] = [x,y]
          
        #corner: top/right
        x, y, _ = projection_virtual_det( m[len(m)-1],t[len(t)-1],n,nperp,ncross,Lj,L[module],e3,Nz)
        x += translate[ module ][0] * mref
        y += translate[ module ][1] * mref
        topright[j,:] = [x,y]

        
        plt.plot(xMesh,yMesh,'{}'.format(colours[j]+symbol) )
        plt.plot(bottomleft[j,0], bottomleft[j,1], 'ko', topright[j,0], topright[j,1], 'ko')
        plt.plot(bottomright[j,0], bottomright[j,1], 'ko', topleft[j,0], topleft[j,1], 'ko')
        plt.xlabel('x')
        plt.ylabel('y')
        
    return xMesh, yMesh, xMesh_ht, yMesh_ht, xMesh_hb, yMesh_hb, xMesh_vl, yMesh_vl, xMesh_vr, yMesh_vr, xMesh_gap1, yMesh_gap1, xMesh_gap2, yMesh_gap2


def build_pointcloud_stripe_540D( s, params ):

    module, j = get_module_strip(s)
    
    #------------------------------------
    #paramsModule = (D, mesh, angles, e3, RX, RY, RZ, L, plot)
    
    D      = params[0]
    mesh   = params[1]
    angles = params[2]
    normal = params[3]
    RX     = params[4]
    RY     = params[5]
    RZ     = params[6]
    L      = params[7]
    plot   = params[8]
    symbol = params[9]
    a      = params[10]
    ox     = params[11] #* 1000 #mm -> microns
    oy     = params[12] #* 1000 #mm -> microns
    boxinfo= params[13] 
    dL     = params[14]
    
    e3 = numpy.array( build_normal( normal ) ).reshape([3,1])
    
    J       = D['J']
    P       = D['P']
    M       = D['M']
    delta   = D['delta']
    colours = D['colours']

    VL = 2 * J * M
    HL = 2 * J * P    
    
    m_module = mesh[module][0]
    t_module = mesh[module][1]
    
    bottomleft = numpy.zeros([P,2])
    topright   = numpy.zeros([P,2])
    bottomright= numpy.zeros([P,2])
    topleft    = numpy.zeros([P,2])
    
    stripes = []
    
    ang = angles[module] - a[module] * numpy.pi/180.
    
    RZ_module_array = numpy.array([[numpy.cos(ang), -1*numpy.sin(ang), 0],[numpy.sin(ang),numpy.cos(ang),0],[0,0,1]])

    z = numpy.array([ boxinfo['ccenter'][0], boxinfo['ccenter'][1], 0]).reshape([3,1])
       
    #---------------
    # using (0,0,1) as reference
    n = numpy.array([0,0,1]).reshape([3,1])
    #n = numpy.copy(e3)
    
    RX_ = ( (RX[module][j] ) * numpy.pi/180.)
    RX_array = numpy.array([[1,0,0],[0,numpy.cos(RX_), -1*numpy.sin(RX_)],[0,numpy.sin(RX_),numpy.cos(RX_)]])
    
    RZ_ = ( (RZ[module][j] ) * numpy.pi/180.)
    RZ_array = numpy.array([[numpy.cos(RZ_), -1*numpy.sin(RZ_), 0],[numpy.sin(RZ_),numpy.cos(RZ_), 0],[0,0,1]])
    
    RY_ = ( (RY[module][j] ) * numpy.pi/180.)
    RY_array = numpy.array([[numpy.cos(RY_), 0, numpy.sin(RY_)],[0,1,0],[-numpy.sin(RY_),0,numpy.cos(RY_)]])
    
    n = numpy.dot(RX_array,n)
    n = numpy.dot(RY_array,n)
        
    nperp  = numpy.array([[n[2,0]],[0],[-n[0,0]]])
    ncross = numpy.array([[ -n[1,0]*n[0,0]],[ n[2,0]**2+n[0,0]**2],[-n[1,0]*n[2,0]]])

    n      = numpy.dot(RZ_array, n)
    ncross = numpy.dot(RZ_array, ncross)
    nperp  = numpy.dot(RZ_array, nperp)
    
    #module rotation        
    n = numpy.dot(RZ_module_array,n)
    nperp = numpy.dot(RZ_module_array,nperp)
    ncross = numpy.dot(RZ_module_array,ncross)
    ###########

    dpnz = n[0] * z[0] + n[1] * z[1] + n[2] * z[2]  
    Nz = numpy.array([ n[0] * dpnz , n[1] * dpnz , n[2] * dpnz ]).reshape([3,1])
    
    Lj = ( L[module]  * numpy.cos( numpy.abs(RX_) ) - j * delta ) + dL[s]
                
    sx    = z[0]
    sy    = z[1]
    Gz    = [(1-n[0]*n[0]+1)*z[0] - n[0]*n[1]*z[1] - n[0]*n[2]*z[2], -n[1]*n[0]*z[0] + (1-n[1]*n[1])*z[1] - n[1]*n[2]*z[2] ] 
    px    = Gz[0] - Lj * n[0]
    py    = Gz[1] - Lj * n[1]
    det   = ncross[0] * nperp[1] - ncross[1] * nperp[0] 
    tref  = 1*(( nperp[1] * ( sx - px )  - nperp[0] * (sy - py) ) / det)
    mref  = 1*((-ncross[1] * ( sx - px ) + ncross[0] * (sy - py) ) / det)
        
    m = m_module + mref + ox[module][j]
    t = t_module[j*J: min( J*P,(j+1)*J )] + tref + oy[module][j]
    
    M_,T_ = numpy.meshgrid( m, t )   
    
    #grid points over the virtual detector
    p0, p1, p2 = pointcloud_det(M_,T_,n,nperp,ncross,Lj, L[module],e3,Nz)
                   
    return p0, p1, p2


def detector_at_virtual_plane_540D(J,P,M, RX, RZ, RY, L, normal, shift, symbol, plot, a, dL, Ox, Oy, boxinfo, typedet, susp ):

    NOSTRIPES = 24

    #boxinfo  = get_bounding_box( normal, RX, RY, RZ, L, dL, a, center )
    
    dL           = numpy.array( dL ).flatten() * 1000 #(microns)
    L            = (numpy.array( 4 * [L] )) * 1000 #(microns)
    D            = parameters(J, P, M, shift, typedet)
    mesh         = build_mesh_540D( D['m_'], D['t_'] )
    angles       = [0, -numpy.pi/2, -numpy.pi, -3*numpy.pi/2.]
    paramsModule = (D, mesh, angles, normal, RX, RY, RZ, L, plot, symbol, a, Ox, Oy, boxinfo, dL, susp )

    stripesModule = []
    
    for s in range(NOSTRIPES):
        
        module, _ = get_module_strip(s) 
        
        xMesh, yMesh, xMesh_ht, yMesh_ht, xMesh_hb, yMesh_hb, xMesh_vl, yMesh_vl, xMesh_vr, yMesh_vr, xMesh_gap1, yMesh_gap1,  xMesh_gap2, yMesh_gap2 = build_mesh_module_540D( s, paramsModule )
        
        stripesModule.append( [ xMesh, yMesh, xMesh_ht, yMesh_ht, xMesh_hb, yMesh_hb, xMesh_vl, yMesh_vl, xMesh_vr, yMesh_vr, xMesh_gap1, yMesh_gap1, xMesh_gap2, yMesh_gap2] )
        
    if plot:
        plt.plot( boxinfo['xbox'][0], boxinfo['ybox'][0], 'sr' )
        plt.plot( boxinfo['xbox'][1], boxinfo['ybox'][1], 'sr' )
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((boxinfo['xbox'][0],boxinfo['ybox'][0]),boxinfo['xbox'][2],boxinfo['ybox'][2], fill=None, alpha=1))        
        
    D['stripes'] = stripesModule
    D['L'] = L
    
    return D


def detector_pointcloud_540D(J,P,M, RX, RZ, RY, L, normal, shift, symbol, plot, a, dL, Ox, Oy, boxinfo, typedet, susp):

    NOSTRIPES = 24
    
    dL           = numpy.array(dL).flatten() * 1000 #(microns)
    L            = (numpy.array( 4 * [L] ) + numpy.array(dL) ) * 1000 #(microns)
    D            = parameters(J, P, M, shift, typedet)
    mesh         = build_mesh_540D( D['m_'], D['t_'] )
    angles       = [0, -numpy.pi/2, -numpy.pi, -3*numpy.pi/2.]
    paramsModule = (D, mesh, angles, normal, RX, RY, RZ, L, plot, symbol, a, Ox, Oy, boxinfo, dL, susp)

    stripesModule = []
    
    for s in range(NOSTRIPES):
        
        module, _ = get_module_strip(s) 
        
        xMesh, yMesh, zMesh = build_pointcloud_stripe_540D( s, paramsModule )
        
        stripesModule.append( [xMesh, yMesh, zMesh] )
    #
            
    D['stripes'] = stripesModule
    D['L'] = L
    
    return D


def pointcloud_540D( params ):
   
    J = 256
    P = 6
    M = 6
        
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
    
    ####

    symbol="*"
    plot=False
    shift=False

    boxinfo = get_bounding_box(v, RxM, RyM, RzM, L, dL, a, center, typedet )
    
    return detector_pointcloud_540D(J,P,M, RxM, RzM, RyM, L, v, shift, symbol, plot, a, dL, Ox, Oy, boxinfo, typedet, susp)


def pointcloud_virtualplane_540D( params ):
   
    J = 256
    P = 6
    M = 6
    
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
    
    ####

    symbol="*"
    plot=True
    shift=False
    
    boxinfo = get_bounding_box(v, RxM, RyM, RzM, L, dL, a, center, typedet )
    
    detector_at_virtual_plane_540D(J,P,M, RxM, RzM, RyM, L, v, shift, symbol, plot, a, dL, Ox, Oy, boxinfo, typedet, susp)

    plt.show()

    return True
    

def build_index_module_540D( s, params ):

    module, j = get_module_strip(s)
    
    P       = params[0]
    stripes = params[1]
    VL      = params[2]
    HL      = params[3]
    xx      = params[4]
    yy      = params[5]
        
    xMesh     = stripes[s][0]
    yMesh     = stripes[s][1]
    xMesh_ht  = stripes[s][2]
    yMesh_ht  = stripes[s][3]
    xMesh_hb  = stripes[s][4]
    yMesh_hb  = stripes[s][5]
    xMesh_vl  = stripes[s][6]
    yMesh_vl  = stripes[s][7]
    xMesh_vr  = stripes[s][8]
    yMesh_vr  = stripes[s][9]
    xMesh_gap1 = stripes[s][10]
    yMesh_gap1 = stripes[s][11]
    xMesh_gap2 = stripes[s][12]
    yMesh_gap2 = stripes[s][13]
    
    #
    #nearest neighbour interp: assuming a regular mesh at the device
    #        
    ix = ( numpy.ceil( (xMesh - xx[0]) /(xx[1]-xx[0]) )).astype(int)
    iy = VL - ( numpy.ceil( (yMesh - yy[0]) /(yy[1]-yy[0]) )).astype(int) 
    
    mix0 = ( ix < 0 )
    mix1 = ( ix >= HL )
    ix[  mix0 ] = 0
    ix[  mix1 ] = HL - 1
    
    miy0 = ( iy < 0 )
    miy1 = ( iy >= VL )
    iy[ miy0 ] = 0
    iy[ miy1 ] = VL - 1
    
    ###

    ix_ht = ( numpy.floor( (xMesh_ht - xx[0]) /(xx[1]-xx[0]) )).astype(int)
    iy_ht = VL - ( numpy.floor( (yMesh_ht - yy[0]) /(yy[1]-yy[0]) )).astype(int)
    
    mix0 = ( ix_ht < 0 )
    mix1 = ( ix_ht >= HL - 1)
    ix_ht[  mix0 ] = 0
    ix_ht[  mix1 ] = HL - 1
    
    miy0 = ( iy_ht < 0 )
    miy1 = ( iy_ht >= VL )
    iy_ht[ miy0 ] = 0
    iy_ht[ miy1 ] = VL - 1
    
    ix_hb = ( numpy.floor( (xMesh_hb - xx[0]) /(xx[1]-xx[0]) )).astype(int)
    iy_hb = VL - ( numpy.floor( (yMesh_hb - yy[0]) /(yy[1]-yy[0]) )).astype(int)
    
    mix0 = ( ix_hb < 0 )
    mix1 = ( ix_hb >= HL - 1)
    ix_hb[  mix0 ] = 0
    ix_hb[  mix1 ] = HL - 1
    
    miy0 = ( iy_hb < 0 )
    miy1 = ( iy_hb >= VL )
    iy_hb[ miy0 ] = 0
    iy_hb[ miy1 ] = VL - 1
    
    ix_vl = ( numpy.floor( (xMesh_vl - xx[0]) /(xx[1]-xx[0]) )).astype(int)
    iy_vl = VL - ( numpy.floor( (yMesh_vl - yy[0]) /(yy[1]-yy[0]) )).astype(int) 
    
    mix0 = ( ix_vl < 0 )
    mix1 = ( ix_vl >= HL - 1)
    ix_vl[  mix0 ] = 0
    ix_vl[  mix1 ] = HL - 1
    
    miy0 = ( iy_vl < 0 )
    miy1 = ( iy_vl >= VL )
    iy_vl[ miy0 ] = 0
    iy_vl[ miy1 ] = VL - 1
    
    ix_vr = ( numpy.floor( (xMesh_vr - xx[0]) /(xx[1]-xx[0]) )).astype(int)
    iy_vr = VL - ( numpy.floor( (yMesh_vr - yy[0]) /(yy[1]-yy[0]) )).astype(int) 
    
    mix0 = ( ix_vr < 0 )
    mix1 = ( ix_vr >= HL - 1)
    ix_vr[  mix0 ] = 0
    ix_vr[  mix1 ] = HL - 1
    
    miy0 = ( iy_vr < 0 )
    miy1 = ( iy_vr >= VL )
    iy_vr[ miy0 ] = 0
    iy_vr[ miy1 ] = VL - 1
    
    ##
    
    iyP1 = iy + 1
    iyM1 = iy - 1
    ixP1 = ix + 1
    ixM1 = ix - 1
    
    iyP1[ iyP1 >= VL -1 ] = VL-1
    iyM1[ iyM1 < 0 ] = 0
    ixP1[ ixP1 >= HL -1 ] = HL-1
    ixM1[ ixM1 < 0 ] = 0

    ##

    ix_gap1 = ( numpy.floor( (xMesh_gap1 - xx[0]) /(xx[1]-xx[0]) )).astype(int)
    iy_gap1 = VL - ( numpy.floor( (yMesh_gap1 - yy[0]) /(yy[1]-yy[0]) )).astype(int) 

    iy_gap1[ iy_gap1 >= VL -1 ] = VL-1
    iy_gap1[ iy_gap1 < 0 ] = 0
    ix_gap1[ ix_gap1 >= HL -1 ] = HL-1
    ix_gap1[ ix_gap1 < 0 ] = 0

    ix_gap2 = ( numpy.floor( (xMesh_gap2 - xx[0]) /(xx[1]-xx[0]) )).astype(int)
    iy_gap2 = VL - ( numpy.floor( (yMesh_gap2 - yy[0]) /(yy[1]-yy[0]) )).astype(int) 

    iy_gap2[ iy_gap2 >= VL -1 ] = VL-1
    iy_gap2[ iy_gap2 < 0 ] = 0
    ix_gap2[ ix_gap2 >= HL -1 ] = HL-1
    ix_gap2[ ix_gap2 < 0 ] = 0

    return iy, ix, iyP1, iyM1, ixP1, ixM1, iy_ht, ix_ht, iy_hb, ix_hb, iy_vl, ix_vl, iy_vr, ix_vr, iy_gap1, ix_gap1, iy_gap2, ix_gap2


def build_geometry_540D(J,P,M,L,normal,Rx,Rz,Ry, cmask, a, z, dL, Ox, Oy, boxinfo, shift, typedet, susp ):

    NOSTRIPES = 24
    
    #start = time.time()

    symbol="*"
    plot=False
    
    D = detector_at_virtual_plane_540D(J,P,M,Rx,Rz,Ry, L, normal, shift, symbol, plot, a, dL, Ox, Oy, boxinfo, typedet, susp )

    #print('Generating mesh/strip: {} sec'.format(round(time.time()-start,3)))
    
    #
    HL = 2 * J * P
    VL = 2 * J * M
    
    #xbox & ybox (microns)
    dx = ((boxinfo['xbox'][1] - boxinfo['xbox'][0])) / (HL - 1 )
    dy = ((boxinfo['ybox'][1] - boxinfo['ybox'][0])) / (VL - 1 )

    #pixelsize @ virtual detector
    step = max( dx, dy )
    
    #millimeters (pixel size)
    SCALE_FACTOR = z * (MEDIPIX / min(dx, dy))

    xx = (SCALE_FACTOR) * numpy.array([ boxinfo['xbox'][0] + k * step for k in range(HL)])
    yy = (SCALE_FACTOR) * numpy.array([ boxinfo['ybox'][0] + k * step for k in range(VL)])

    X, Y = numpy.meshgrid(xx, yy)
    
    stripes    = D['stripes']
    
    paramsModule = (P, stripes, VL, HL, xx, yy )

    #start = time.time()

    interpModule = []

    ixGpu = numpy.zeros([24,256,1536], dtype=numpy.int32)
    iyGpu = numpy.zeros([24,256,1536], dtype=numpy.int32)

    _xmin = numpy.zeros([24,],dtype=numpy.int32)
    _xmax = numpy.zeros([24,],dtype=numpy.int32)
    _ymin = numpy.zeros([24,],dtype=numpy.int32)
    _ymax = numpy.zeros([24,],dtype=numpy.int32)
    
    for s in range(NOSTRIPES):
        
        iy, ix, iyP1, iyM1, ixP1, ixM1, iy_ht, ix_ht, iy_hb, ix_hb, iy_vl, ix_vl, iy_vr, ix_vr, iy_gap1, ix_gap1, iy_gap2, ix_gap2 = build_index_module_540D(s, paramsModule)

        interpModule.append( [iy, ix, iyP1, iyM1, ixP1, ixM1, iy_ht, ix_ht, iy_hb, ix_hb, iy_vl, ix_vl, iy_vr, ix_vr, iy_gap1, ix_gap1, iy_gap2, ix_gap2] )
    
        ixGpu[s][:,:] = ix
        iyGpu[s][:,:] = iy

        _xmin[s] = ix.min()
        _xmax[s] = ix.max()
        _ymin[s] = iy.min()
        _ymax[s] = iy.max()
        
    #print('Generating LUT(index)/strip: {} sec'.format(round(time.time()-start,3)))
    
    ######

    if cmask:
        mask = 0
        for k in range(NOSTRIPES):
            
            iy, ix, iyP1, iyM1, ixP1, ixM1, iy_ht, ix_ht, iy_hb, ix_hb, iy_vl, ix_vl, iy_vr, ix_vr = interpModule[k]
            z  = numpy.zeros([VL, HL])
            z[iy, ix] = 1
            z[iyP1, ix] = 1
            z[iyM1, ix] = 1
            z[iy, ixP1] = 1
            z[iy, ixM1] = 1
            mask += z
            

            '''
            iy, ix, iyP1, iyM1, ixP1, ixM1, iy_ht, ix_ht, iy_hb, ix_hb, iy_vl, ix_vl, iy_vr, ix_vr = interpModule[k]
            z  = numpy.zeros([VL, HL])
            #z[iy,ix] = 1
            z[iy_ht, ix_ht] = 1
            z[iy_hb, ix_hb] = 1

            z[iy_vl, ix_vl] = 1
            z[iy_vr, ix_vr] = 1
            
            mask += z
            '''
            
            
        start = time.time()
        mask = (mask == 2).astype(numpy.double)
        struct = ndimage.generate_binary_structure(2,1)
        mask = ndimage.binary_dilation(mask, structure=struct).astype(mask.dtype)
        print('Dilation: {} sec'.format(round(time.time()-start,3)))
        
    else:
        mask = numpy.zeros([VL, HL])    
        
    ######
    
    geometry = {
        'interp': interpModule,
        'LUT': [ixGpu, iyGpu],
        'borders': {'xmin': _xmin, 'xmax':_xmax,'ymin':_ymin, 'ymax': _ymax},
        'geom': D,
        'pxlsize': (xx[1] - xx[0]), #microns,
        'Nx': HL,
        'Ny': VL,
        'overlap': mask,
        'P': 6,
        'M': 6,
        'J': 256
         }
    
    return geometry


def correct_image_forward_540D(frame, geometry):
    
    P = geometry['geom']['P']
    M = geometry['geom']['M']
    J = 256
    
    images = []
    olap   = [] 
    
    overlap = geometry['overlap']
    
    for module in range(4):
        
        imModule = numpy.zeros( [J*P, J*M])
        overlapModule = numpy.zeros( [J*P, J*M ])
        
        for j in range(P):

            s = set_module_strip( module, j)
            
            iy, ix, iyP1, iyM1, ixP1, ixM1, _, _, _, _, _, _, _, _, _, _, _, _ = geometry['interp'][s]
            
            nearest = frame[ iy, ix ]
   
            nearest_gap = overlap[iy, ix]
            #-------------------------------------------------
            # assuming that overlap will not exceed half strip
            nearest_gap[0:J//2,:] = 0
            #-------------------------------------------------
            
            imModule[j*J: j*J + J, :] = nearest
            
            overlapModule[j*J: j*J + J, :] = nearest_gap
            
        images.append( numpy.flipud( imModule )  )
        olap.append( numpy.flipud( overlapModule )  )
        
    top    = numpy.hstack(( images[0], images[1] ))
    bottom = numpy.hstack(( images[3], images[2] )) 
    new    = numpy.vstack(( top, bottom ))

    top        = numpy.hstack(( olap[0], olap[1] ))
    bottom     = numpy.hstack(( olap[3], olap[2] )) 
    newOverlap = numpy.vstack(( top, bottom ))
    
    new [ newOverlap == 1 ] = -1
    
    return new 


def correct_image_backward_540D(frame, geometry, *args):

    SUSP540D = -10
    
    if not args:
        do_rot = False
    else:
        do_rot = args[0]
        
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

    def _rotation3D_( img, a):
        if a == 90:
            o = numpy.transpose( numpy.flip( img, axis=1), [0,2,1] ) 
        elif a == 180:
            o = numpy.transpose( numpy.flip( img, axis=1), [0,2,1] )
            o = numpy.transpose( numpy.flip( o, axis=1), [0,2,1] )
        elif a == 270:
            o = numpy.transpose( numpy.flip( img, axis=1), [0,2,1] )
            o = numpy.transpose( numpy.flip( o, axis=1), [0,2,1] )
            o = numpy.transpose( numpy.flip( o, axis=1), [0,2,1] )    
        else:
            o = img

        return o
    ##
    
    P = geometry['geom']['P']
    M = geometry['geom']['M']
    J = 256

    #new = numpy.zeros( frame.shape )

    new = - numpy.ones( frame.shape )

    angles = [0, 90, 180, 270]
    
    if len(frame.shape)==3:
        print('ssc-pimega error! Function not implemented for image blocks!')
                    
    if len(frame.shape)==2:

        #
        #images = [ frame[0:1536,0:1536], frame[0:1536,1536:3072], frame[1536:3072,1536:3072], frame[1536:3072,0:1536] ]

        #gambiarra com tempo definido para acabar
        #rotacao vem do backend
        if do_rot is True:
            images = [ _rotation2D_( frame[0:1536,0:1536],0),
                       _rotation2D_( frame[0:1536,1536:3072], 90),
                       _rotation2D_( frame[1536:3072,1536:3072], 180),
                       _rotation2D_( frame[1536:3072,0:1536], 270)
                      ]
        else:
            images = [  frame[0:1536,0:1536], frame[0:1536,1536:3072], frame[1536:3072,1536:3072], frame[1536:3072,0:1536] ]


        for module in range(4):

            imModule =  numpy.flipud( images[module] )
                
            for j in range(P):

                s = set_module_strip( module, j )

                if s in geometry['hexa']:
                
                    #iy, ix,  _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = geometry['interp'][s]
                    #ix = geometry['LUT'][0][s]
                    #iy = geometry['LUT'][1][s]

                    iy, ix, iyp1, iym1, ixp1, ixm1, iy_ht, ix_ht, iy_hb, ix_hb, iy_vl, ix_vl, iy_vr, ix_vr, iy_gap1, ix_gap1, iy_gap2, ix_gap2 = geometry['interp'][s]
                    
                    stripe =  imModule[j*J: j*J + J, :]

                    #nearest-neighbourhood interpolation
                    new[iy, ix] =  stripe

                    #improving interpolation for missing values
                    if geometry['fill'] == True:
                        
                        projectedHexa = new[ iy.min():iy.max(), ix.min():ix.max() ]
                        
                        missing = (projectedHexa == -1)
                    
                        projectedHexa2 = interpolate_missing_pixels(projectedHexa, missing, 'nearest', SUSP540D)
                    
                        new[ iy.min():iy.max(), ix.min():ix.max()] = projectedHexa2

    new[ new < 0 ] = -1

    return new


def get_geometry540D ( params, *args ):

    J = 256
    P = 6
    M = 6
    
    L = params['distance']

    susp = params['susp']
    v =  params['normal'] 
    a  = params['a']
    z  = params['z']
    offsetx = params['ox']
    offsety = params['oy']
    center = params['center']
    shift  = params['shift']
    typedet = params['typedet']

    RxM = numpy.array( [ params['rotx'][0], params['rotx'][1], params['rotx'][2], params['rotx'][3] ] )
    RyM = numpy.array( [ params['roty'][0], params['roty'][1], params['roty'][2], params['roty'][3] ] )
    RzM = numpy.array( [ params['rotz'][0], params['rotz'][1], params['rotz'][2], params['rotz'][3] ] )
    Ox = numpy.array( [ params['ox'][0], params['ox'][1], params['ox'][2], params['ox'][3] ] )
    Oy = numpy.array( [ params['oy'][0], params['oy'][1], params['oy'][2], params['oy'][3] ] )
            
    dL = numpy.array( [ params['offset'][0], params['offset'][1], params['offset'][2], params['offset'][3] ] )
   
    ####

    cmask = False #compute mask for gaps? only needed for the forward problem | on demand ! 
    
    if not args:
        boxinfo = get_bounding_box( v, RxM, RyM, RzM, L, dL, a, center, typedet )  
    else:
        boxinfo = args[0]

    #
    
    geometry = build_geometry_540D(J, P, M, L, v, RxM, RzM, RyM, cmask, a, z, dL, Ox, Oy, boxinfo, shift, typedet, susp)
    
    geometry['susp'] = params['susp']
    geometry['detector'] = ['all', None, None]
    geometry['boxinfo'] = boxinfo
    geometry['typedet'] = params['typedet']
    geometry['mode'] = params['mode']
    geometry['hexa'] = params['hexa']
    geometry['fill'] = params['fill']
    geometry['crop'] = params['crop']
    geometry['module'] = params['module']

    return geometry

def gaps540D( geometry ) :

    """ Function to compute gaps related to planar and nonplanar pimega/540D detector. 
    
    Args:
        geometry: Geometrical information from <geometry540D>

    Returns:
        (ndarray): 3072x3072 matrix with gaps 
    
    """

    project = geometry['project'] 
    params  = project['input']
    distance = geometry['distance']

    _params   = {'geo': params['geo'], 'opt': params['opt'], 'mode': params['mode'] ,'susp': params['susp'], 'fill': True}
    _project  = dictionary540D( distance, _params )
    _geometry = geometry540D( _project )

    gaps = backward540D( numpy.ones([3072,3072]), _geometry)
    
    gaps = ( gaps < 0 ) * 1.0
    
    return gaps


def geometry540D ( params, *args ):

    """ Function to compute prior geometrical information related to planar and nonplanar pimega/540D detector. 
    
    Args:
        params: input parameters 
        args: extra arguments

    Returns:
        (dict): Geometrical information
    
    """
     
    if params['typedet'] == "nonplanar":
        
        geometry = get_geometry540D( params, *args )

    elif params['typedet'] == "planar":
    
        if params['mode'] == "virtual":
            
            # 04/Nov/2022
            # temporary solution: remove virtual images for real/planar 540D
            # E.Miqueles
 
            #geometry = get_geometry540D( params, *args )
            geometry = get_geometry540D_planar( params, *args )
        else:
        
            geometry = get_geometry540D_planar( params, *args )
           

    #
    #
    
    geometry['distance'] = params['distance']
    geometry['project']  = params
    #
    #
    #

    return geometry


def get_backward540D( matrix, geometry, *args ):
    
    if geometry['detector'][0] == 'all':
           
        # Added 20.March.2023
        # @miqueles: requested by DET/CAT for XPCS techniques
        if matrix.shape[0]==1536 and matrix.shape[1]==1536:
          
            if 'module' in geometry.keys():

                _matrix_ = numpy.zeros([ 3072, 3072 ])

                if geometry['module'] == 0: 
                    _matrix_[ 0:1536, 0:1536 ] = matrix
                elif geometry['module']==1:
                    _matrix_[ 0:1536, 1536:3072 ] = matrix
                elif geometry['module']==2:
                    _matrix_[ 1536:3072, 1536:3072 ] = matrix 
                elif geometry['module']==3:
                    _matrix_[ 1536:3072, 0:1536 ] = matrix
                else:
                    print('ssc-pimega error: wrong module number!')
                    return None

                matrix = numpy.copy(_matrix_)

            else:
                print('ssc-pimega error: please define a module!')
                return None

        else:
            if matrix.shape[0]!=3072 or matrix.shape[1]!=3072:
                print('ssc-pimega error: go find a 3072 x 3072 image!')
                return None
        #
        #


        susp = geometry['susp']

        new = set_suspicious_pixels_540D( matrix, susp )
        
        new = correct_image_backward_540D( new, geometry, *args)

        if geometry['module'] == 0:
            new = new[0:1536, 0:1536]
        elif geometry['module'] == 1:
            new = new[0:1536, 1536:3072]
        elif geometry['module'] == 2:
            new = new[1536:3072, 1536:3072]
        elif geometry['module'] == 3:
            new = new[1536:3072, 0:1536]
        
        return new
        

def backward540D( matrix, geometry, *args ):
    """ Function to restore a given frame using a measured pimega/pi540D data
    
    Args:
        matrix: digital 3072x3072 measured matrix from pimega/pi540D.
        geometry: geometrical data related to pimega/pi540D (see ``dictionary540D()``). 

    Returns:
        (ndarray): restored matrix
    
    """
    
    if geometry['typedet'] == "nonplanar":
        
        new = get_backward540D( matrix, geometry, *args )
            
    else:
        
        if geometry['mode'] == "virtual":
                
            # 04/Nov/2022
            # temporary solution: zoom out virtual/planar images for the backend visualizer
            # E.Miqueles
            
            #forcing to be real/planar
            geometry['mode'] = "real" 
            
            new_real = get_backward540D_planar( matrix, geometry, *args )
            
            #zoom out
            def zoom_feature(img,a):    
                d = abs(img.shape[1] - img.shape[0])
                if min(img.shape[1], img.shape[0]) == img.shape[0]:
                    img = numpy.vstack((img, -1 * numpy.ones([d,img.shape[1]] )))
                else:
                    img = numpy.hstack((img, -1 * numpy.ones([img.shape[0],d] )))
                    
                    xx = numpy.linspace(-1.0,1.0,max(img.shape[1], img.shape[0]) )
                yy = numpy.linspace(-1.0,1.0,max(img.shape[1], img.shape[0]))
                fun =  interpolate.RectBivariateSpline(xx, yy, img, kx=1, ky=1, s=0)
                x = numpy.linspace(-a,a,3072) 
                y = numpy.linspace(-a,a,3072)
                return fun(x,y)
                
            new = zoom_feature( new_real, 1)            

        else:

            new = get_backward540D_planar( matrix, geometry, *args)


    return new


def forward540D( matrix, geometry ):
    """ Function to simulate a given frame using Pi540D with a simulated image
    
    Args:
        matrix: digital 3072x3072 input matrix.
        geometry: geometrical data related to Pimega/Pi540D. See function geometry540D(). 

    Returns:
        (ndarray): (simulated) measured matrix
    
    """
   
    if matrix.shape[0]!=3072 or matrix.shape[1]!=3072:
        print('ssc-pimega error: go find a 3072 x 3072 image!')
        return None
    else:

        if geometry['typedet'] == "nonplanar":
    
            new = correct_image_forward_540D( matrix , geometry)

        else:

            new = correct_image_forward_540D_planar( matrix , geometry) 
            
    return new


def rotate(im, angle):

    N = im.shape[0]

    #mesh
    x = numpy.linspace(-1,1,N)
    xx,yy=numpy.meshgrid(x,x)

    #mesh rotation
    X =   xx * numpy.cos(angle) + yy * numpy.sin(angle)
    Y = - xx * numpy.sin(angle) + yy * numpy.cos(angle)

    #index
    ix = numpy.floor((X - x[0])/(x[1]-x[0])).astype(numpy.int)
    iy = numpy.floor((Y - x[0])/(x[1]-x[0])).astype(numpy.int)

    ix[ ix > N-1] = N-1
    ix[ ix < 0] = 0
    iy[ iy > N-1] = N-1
    iy[ iy < 0] = 0

    return im[iy, ix]


def get_project_values_geometry( *args ):

    if not args:
        ref = {'geo':'nonplanar',  'opt':True, 'mode': 'virtual'}
    else:    
        ref = args[0]
        
    if ref['geo'] == "nonplanar":

        dummy = 24 * [0]
        
        #only virtual mode implemented!
        
        if ref['opt'] == True:

            #August/2021
            
            '''
            _ox_ = [205.86666671, 234.19929424, 230.61359106, 190.06388522, 199.8262727,
                    223.59649503, 152.49413884, 189.82181656, 190.99168855, 159.54318735,
                    123.96292709, 142.82486717, 264.65722539, 155.54900107, 298.94580806,
                    263.03181203, 319.3004465,  225.28608995, 196.06045161, 187.99610669,
                    239.57376855, 212.76689554, 140.73680125, 182.74192213]
            
            _oy_ = [ 578.94010541, 772.19953675,  854.84756214, 1102.35682257, 1404.19988675,
                     1540.3769321,  413.35229965,  771.99217147,  799.29756268, 1139.91260565,
                     1200.62329679, 1595.30465623,  592.29993094,  895.81762205,  923.64336736,
                     1280.94305346, 1308.02778451, 1444.41623425,  524.69611957,  717.5640991,
                     854.78817484, 1267.18264634, 1293.90067334, 1650. ]
            
            a  = numpy.array( 4 * [0] )
            rx = numpy.array( 24 * [-6.75] )
            ry = numpy.array( 24 * [0] )
            rz = numpy.array( 24 * [0] )
            L  = numpy.array( 24 * [0] )
            v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
            center = [0,0]
            z = 1
            ox  = numpy.array( _ox_ )
            oy  = numpy.array( _oy_ )
            '''

            #10/August/2022 (CAT/SAPUCAIA)
            '''
            a  = numpy.array([0., 0., 0., 0.])

            rx = numpy.array([-6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75,
                              -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75,
                              -6.75, -6.75, -6.75, -6.75, -6.75, -6.75])

            ry = numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0.])

            rz = numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0.])

            L = numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0.])

            ox = numpy.array([ -93.60127772,  -53.04697773, -120.82169554,  -84.31745284,
                               -56.5651624 ,  -63.97768354, -132.95628706,  -82.87116618,
                               -36.24221455,  -54.22205047,   15.36596832,    3.1026839 ,
                               -150.46633181,  -51.00595363,   49.50791824,  142.4871507 ,
                               170.85469722,  132.98511417, -173.22518986,  -94.1377824 ,
                               -15.33952014,   27.93549775,   23.07319651,   49.89657887])

            oy = numpy.array([ 168.05429447,  401.16253405,  620.16855194,  784.96740027,
                               1032.71569908, 1169.40076099,  257.20392745,  534.94168169,
                               674.33555799,  922.19602498, 1169.19942938, 1196.38545022,
                               201.4048795 ,  451.81828022,  644.31089315,  836.49460112,
                               973.87184114, 1169.48669668,  169.03809743,  364.88089303,
                               563.57132147,  754.25869976, 1003.0932477 , 1161.5364978 ])

            v = numpy.array([0., 0., 0.])
            center = numpy.array([0., 0.])
            z = 1.0
            '''

            #24/October/2022 (CAT/SAPUCAIA)

            a = numpy.array([0., 0., 0., 0.])
            rx = numpy.array([-6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75,
                         -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75, -6.75,
                         -6.75, -6.75, -6.75, -6.75, -6.75, -6.75])
            ry = numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0.])
            rz = numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0.])
            L = numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0.])
            
            ox = numpy.array([-90.72660962, -63.25296052, 16.98226961, 35.0772516, 89.27059423, 124.72548294, 
                         -212.31345687, -116.45961454, -22.41896552, 56.14109804, 116.78865497, 45.08773692, 
                         -92.54939172, -11.75287748, 13.13305114, 85.49641223, 84.50338608, 61.52684817, 
                         -246.98045781, -293.02877083,  -226.49346356,  -220.79768323,  -400.8674323 ,  -420.76187557] )

            oy = numpy.array([ 264.35990633,  430.4784431 ,  596.36695308,  817.42400924,
                          1037.07985285, 1200.91967259,  232.8455623 ,  482.70372941,
                          622.70798484,  873.24025367, 1122.42306266, 1259.44562535,
                          267.58393971,  517.26427973,  709.63331812,  907.29198848,
                          1095.83418477, 1288.12536874,  231.33593393,  478.94434469,
                          670.48055412,  862.36849373, 1108.68732295, 1246.48078927])
            v = numpy.array([0., 0., 0.])
            center = numpy.array([0., 0.])
            z = 0.98
            

            #####
            
            x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z, dummy)) ) )

        else:
            pxl = MEDIPIX
            ox_ = 4 * pxl
            oy_ = 10 * pxl
            
            a  = numpy.array( 4 * [0] )
            rx = numpy.array( 24 * [-6.75] )
            ry = numpy.array( 24 * [0] )
            rz = numpy.array( 24 * [0] )
            L  = numpy.array( 24 * [0] )
            v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
            center = [0,0]
            z = 0.98
            ox  = numpy.array( 24 * [ox_] )
            oy  = numpy.array( [oy_, oy_ + 3 * pxl, oy_ + 6 * pxl, oy_ + 9 * pxl, oy_ + 12 * pxl, oy_ + 15 * pxl,
                                oy_, oy_ + 3 * pxl, oy_ + 6 * pxl, oy_ + 9 * pxl, oy_ + 12 * pxl, oy_ + 15 * pxl,
                                oy_, oy_ + 3 * pxl, oy_ + 6 * pxl, oy_ + 9 * pxl, oy_ + 12 * pxl, oy_ + 15 * pxl,
                                oy_, oy_ + 3 * pxl, oy_ + 6 * pxl, oy_ + 9 * pxl, oy_ + 12 * pxl, oy_ + 15 * pxl] )
            
            x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z, dummy)) ) )

    else: # planar geometry

        dummy = 24 * [0]
        
        if ref['opt'] == True:

            if ref['mode'] == "virtual":

                '''
                #to be done!
                ox_ = - 3*55
                oy_ = 0
                a  = numpy.array( 4 * [0] )
                rx = numpy.array( 24 * [0] )
                ry = numpy.array( 24 * [0] )
                rz = numpy.array( 24 * [0] )
                L  = numpy.array( 24 * [0] )
                v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
                center = [0,0]
                z = 1.13
                ox  = numpy.array( 24 * [ox_] )
                
                oy  = numpy.array( [227, 2602, 2 * 2700 , 3 * 2750, 4 * 2763, 5 * 2552,
                                    233, 2769, 2 * 2758 , 3 * 2777, 4 * 2764, 5 * 2712,
                                    255, 2831, 2 * 2793 , 3 * 2780, 4 * 2746, 5 * 2677, 
                                    231, 2732, 2 * 2711 , 3 * 2712, 4 * 2748, 5 * 2743 ] )
                
                x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z, dummy)) ) )
                '''

                pxl = MEDIPIX
                ox_ = 0
                oy_ = 0
                a  = numpy.array( 4 * [0] )
                rx = numpy.array( 24 * [0] )
                ry = numpy.array( 24 * [0] )
                rz = numpy.array( 24 * [0] )
                L  = numpy.array( 24 * [0] )
                v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
                center = [0,0]
                z = 1

                gaps = numpy.array([[50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0]]).flatten()
                
                oy = numpy.array([[0,0,-4,-4,-2,0],
                                  [0,0,0,-4,-4,0],
                                  [0,0,-4,0,0,-8],
                                  [5,5,5,5,5,0]]).flatten()
                
                ox = numpy.array([[-11,-11,-11,-11,-11,-11],
                                  [-12,-12,-12,-12,-12,-12],
                                  [-26,-26,-26,-26,-26,-26],
                                  [-32,-32,-32,-32,-32,-32]]).flatten()
                
                x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z,gaps)) ) )
 

            else:
                
                #to be done! 
                ox_ = - 3*55
                oy_ = 0
                a  = numpy.array( 4 * [0] )
                rx = numpy.array( 24 * [0] )
                ry = numpy.array( 24 * [0] )
                rz = numpy.array( 24 * [0] )
                L  = numpy.array( 24 * [0] )
                v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
                center = [0,0]
                z = 0.98

                '''
                gaps = numpy.array([[50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0]]).flatten()
                
                oy = numpy.array([[4,4,4,4,4,4],
                                  [4,4,4,4,4,4],
                                  [4,4,4,4,4,4],
                                  [4,4,4,4,4,4]]).flatten()
                
                n = 5 * 3 #due to gaps
                ox = numpy.array([[-4,-4,-4,-4,-4,-4],
                                  [-4,-4,-4,-4,-4,-4],
                                  [-n-4,-n-4,-n-4,-n-4,-n-4,-n-4],
                                  [-n-4,-n-4,-n-4,-n-4,-n-4,-n-4]]).flatten()

                '''
                
                gaps = numpy.array([[50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0]]).flatten()
                
                oy = numpy.array([[0,0,-4,-4,-2,0],
                                  [0,0,0,-4,-4,0],
                                  [0,0,-4,0,0,-8],
                                  [5,5,5,5,5,0]]).flatten()
                
                ox = numpy.array([[-11,-11,-11,-11,-11,-11],
                                  [-12,-12,-12,-12,-12,-12],
                                  [-26,-26,-26,-26,-26,-26],
                                  [-32,-32,-32,-32,-32,-32]]).flatten()
                
                
                x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z, gaps)) ) )

        else:

            if ref['mode'] == "virtual":

                '''
                ox_ = - 3*55
                oy_ = 0
                a  = numpy.array( 4 * [0] )
                rx = numpy.array( 24 * [0] )
                ry = numpy.array( 24 * [0] )
                rz = numpy.array( 24 * [0] )
                L  = numpy.array( 24 * [0] )
                v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
                center = [0,0]
                z = 1
                ox  = numpy.array( 24 * [ox_] )
                
                oy  = numpy.array( [227, 2602, 2 * 2700 , 3 * 2750, 4 * 2763, 5 * 2552,
                                    233, 2769, 2 * 2758 , 3 * 2777, 4 * 2764, 5 * 2712,
                                    255, 2831, 2 * 2793 , 3 * 2780, 4 * 2746, 5 * 2677, 
                                    231, 2732, 2 * 2711 , 3 * 2712, 4 * 2748, 5 * 2743 ] )
                
                x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z, dummy)) ) )
                '''

                pxl = MEDIPIX
                ox_ = 0
                oy_ = 0
                a  = numpy.array( 4 * [0] )
                rx = numpy.array( 24 * [0] )
                ry = numpy.array( 24 * [0] )
                rz = numpy.array( 24 * [0] )
                L  = numpy.array( 24 * [0] )
                v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
                center = [0,0]
                z = 0.98

                gaps = numpy.array([[50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0]]).flatten()
                
                oy = numpy.array([[0,0,-4,-4,-2,0],
                                  [0,0,0,-4,-4,0],
                                  [0,0,-4,0,0,-8],
                                  [5,5,5,5,5,0]]).flatten()
                
                ox = numpy.array([[-11,-11,-11,-11,-11,-11],
                                  [-12,-12,-12,-12,-12,-12],
                                  [-26,-26,-26,-26,-26,-26],
                                  [-32,-32,-32,-32,-32,-32]]).flatten()
                
                x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z,gaps)) ) )
 

            else:
                    
                pxl = MEDIPIX
                ox_ = 0
                oy_ = 0
                a  = numpy.array( 4 * [0] )
                rx = numpy.array( 24 * [0] )
                ry = numpy.array( 24 * [0] )
                rz = numpy.array( 24 * [0] )
                L  = numpy.array( 24 * [0] )
                v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
                center = [0,0]
                z = 0.98

                '''
                gaps = numpy.array([[50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0]]).flatten()
                
                oy = numpy.array([[4,4,4,4,4,4],
                                  [4,4,4,4,4,4],
                                  [4,4,4,4,4,4],
                                  [4,4,4,4,4,4]]).flatten()
                
                n = 5 * 3 #due to gaps
                ox = numpy.array([[-4,-4,-4,-4,-4,-4],
                                  [-4,-4,-4,-4,-4,-4],
                                  [-n-4,-n-4,-n-4,-n-4,-n-4,-n-4],
                                  [-n-4,-n-4,-n-4,-n-4,-n-4,-n-4]]).flatten()
                
                
                '''
                gaps = numpy.array([[50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0],
                                    [50, 50, 50, 50, 50,0]]).flatten()
                
                oy = numpy.array([[0,0,-4,-4,-2,0],
                                  [0,0,0,-4,-4,0],
                                  [0,0,-4,0,0,-8],
                                  [5,5,5,5,5,0]]).flatten()
                
                ox = numpy.array([[-11,-11,-11,-11,-11,-11],
                                  [-12,-12,-12,-12,-12,-12],
                                  [-26,-26,-26,-26,-26,-26],
                                  [-32,-32,-32,-32,-32,-32]]).flatten()
                
                x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z,gaps)) ) )
 
    return x


def dictionary540D( L0, *args ):

    """ Get default values for geometrical image restoration at the pimega/pi540D.

    Args:
        L0: distance sample to detector
        args: extra arguments

    Returns:
        (dict): Dictionary with informations about the geometrical setup 

    * The output dictionary is given below:

    .. code-block:: python 
        
       det = {
        'rotz': [ [_], [_], [_], [_] ], 
        'roty': [ [_], [_], [_], [_] ],
        'rotx': [ [_], [_], [_], [_] ],
        'distance': L0,
        'normal': [ _ , _ , _ ],
        'a': [ _ , _ , _ , _ ],
        'z': _,
        'offset':  [ [_], [_], [_], [_] ] , 
        'ox': [ [_], [_], [_], [_] ],
        'oy': [ [_], [_], [_], [_] ],
        'shift': _,
        'typedet': _
        'mode': _,
        'gaps': [ [_], [_], [_], [_] ] ,
        'hexa': _,
        'fill': _,
        'scale': _,
        'susp': _
        }

    * ``det['rotx']``, ``det['roty']``, ``det['rotz']`` 

        These are angle lists for each stripe. Each position of this list is related to a module, 
        with another 6-length list embeeded. The first position of this 6-length list relates to the 
        bottom stripe, last position to the  top stripe. Each orthornomal basis :math:`\{n, n^\perp, n^{\times}\}` 
        is defined as a rotation of the vector (0,0,1) with respect to ``rotx``, ``roty`` and ``rotz``, respectively.
        Angles are defined in degrees. 

    * ``det['distance']`` 

        The the input distance, from sample to the virtual (restored) image.

    * ``det['normal']``

        These are the angles (rx,ry,rz) to set the normal direction for the virtual image.
        The normal vector is defined as a rotation of the vector (0,0,1) with respect to ``rx``, ``ry`` 
        and ``rz``, respectively. Angles are defines in degrees.

    * ``det['a']``
    
        This is a 4-length list, each one representing a rotation wrt to the z-axis for a
        given module (following module indexes). Angles are defined in degrees.

    * ``det['z']``

        Factor scale (zoom in/out) for virtual images. Default is 1.

    * ``det['offset']``

        These are offset (in the beam direction) lists for each stripe. Each position of this list is related to a module, 
        with another 6-length list embeeded. The first position of this 6-length list relates to the 
        bottom stripe, last position to the  top stripe. Natural offsets are computed with respect to the input 
        distance L0; hence the values given here represent numerical deviations for these values. Offset values
        are defined in m.
        
    * ``det['ox']``, ``det['oy']``

        These are shift (in the orthogonal beam direction) lists for each stripe. Each position of this 
        list is related to a module, with another 6-length list embeeded. The first position of this 6-length list relates to the 
        bottom stripe, last position to the  top stripe. 

    * ``det['shift']``

        ``True`` indicates that for pixels lying in the chip boundaries, pixel center is 
        shifted to the center. On the opposite, ``False`` indicates that the *bump-bond* is
        the center of the pixel. Note that for all other pixels, except boundary ones, the 
        bump-bond always coincide with the pixel center. 

    * ``det['typedet']``
    
        The detector type, defined according to strings ``planar`` or ``nonplanar``.
        
    * ``det['hexa']``  

        Integer sequence indicating the hexa's that we want to restore. As an example,
        ``[3,9,15,21]`` indicates that only the fourth hexa of each module will be restored.
        On the other hand, ``range(24)`` indicates that everything will be restored.

    * ``det['mode']``

        Flag indicating ``real`` or ``vìrtual`` restored images. Pimega/540D only
        provide virtual images.

    * ``det['gaps']``
    
       There are 24 gaps that could be defined as input for the planar case. 
       For the nonplanar case, they are defined as a list full of zeros. For 
       instance, the following gap list define 1um as the gap sequence for 
       the module 0, 2um for the module 1, 3um for the module 2 and 4um for 
       the module 3.  

        .. code-block:: python

        'gaps': [ [1,1,1,1,1,1] , [2,2,2,2,2,2] , [3,3,3,3,3,3], [4,4,4,4,4,4] ]
        
        These are gap lists for each stripe. Each position of this 
        list is related to a module, with another 6-length list embeeded. The first 
        position of this 6-length list relates to the bottom stripe, last position to the 
        top stripe. 

    * ``det['scale']``

        Scale factor (zoom-in / zoom-out )

    * ``det['susp']``

        Number of suspicious pixels to be removed for each Medipix/Chip
    
    * Extra arguments are given below. Once the geometrical optimization is 
      performed, the API include optimal values for offset, angles and gap. In 
      this case, the flat 'opt' will be 'True', otherwise 'False'.

    .. code-block:: python
  
       args[0] = {'geo': 'planar', 'opt': True, 'mode': 'real', 'x': xdet}

       xdet = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z)) ) )

    """
    
    #x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z)) ) )      

    if not args:
        x = get_project_values_geometry( *args )
        ref = {'geo': 'nonplanar', 'opt': False, 'mode': 'virtual'}
    else:            
        ref = args[0]
        if 'x' in ref.keys():
            x = ref['x']
            if ref['geo'] == 'planar':
                a  = numpy.array( 4 *[0] )
                r  = numpy.array( 24 * [0] )
                L  = numpy.array( 24 * [0] )
                v  = numpy.array( 3 * [0] )
                center = numpy.array( 2 * [0] )
                z = 1
                dic = ref['x']
                if 'rz' in dic.keys():
                    x = numpy.array( list(numpy.hstack((a,r,r,dic['rz'],L,dic['ox'],dic['oy'],v,center,z,dic['gaps'])) ) )
                else:
                    x = numpy.array( list(numpy.hstack((a,r,r,r,L,dic['ox'],dic['oy'],v,center,z,dic['gaps'])) ) )
            else:
                dic = ref['x']
                x = numpy.array( list(numpy.hstack((dic['a'],dic['rx'],dic['ry'],dic['rz'],dic['offset'],dic['ox'],dic['oy'],dic['normal'],
                                                    dic['center'], dic['z'], numpy.array( 24*[0]) )) ))
            
        else:
            x = get_project_values_geometry( *args )
    
            
    a = x[0:4]
    r = x[4:76]
    L = x[76:100]
    o = x[100:148]
    v = x[148:151]
    center = x[151:153] 
    z = x[153]

    if 'crop' in ref.keys():
        crop = ref['crop']
    else:
        crop = True #only for planar devices
    
    if 'scale' in ref.keys():
        z = ref['scale']

    if 'fill' in ref.keys():
        interp = ref['fill']
    else:
        interp = False

    if 'module' in ref.keys():
        module = ref['module']
    else:
        module = -1

    if 'susp' in ref.keys():
        susp = ref['susp']
    else:
        susp = 0
        
    if ref['geo'] == "planar":
        gaps = x[154:len(x)]
        typedet = "planar"
        mode    = ref['mode']        
    else:
        gaps    = 24*[0]
        typedet = "nonplanar"
        mode    = ref['mode']

    if 'hexa' in ref.keys():
        hexa = ref['hexa']
    else:
        hexa = range(24)
        
    det = {
        'rotz': numpy.array( r[48:72]).reshape([4,6]),   #[ r[48:54], r[54:60], r[60:66], r[66:72] ], 
        'roty': numpy.array( r[24:48]).reshape([4,6]),   #[ r[24:30], r[30:36], r[36:42], r[42:48] ],
        'rotx': numpy.array( r[0:24]).reshape([4,6]),    #[  r[0:6], r[6:12], r[12:18], r[18:24] ],
        'distance': L0,
        'normal': [ v[0], v[1], v[2] ],
        'a': [a[0], a[1], a[2], a[3]],
        'z': z,
        'offset':  numpy.array( L[0:24]).reshape([4,6]), #[  L[0:6], L[6:12], L[12:18], L[18:24] ] , 
        'ox':  numpy.array( o[0:24] ).reshape([4,6]),    #[ o[0:6], o[6:12], o[12:18], o[18:24] ],
        'oy':  numpy.array( o[24:48]).reshape([4,6]),    #[ o[24:30], o[30:36], o[36:42], o[42:48] ],
        'center': center,
        'shift': False,
        'typedet': typedet,
        'mode': mode,
        'gaps': numpy.array( gaps[0:24] ).reshape([4,6]),  # [ gaps[0:6], gaps[6:12], gaps[12:18], gaps[18:24] ]
        'hexa': hexa,
        'fill': interp,
        'susp': susp,
        'crop': crop,
        'module': module,
        'input': ref
        }

    return det
    


def _extract_detector_dictionary_( x, L0, args):
    
    #x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,z)) ) )
    
    a = x[0:4]
    r = x[4:76]
    L = x[76:100]
    o = x[100:148]
    v = x[148:151]
    center = x[151:153] 
    z = x[153]

    ref = args
    
    if ref['geo'] == "planar":
        gaps = x[154:-1]
        typedet = "planar"
        mode    = ref['mode']
    else:
        gaps    = 24*[0]
        typedet = "nonplanar"
        mode    = ref['mode']

    if 'hexa' in ref.keys():
        hexa = ref['hexa']
    else:
        hexa = range(24)
        
    det = {
        'rotz': [ r[48:54], r[54:60], r[60:66], r[66:72] ], 
        'roty': [ r[24:30], r[30:36], r[36:42], r[42:48] ],
        'rotx': [  r[0:6], r[6:12], r[12:18], r[18:24] ],
        'distance': L0,
        'normal': [ v[0], v[1], v[2] ],
        'a': [a[0], a[1], a[2], a[3]],
        'z': z,
        'offset':  [  L[0:6], L[6:12], L[12:18], L[18:24] ] , 
        'ox': [ o[0:6], o[6:12], o[12:18], o[18:24] ],
        'oy': [ o[24:30], o[30:36], o[36:42], o[42:48] ],
        'center': center,
        'shift': False,
        'typedet': typedet,
        'mode': mode,
        'gaps': [ gaps[0:6], gaps[6:12], gaps[12:18], gaps[18:24] ] ,
        'hexa': hexa
        }

    return det


def get_stripe_from_measure_540D(img, k):
    #
    #Extract measure from a given module/stripe
    #
    #mathematical order
    m, s = get_module_strip(k)
    
    s = 5 - s
    
    if m==0:
        out = img[s*256:(s+1)*256,0:1536]
    elif m==1:
        out= img[s*256:(s+1)*256,1536:3072]
    elif m==2:
        out = img[s*256+1536:(s+1)*256+1536,1536:3072]
    elif m==3:
        out = img[s*256+1536:(s+1)*256+1536,0:1536]        

    return out

def set_stripe_from_measure_540D(img, stripe, k):
    #
    #Set measure to a given module/stripe
    #
    #mathematical order
    m, s = get_module_strip(k)
    
    s = 5 - s
    
    if m==0:
        img[s*256:(s+1)*256,0:1536] = stripe
    elif m==1:
        img[s*256:(s+1)*256,1536:3072] = stripe
    elif m==2:
        img[s*256+1536:(s+1)*256+1536,1536:3072] = stripe
    elif m==3:
        img[s*256+1536:(s+1)*256+1536,0:1536] = stripe

    return img


######################################################
#
#
#
#
# PLANAR GEOMETRY: 540D
#
#
#
#
#####################################################

def correct_image_forward_540D_planar(frame, geometry):

    sv = geometry['susp']
    cv = geometry['crop']
    
    #forcing for forward simulation
    geometry['susp'] = 0
    geometry['crop'] = False
    
    back = backward540D ( numpy.ones([3072,3072]), geometry )
    
    mask = (back < 0)
    
    cy = back.shape[0]//2
    cx = back.shape[1]//2
    back[cy - 1536 : cy + 1536, cx - 1536 : cx + 1536 ] = frame
    back[mask]  = -1
    
    mask = (back < 0)
    
    new = - numpy.ones([3072,3072])
        
    for module in range(4):
        
        for j in range(6):
            
            s = set_module_strip(module, j)
                
            ix = geometry['geom']['LUT'][0][s].astype(int)
            iy = geometry['geom']['LUT'][1][s].astype(int)
        
            hexa = back[iy, ix]

            new = set_image_stripe_detector( new, hexa, module, j)

    geometry['susp'] = sv
    geometry['crop'] = cv
            
    return new 


def correct_image_backward_540D_planar(frame, geometry, *args):

    def get_image_stripe_detector( img, m, s, dim):
        #
        #Extract measure from a given module/stripe
        #
        #mathematical order
        s = 5 - s

        if dim==2:
            if m==0:
                out = img[s*256:(s+1)*256,0:1536]
            elif m==1:
                out= img[s*256:(s+1)*256,1536:3072]
            elif m==2:
                out = img[s*256+1536:(s+1)*256+1536,1536:3072]
            elif m==3:
                out = img[s*256+1536:(s+1)*256+1536,0:1536]        
        else:
            if m==0:
                out = img[:,s*256:(s+1)*256,0:1536]
            elif m==1:
                out= img[:,s*256:(s+1)*256,1536:3072]
            elif m==2:
                out = img[:,s*256+1536:(s+1)*256+1536,1536:3072]
            elif m==3:
                out = img[:,s*256+1536:(s+1)*256+1536,0:1536]        
                
        return out

    N, shape, rowv, colv = geometry['geom']['shape']
   
    if len(frame.shape)==3:
        print('ssc-pimega error! Function not implemented for image blocks!')
    
    if len(frame.shape)==2:

        new = -1 * numpy.ones( [N,N] )

        for module in range(4):

            for j in range(6):

                s = set_module_strip (module, j)
                
                if s in geometry['hexa']:
            
                    ix = geometry['geom']['LUT'][0][s].astype(int)
                    iy = geometry['geom']['LUT'][1][s].astype(int)
            
                    stripe = get_image_stripe_detector( frame, module, j, 2)
            
                    new[iy, ix] = stripe

                    
        if geometry['crop'] == True:
            #cropped!
            arr = new[ rowv[0]:rowv[1], colv[0]:colv[1] ]
        else:
            #uncropped!
            arr = new

    arr [ arr < 0 ] = -1
    
    return arr


def build_geometry_540D_planar(Ox, Oy, Gaps):

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

        _row_ = numpy.array(_row_, dtype=int).flatten()
        _col_ = numpy.array(_col_, dtype=int).flatten()
    
        nrows = max(_row_) - min(_row_)
        ncols = max(_col_) - min(_col_)

        rowv = [ min(_row_), max(_row_) ]
        colv = [ min(_col_), max(_col_) ]
    
        return (nrows,ncols), rowv, colv

    N = 4000 #adjust: TODO
    
    shape, rowv, colv = get_shape_pi540D_planar(N, Ox, Oy, Gaps)

    new = -1 * numpy.ones( [N,N])
    cr = N//2
    cc = N//2
    
    LUT = ( numpy.zeros([24,J,H]), numpy.zeros([24,J,H]) ) 

    for k in range(6): #six 'hexa' modules
        
        start_0 = cr - (k+1)*J - Gaps[0][0:k].sum() - Oy[0][k]
        end_0   = start_0 + J
        
        end_1   = cc + (k+1)*J + Gaps[1][0:k].sum() + Oy[1][k]
        start_1 = end_1 - J
        
        end_2   = cr + (k+1)*J + Gaps[2][0:k].sum() + Oy[2][k]
        start_2 = end_2 - J
        
        start_3 = cc - (k+1)*J - Gaps[3][0:k].sum() - Oy[3][k]
        end_3   = start_3 + J
        
        #module-0
        #_ix_ = numpy.arange(cc + Ox[0][k] - H, cc + Ox[0][k])
        _ix_ = get_index_withgap(cc + Ox[0][k] - H, cc + Ox[0][k] ) 
        _iy_ = numpy.arange(start_0,end_0)
        [ix,iy] = numpy.meshgrid(_ix_, _iy_)
        LUT[0][ 0*6 + k] = ix
        LUT[1][ 0*6 + k] = iy
        
        #module-1
        _ix_ = numpy.arange(start_1, end_1)
        #_iy_ = numpy.arange(cr + Ox[1][k] - H, cr + Ox[1][k])
        _iy_ =  get_index_withgap(cr + Ox[1][k] - H, cr + Ox[1][k])
        [iy,ix] = numpy.meshgrid(_iy_, _ix_)
        LUT[0][ 1*6 + k ] = numpy.flipud(ix)
        LUT[1][ 1*6 + k ] = numpy.flipud(iy)
        
        #module-2
        #_ix_ = numpy.arange(cc - Ox[2][k],cc - Ox[2][k] + H)
        _ix_ = get_index_withgap(cc - Ox[2][k], cc - Ox[2][k] + H)
        _iy_ = numpy.arange(start_2, end_2)
        [ix,iy] = numpy.meshgrid(_ix_, _iy_)
        LUT[0][ 2*6 + k ] = numpy.fliplr( numpy.flipud( ix ) )
        LUT[1][ 2*6 + k ] = numpy.fliplr( numpy.flipud( iy ) )
        
        #module-3
        _ix_ = numpy.arange(start_3, end_3)
        #_iy_ = numpy.arange(cc - Ox[3][k],cc - Ox[3][k] + H)
        _iy_ = get_index_withgap( cc - Ox[3][k], cc - Ox[3][k] + H)
        [iy,ix] = numpy.meshgrid(_iy_, _ix_)
        LUT[0][ 3*6 + k ] = numpy.flipud( numpy.fliplr( numpy.flipud( ix ) ) )
        LUT[1][ 3*6 + k ] = numpy.flipud( numpy.fliplr( numpy.flipud( iy ) ) )

    geometry = {}
    geometry['geom'] = {}
    geometry['geom']['LUT'] = [ LUT[0], LUT[1] ]
    geometry['geom']['shape'] = [N, shape, rowv, colv]
    geometry['geom']['P'] = 6
    geometry['geom']['M'] = 6
    geometry['geom']['J'] = 256
    
    return geometry


def get_geometry540D_planar( params, *args ):
             
    Ox    = numpy.array( [ params['ox'][0], params['ox'][1], params['ox'][2], params['ox'][3] ] )
    Oy    = numpy.array( [ params['oy'][0], params['oy'][1], params['oy'][2], params['oy'][3] ] )
    Gaps  = numpy.array( [ params['gaps'][0], params['gaps'][1], params['gaps'][2], params['gaps'][3] ]  )

    geometry = build_geometry_540D_planar( Ox, Oy, Gaps )
       
    geometry['typedet'] = params['typedet']
    geometry['mode'] = params['mode']
    geometry['hexa'] = params['hexa']
    geometry['susp'] = params['susp']
    geometry['crop'] = params['crop']

    return geometry


def get_backward540D_planar( matrix, geometry, *args ):
        
    susp = geometry['susp']
    new = set_suspicious_pixels_540D( matrix, susp )
        
    new = correct_image_backward_540D_planar( new, geometry, *args)
    return new
        

def forward540D_planar( matrix, geometry ):
    
    if matrix.shape[0]!=3072 or matrix.shape[1]!=3072:
        print('ssc-pimega input error: go find a 3072 x 3072 image!')
        return None
    else:
        new = correct_image_forward_540D_planar( matrix , geometry)
        
    return new

######################################################
#
#
#
#
# GPU FUNCTIONS
#
#
#
#
#####################################################

def next_power_of_2(x):  
        return 1 if x == 0 else 2**(x - 1).bit_length()

def ioSet_Backward540D( dic ):

    """ Function to restore a given H5 file using a superscalar pipeline
        Data is written at an specific folder given within dic.
    
    Args:
        dic: input dictionary

    Returns:
        (None)

    """

    path     = dic['path']
    outpath  = dic['outpath']
    order    = dic['order']
    dataset  = dic['dataset']
    rank     = dic['rank']
    ngpus    = len( dic['gpus'] )
    gpus     = dic['gpus']
    init     = dic['init']
    final    = dic['final']
    timing   = dic['timing']
    saving   = dic['saving']
    blocksize= next_power_of_2 ( dic['blocksize'] )
    geometry = dic['geometry']

    filling = geometry['fill']
    if filling == False:
        fill = 0
    else:
        fill = 1
    
    ##
    ##

    gaps = gaps540D( geometry ) 

    #_distance = geometry['geom']['L'][0]
    #_params   = {'geo': 'nonplanar', 'opt': True, 'mode': 'virtual' ,'susp': geometry['susp'], 'scale': 1}
    #_project  = dictionary540D( _distance, _params )
    #_geometry = geometry540D( _project )
    #gaps      = backward540D( numpy.zeros([3072,3072]), _geometry)
    #gaps[ gaps < 0] = 1
    ##
    ##
    ##

    daxpy_con = dic['daxpy'][0]
    daxpy_img = dic['daxpy'][1]

    roi      = dic['roi']
    center   = dic['center']
    flatimg  = dic['flat']
    emptyimg = dic['empty']
    maskimg  = dic['mask']
    oshape   = [ 2*roi , 2*roi]

    ix = geometry['LUT'][0]
    iy = geometry['LUT'][1]

    xmin = geometry['borders']['xmin']
    xmax = geometry['borders']['xmax']
    ymin = geometry['borders']['ymin']
    ymax = geometry['borders']['ymax']
    
    # create byte objects from the strings
    b_path    = path.encode('utf-8')
    b_outpath = outpath.encode('utf-8')
    b_order   = order.encode('utf-8')
    b_dataset = dataset.encode('utf-8')
    b_rank    = rank.encode('utf-8')

    ishape    = numpy.array([0,0,0], dtype=numpy.intc )
    b_ishape  = numpy.array(ishape, dtype=numpy.intc).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    b_ngpus = ctypes.c_int( ngpus )
    b_gpus  = numpy.array(gpus, dtype=numpy.intc).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    b_shape = numpy.array(oshape, dtype=numpy.intc).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    b_init   = ctypes.c_int( init )
    b_final  = ctypes.c_int( final )
    b_timing = ctypes.c_int( timing * 1 )
    b_saving = ctypes.c_int( saving * 1 )
    b_blocksize = ctypes.c_int( blocksize )
    b_fill      = ctypes.c_int( fill )

    b_center  = numpy.array(center, dtype=numpy.intc).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    b_roi     = ctypes.c_int( roi )
    b_susp    = ctypes.c_int( geometry['susp'] )

    b_daxpycon= ctypes.c_float( daxpy_con ) 
    b_daxpyimg  = daxpy_img.astype( numpy.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    b_gapsimg  = gaps.astype( numpy.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_flatimg  = flatimg.astype( numpy.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_emptyimg = emptyimg.astype( numpy.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_maskimg = maskimg.astype( numpy.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    b_ix = getPointer( ix.flatten() , dtype=numpy.intc ).ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
    b_iy = getPointer( iy.flatten() , dtype=numpy.intc ).ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
    
    b_xmin = getPointer( xmin.flatten() , dtype=numpy.intc ).ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
    b_xmax = getPointer( xmax.flatten() , dtype=numpy.intc ).ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
    b_ymin = getPointer( ymin.flatten() , dtype=numpy.intc ).ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
    b_ymax = getPointer( ymax.flatten() , dtype=numpy.intc ).ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
    
    name   = str( uuid.uuid4() )
    b_uuid = name.encode('utf-8')


    nimgs = libssc_pimega.ssc_pimega_pi540D_backward_pipeline(b_ishape,
                                                              b_path ,
                                                              b_outpath,
                                                              b_order,
                                                              b_rank,
                                                              b_dataset,
                                                              b_ngpus,
                                                              b_gpus,
                                                              b_shape,
                                                              b_init,
                                                              b_final,
                                                              b_blocksize,
                                                              b_timing,
                                                              b_saving,
                                                              b_ix,
                                                              b_iy,
                                                              b_xmin,
                                                              b_xmax,
                                                              b_ymin,
                                                              b_ymax,
                                                              b_center,
                                                              b_roi,
                                                              b_flatimg,
                                                              b_emptyimg,
                                                              b_maskimg,
                                                              b_daxpyimg,
                                                              b_daxpycon,
                                                              b_susp,
                                                              b_uuid,
                                                              b_gapsimg,
                                                              b_fill)


    return name, nimgs


def ioGet_Backward540D( dic, uid, nimgs ):

    """ Function to load previously restored H5 file.
    
    Args:
        dic: input dictionary
        uid: uuid4 string from <ioSet_Backward540D>
        nimgs: number of images

    Returns:
        (ndarray): restored volume wrt to dic parameters!
        (ndarray): output average without restoration
        (ndarray): input average

    """

    outpath  = dic['outpath']
    roi      = dic['roi']
    oshape   = [ 2*roi , 2*roi ]

    output = numpy.zeros([ nimgs , oshape[0], oshape[1] ] )
    
    blocksize = next_power_of_2 ( dic['blocksize'] )
    
    if nimgs < blocksize:
        nblocks = 1
    else:
        #nblocks   = ( nimgs )//blocksize
        nblocks    = int ( numpy.ceil( (nimgs )/blocksize ) )
        
    for k in range(nblocks):
 
        path = outpath + '/ssc_temp_' + uid + '_{}.b'.format(k)
        block  = numpy.fromfile( path, dtype=numpy.float32).reshape([blocksize, 2*dic['roi'], 2*dic['roi']])
        _start_ = k * blocksize 
        _end_   = min( (k+1) * blocksize, nimgs) 
        output[ _start_: _end_, :, : ] = block[0: (_end_ - _start_)  ,:,:]

        
    return output 



def ioClean_Backward540D( dic, uid ):

    """ Function to clean previously restored H5 file from a given folder.
    
    Args:
        dic: input dictionary
        uid: uuid4 string from <ioSet_Backward540D>

    Returns:
        (None)

    """

    cmd = "rm " + dic['outpath'] + "/ssc_temp_" + uid + "_*"

    subprocess.call(cmd, shell=True)

    #cmd = "rm " + dic['outpath'] + "/ssc_temp_avgInput_" + uid + "_*"
    #subprocess.call(cmd, shell=True)
    #
    #cmd = "rm " + dic['outpath'] + "/ssc_temp_avgOutput_" + uid + "_*"
    #subprocess.call(cmd, shell=True)




############################################
####
#### IO read and restore for multiple files
####
############################################

def _get_size_from_shape(shape):
    return functools.reduce(lambda x, y: x * y, shape)
 
def _create_np_shared_array(shape, dtype, ctype):
    # Feel free to create a map from usual dtypes to ctypes. Or suggest a more elegant way
    size = _get_size_from_shape(shape)
    shared_mem_chunck = multiprocessing.sharedctypes.RawArray(ctype, size)
    numpy_array_view = numpy.frombuffer(shared_mem_chunck, dtype).reshape(shape)
    return numpy_array_view

def _worker_io_batch_(params, idx_start,idx_end, info, gpu):

    dic = params[2]

    for k in range(idx_start, idx_end):

        start = time.time()

        diccopy = dic.copy()

        diccopy['path'] = dic['path'][k]
        diccopy['ngpus'] = 1
        diccopy['gpus'] = [ gpu ]

        print('ssc-pimega: reading and restoring ' + diccopy['path'] + ' @ GPU no. {}'.format(gpu)) 

        uid, nimgs = ioSet_Backward540D( diccopy )

        print('ssc-pimega: file ' + diccopy['path'] + ' sent to: {}'.format(diccopy['outpath']) + '/ {} images @ uid: {}'.format(nimgs, uid)  ) 
        
        elapsed0 = time.time() - start

        for i in range( info.shape[1]-2 ):
            info[k, i] = ord( uid[i] ) 

        info[k, info.shape[1] - 2] = nimgs
        info[k, info.shape[1] - 1] = elapsed0


def _worker_io_restore_multiple_540D_(params):

    V = params[0]
    t = params[1]
    b = int( numpy.ceil(V/t) )

    size = len(str(uuid.uuid4()))

    info = _create_np_shared_array([V,size + 2], numpy.float32, ctypes.c_float)

    processes = []
    for k in range(t):
        begin_ = k*b
        end_   = min( (k+1)*b, V)

        gpu = params[2]['gpus'][k]

        p = multiprocessing.Process(target=_worker_io_batch_, args=(params, begin_, end_, info, gpu))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    return info


####

def ioSetM_Backward540D( dic ) : 
        
    """ Function to restore multiple H5 files using a superscalar pipeline
        Data is written at an specific folder given within dsc.
 
    Args:
        dic: input dictionary

    Returns:
        (None)
    """

    npaths = len(dic['path'])
    nproc = len( dic['gpus'])

    params = (npaths, nproc, dic)

    info = _worker_io_restore_multiple_540D_(params)

    return info


def ioGetM_Backward540D( dic, info, n) : 
        
    """ Function to restore multiple H5 files using a superscalar pipeline
        Data is written at an specific folder given within dsc.
 
    Args:
        dic: input dictionary
        k: kth file from given input list 
        info: information retrieved fom function <ioSetM_Backward540D>

    Returns:
        (None)
    """

    uid=""
    for k in range(info.shape[1]-2):
        uid += chr(int(info[n,k]))

    nimgs = int( info[n, info.shape[1] - 2 ] )

    output = ioGet_Backward540D( dic, uid, nimgs )

    return output


def ioCleanM_Backward540D( dic, info ):

    """ Function to clean previously restored multiple H5 files from a given folder.
    
    Args:
        dic: input dictionary
        info:  information retrieved from function <ioSetM_Backward540D>

    Returns:
        (None)

    """

    for n in range(info.shape[0]):
        uid=""
        for k in range(info.shape[1]-2):
            uid += chr(int(info[n,k]))

        nimgs = int( info[n, info.shape[1] - 2 ] )

        ioClean_Backward540D( dic, uid )

