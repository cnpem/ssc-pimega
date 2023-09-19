import os
import sys
import ctypes
import numpy
from ..pimegatypes import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob
import time

import multiprocessing
from PIL import Image
import sys

import functools
from functools import partial

from scipy import ndimage
import uuid
import SharedArray as sa


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


def set_suspicious_pixels( image, J, P, M, epsilon):

    SUSP135D = -10
    
    img = numpy.copy(image)
           
    p = epsilon
    
    if p > 0:

        #
        #remove columns (set -1)
        #
        for x in range(M-1):

            left = (x+1)*J - 1
            right = min( (x+1)*J, J*M )

            _from_ = left-p+1
            _to_   = left+1

            img[:,_from_: _to_] = SUSP135D

            _from_ = right
            _to_   = right+p-1
            
            img[:,_from_: _to_] = SUSP135D

            #print('-> left:', left-p+1,':', left)
            #print('-> right:', right,':', right+p-1)
            #print('')
            
        #
        #remove rows
        #
        for j in range(1,P):

            up   = j*J-1
            down = j*J

            _from_ = up-p+1
            _to_   = up+1
            
            img[_from_ : _to_, :]    = SUSP135D
            
            _from_ = down
            _to_   = down + p -1
            
            img[_from_ : _to_, :]    = SUSP135D
            
            #print('-> up:',up-p+1, ':', up)
            #print('-> down',down,':',down+p-1 )
            #print('')
            
        #
        #remove borders
        #
        
        img[0:p,:] = SUSP135D 
        img[:,0:p] = SUSP135D 
        img[img.shape[0]-p:img.shape[0],:] = SUSP135D
        img[:,img.shape[1]-p:img.shape[1]] = SUSP135D

        img[ img == SUSP135D ] = SUSP135D

        
    return img


def set_suspicious_pixels_block( image, J, P, M, epsilon):

    SUSP135D = -10
    
    img = numpy.copy(image)
           
    p = epsilon
    
    if p > 0:

        #
        #remove columns (set -1)
        #
        for x in range(M-1):

            left = (x+1)*J - 1
            right = min( (x+1)*J, J*M )

            _from_ = left-p+1
            _to_   = left+1

            img[:,:,_from_: _to_] = SUSP135D

            _from_ = right
            _to_   = right+p-1
            
            img[:,:,_from_: _to_] = SUSP135D

            #print('-> left:', left-p+1,':', left)
            #print('-> right:', right,':', right+p-1)
            #print('')
            
        #
        #remove rows
        #
        for j in range(1,P):

            up   = j*J-1
            down = j*J

            _from_ = up-p+1
            _to_   = up+1
            
            img[:,_from_ : _to_, :]    = SUSP135D
            
            _from_ = down
            _to_   = down + p -1
            
            img[:,_from_ : _to_, :]    = SUSP135D
            
            #print('-> up:',up-p+1, ':', up)
            #print('-> down',down,':',down+p-1 )
            #print('')
            
        #
        #remove borders
        #
        
        img[:,0:p,:] = SUSP135D 
        img[:,:,0:p] = SUSP135D 
        img[:,img.shape[0]-p:img.shape[0],:] = SUSP135D
        img[:,:,img.shape[1]-p:img.shape[1]] = SUSP135D

        img[ img == SUSP135D ] = SUSP135D

    return img


#################

def extract_with_pad( image, j):
    #size of reference image (medipix)
    J = 256
    j = 5 - j

    new = numpy.zeros(image.shape)
    new[j*J:j*J + J, :] = image[ j*J:j*J + J, :]

    return new

def extract_without_pad( image, j):
    #size of reference image (medipix)
    J = 256
    j = 5 - j

    return image[ j*J:j*J + J, :]


#################

def device( status, typedet ):
    
    shift = status  #True or False
    
    #vertical distance between bump bonds
    A = [ 113.8722, 55, -37.75 ]
    
    #horizontal distances bump bonds
    B = [114.1279, 55., 220.00]
    
    #gap between stripes: 1.711 mm
    if typedet=="nonplanar":
        delta = 1.71 * 1e+3 
    else:
        delta = 0
        
    #shift to pixel center
    st = A[0] - A[1]/4. #t axis (simulated values)
    sm = B[0] - B[1]/4. #m axis (simulated values)
        
    #device horizontal size: 85.478 mm
    T1 = 85.478 * 1e+3
    
    #gap intersection length: 0.2655 mm
    if typedet=="nonplanar":
        epsilon = 0.2655 * 1e+3
    else:
        epsilon = 0
        
    #stripe vertical length: 14.253 mm
    h = 14.253 * 1e+3
    
    return {'h': h, 
            'epsilon':epsilon,
            'delta': delta, 
            'T1': T1, 
            'A': A,
            'B': B,
            'st': st,
            'sm': sm,
            'shift': shift}
    
#################

def parameters(J, P, M, shift, typedet):

    const = device( shift, typedet )
    
    h       = const['h']
    epsilon = const['epsilon']
    delta   = const['delta']
    T1      = const['T1']
    A       = const['A']
    B       = const['B']
    
    ###########
    
    T2 = P * h - (P-1) * epsilon 
    
    ones = numpy.ones([J-1,])

    beta_border = B[0]
    beta      = numpy.hstack( ( 1, ones * B[1] ) )
    beta_bdry = B[2]
    beta[0]   = beta_border
    start     = 0 # - T1/2.0
    m         = numpy.array([])
    mref      = []
    m_bdry    = []         
    
    for x in range(M):
        mref.append(start)
        m_ = numpy.array([ start + sum(beta[0:k]) for k in range(1,len(beta)+1) ] )    
        m_bdry.append([m_[0], m_[-1]]) 
        m = numpy.hstack((m,m_))
        start   = m[len(m)-1]
        beta[0] = beta_bdry

    mref = numpy.array(mref)
    
    m_bdry = numpy.array(m_bdry).flatten()
    
    #
    #
    
    alpha_border = A[0]
    alpha      = numpy.hstack( (1, ones * A[1] ))
    alpha_bdry = A[2]
    alpha[0]   = alpha_border
    start      = 0 #- T2/2.0
    start_     = start
    t          = numpy.array([])
    tref       = []

    horLengthStripe = m[len(m)-1]-m[0] + 2*beta_border
    lengths = []
    t_bdry = []
    for y in range(P):
        tref.append(start_)
        t_ = numpy.array([ start + sum(alpha[0:k]) for k in range(1,len(alpha)+1) ] )
        t_bdry.append([t_[0],t_[-1]])
        t = numpy.hstack((t,t_))
        start    = t[len(t)-1]
        start_   = start + alpha_bdry - alpha_border
        alpha[0] = alpha_bdry
        
        verLengthStripe = t[J-1] - t[0] + 2*alpha_border  
        lengths.append([horLengthStripe, verLengthStripe ])
        
    tref = numpy.array(tref)

    t_bdry = numpy.array( t_bdry ).flatten()

    t_hexa = numpy.copy(t_)
   
    cores=['k','r','g','b','c','y','m']

    #shift to center position (theoretical modelling for the mesh)

    ct = t[ J*P//2 - 1 ]
    cm = m[ J*P//2 - 1 ]

    t = t - ct
    m = m - cm

    t_bdry = t_bdry - ct
    m_bdry = m_bdry - cm
         
    mref = mref - cm
    tref = tref - ct
    
    if const['shift']==True:
        
        u = numpy.zeros([J,])
        u[0]   =   const['st']
        u[J-1] = - const['st']
        T = []
        for x in range(M):
            T = numpy.hstack((T,u))

        u = numpy.zeros([J,])
        u[0]   =   const['sm']
        u[J-1] = - const['sm']
        U = []
        for y in range(P):
            U = numpy.hstack((U,u))

        t = t - T
        m = m - U
        
    #Normalization matrix (per stripe)
        
    flat = numpy.ones( [J, J] ) * A[1]
   
    A3 = A[0] + A[1]/2.
    B3 = B[0] + B[1]/2.

    flat[0,0] = A3 * B3
    flat[0,1:J-1] = A3 * B[1]/2.
    flat[0,J-1] = A3 * B3
    flat[1:J-1,0] = B3 * A[1]/2.
    flat[J-1,0] = B3 * A3
    flat[J-1, J-1] = B3 * A3
    flat[J-1,1:J-1] = A3 * B[1]/2.
    flat[1:J-1, J-1] = B3 * A[1]/2.
        
    norm = numpy.copy(flat)
    normChip = numpy.copy(flat)
    normChipx = numpy.copy(flat)
    for x in range(M-1):
        norm = numpy.hstack((flat,norm))
        normChipx = numpy.hstack((normChip, normChipx))
    
    normChipy = numpy.copy(flat)
    flat = numpy.copy( norm )
    for y in range(P-1):
        norm = numpy.vstack((norm, flat))
        normChipy = numpy.vstack((normChip, normChipy))
    
    return {'t': t,
            'm': m,
            't_hexa': t_hexa,
            't_bdry': t_bdry,
            'm_bdry': m_bdry,
            't_': t + ct,
            'm_': m + cm, 
            'tref':tref,
            'mref': mref,
            'stripe': lengths,
            'h': h,
            'epsilon': epsilon,
            'delta': delta, 
            'colours': cores,
            'J': J,
            'T1':T1,
            'T2':T2,
            'P': P,
            'M': M,
            'A': A,
            'B': B,
            'norm': [norm, normChip, normChipx, normChipy]}

#################

def get_project_values_geometry( *args ):

    if not args:
        ref = {'geo':'nonplanar','opt':True,'mode': 'virtual'}
    else:
        ref = args[0]

    if ref['geo'] == "nonplanar":

        #only virtual mode implemented

        if ref['opt'] == True:

            # to be done!
            
            a  = numpy.array( [0] )
            rx = numpy.array( 6 * [-6.75] )
            ry = numpy.array( 6 * [0] )
            rz = numpy.array( 6 * [0] )
            L  = numpy.array([ 0 ] )
            v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
            center = [0,0]
            
            ox  = numpy.array( 6 * [0] )
            oy  = numpy.array( 6 * [0] )
            
            x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center )) ) )

        else:

            a  = numpy.array( [0] )
            rx = numpy.array( 6 * [-6.75] )
            ry = numpy.array( 6 * [0] )
            rz = numpy.array( 6 * [0] )
            L  = numpy.array([ 0 ] )
            v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
            center = [0,0]
            
            ox  = numpy.array( 6 * [0] )
            oy  = numpy.array( 6 * [0] )
            
            x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center )) ) )

    else: #planar geometry

        if ref['opt'] == True:

            if ref['mode'] == "virtual":

                a  = numpy.array( [0] )
                rx = numpy.array( 6 * [0] )
                ry = numpy.array( 6 * [0] )
                rz = numpy.array( 6 * [0] )
                L  = numpy.array([ 0 ] )
                v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
                center = [0,0]
                
                ox  = numpy.array( 6 * [0] )
                oy  = numpy.array( 6 * [0] )
                
                x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center )) ) )
            else:
                # to be done!

                a  = numpy.array( [0] )
                rx = numpy.array( 6 * [0] )
                ry = numpy.array( 6 * [0] )
                rz = numpy.array( 6 * [0] )
                L  = numpy.array([ 0 ] )
                v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
                center = [0,0]
                
                ox  = numpy.array( 6 * [0] )
                oy  = numpy.array( 6 * [0] )
                gaps = numpy.array( [50, 50, 50, 50, 50, 0])

                x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,gaps)) ) )

                #print(x)
        else:

            a  = numpy.array( [0] )
            rx = numpy.array( 6 * [0] )
            ry = numpy.array( 6 * [0] )
            rz = numpy.array( 6 * [0] )
            L  = numpy.array([ 0 ] )
            v   = numpy.array([0,0,0]) #angle rotation of [0,0,1]
            center = [0,0]
            
            ox  = numpy.array( 6 * [0] )
            oy  = numpy.array( 6 * [0] )
            gaps = numpy.array( [50, 50, 50, 50, 50, 0])
            
            x = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center, gaps)) ) )

            
    return x


def dictionary135D( L0, *args ):

    """ Get default values for geometrical image restoration at the pimega/pi135D.

    Args:
        L0: distance sample to detector
        args: extra arguments

    Returns:
        (dict): Dictionary with informations about the geometrical setup 

    * The output dictionary is given below:

    .. code-block:: python 
        
       det = {
        'rotz': [_], 
        'roty': [_],
        'rotx': [_],
        'distance': L0,
        'normal': [_,_,_],
        'a': [_,_,_,_],
        'z': _,
        'offset':  [_] , 
        'ox': [_],
        'oy': [_],
        'shift': _,
        'typedet': _
        'mode': _,
        'gaps': [_] ,
        'hexa': _,
        'fill': _,
        'scale': _,
        'susp': _ 
        }

    * ``det['rotx']``, ``det['roty']``, ``det['rotz']`` 

        These are angle lists for each stripe. Each position of this list is related to a stripe, 
        The first position of this 6-length list relates to the bottom stripe, last position to the top 
        stripe. Each orthornomal basis :math:`\{n, n^\perp, n^{\times}\}` 
        is defined as a rotation of the vector (0,0,1) with respect to ``rotx``, ``roty`` and ``rotz``, 
        respectively. Angles are defined in degrees. 

    * ``det['distance']`` 

        The the input distance, from sample to the virtual (restored) image.

    * ``det['normal']``

        These are the angles (rx,ry,rz) to set the normal direction for the virtual image.
        The normal vector is defined as a rotation of the vector (0,0,1) with respect to ``rx``, ``ry`` 
        and ``rz``, respectively. Angles are defines in degrees.

    * ``det['a']``
    
        This is a scalar representing a rotation wrt to the z-axis for the complete
        module. Angles are defined in degrees.
    
    * ``det['offset']``

        These are offset (in the beam direction) lists for each stripe. Natural offsets are computed with respect to the input 
        distance L0; hence the values given here represent numerical deviations for these values. Offset values
        are defined in m.
        
    * ``det['ox']``, ``det['oy']``

        These are shift (in the orthogonal beam direction) list for each stripe. The first position of this 6-length list relates to the 
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
        ``[0,1,4]`` indicates that only hexa's 0, 1, 4  will be restored.
        On the other hand, ``range(6)`` indicates that everything will be restored.

    * ``det['mode']``

        Flag indicating ``real`` or ``vìrtual`` restored images. Pimega/135D only
        provide virtual images.

    * ``det['gaps']``
    
       There are 5 gaps that could be defined as input for the planar case. 
       For the nonplanar case, they are defined as a list full of zeros. For 
       instance, the following gap list define 1um as the gap distance between
       stripe 0 and 1, 2um between stripe 1 and 2, and so on. Last position is 
       arbitrary.

        .. code-block:: python

        'gaps': [ 1,2,3,4,5,0 ]

    * ``det['scale']``

        Scale factor (zoom-in / zoom-out )

    * ``det['susp']``

        Number of suspicious pixels to be removed for each Medipix/Chip
    
    * Extra arguments are given below. Once the geometrical optimization is 
      performed, the API include optimal values for offset, angles and gap. In 
      this case, the flat 'opt' will be 'True', otherwise 'False'.

    .. code-block:: python
  
       args[0] = {'geo': 'planar', 'opt': True, 'mode': 'real', 'x': xdet}

       xdet = numpy.array( list( numpy.hstack((a,rx,ry,rz,L,ox,oy,v,center,gaps)) ) )

    """
    
    ref = args[0]
    
    if 'x' in ref.keys():
        x = ref['x']
        if ref['geo'] == 'planar':
            r  = numpy.array( 6 * [0] )
            L  = numpy.array( 1 * [0] )
            v  = numpy.array( 3 * [0] )
            center = numpy.array( 2 * [0] )
            dic = ref['x']
            x = numpy.array( list(numpy.hstack((dic['a'],r,r,r,L,dic['ox'],dic['oy'],v,center,dic['gaps'])) ) ) 
        else:
            dic = ref['x']
            x = numpy.array( list(numpy.hstack((dic['a'],dic['rx'],dic['ry'],dic['rz'],dic['L'],dic['ox'],dic['oy'],dic['v'],dic['center'],dic['gaps'])) ))
    else:
        x = get_project_values_geometry( *args )
  
    
    a = x[0]
    r = x[1:19]
    L = x[19]
    o = x[20:32]
    v = x[32:35]
    center = x[35:37]
    
    ref = args[0]

    if 'crop' in ref.keys():
        crop = ref['crop']
    else:
        crop = True #only for planar detectors
        
    
    if 'scale' in ref.keys():
        z = ref['scale']
    else:
        z = 0.98

    if 'fill' in ref.keys():
        interp = ref['fill']
    else:
        interp = False

    if 'susp' in ref.keys():
        susp = ref['susp']
    else:
        susp = 0
    
    if ref['geo'] == "planar":
        gaps = x[37:43]
        typedet = "planar"
        mode    = ref['mode']
    else:
        gaps    = 6*[0]
        typedet = "nonplanar"
        mode    = ref['mode']

    if 'hexa' in ref.keys():
        hexa = ref['hexa']
    else:
        hexa = range(6)
       

    det = {
        'rotz':  r[12:18], 
        'roty':  r[6:12],
        'rotx':  r[0:6],
        'distance': L0,
        'normal': [v[0], v[1], v[2]],
        'a': a,
        'z': z,
        'symmetric': False,
        'offset':  L, 
        'ox':  o[0:6] ,
        'oy':  o[6:12] ,
        'gaps': gaps[:],
        'center': center,
        'typedet': typedet,
        'mode': mode,
        'hexa': hexa,
        'fill': interp,
        'susp': susp,
        'crop': crop,
        'input': ref
    }

    return det

#################

def detector_at_normal_plane(s1, s2, J, P, M, symbol, shift, typedet ):

    D = parameters( s1, s2, J, P, M, shift, typedet )
    
    t = D['t']
    m = D['m']
    tt = D['tref']
    mm = D['mref']
    lengths = D['stripe']
    
    for j in range(P):
        t = D['t'][j*J:min(J*P,(j+1)*J)]

        M,T = numpy.meshgrid(m,t)

        plt.plot(M,T,'{}'.format(D['colours'][j]+symbol))
        plt.plot(mm[0]*numpy.ones(D['P']),tt,'kx')
        plt.plot(mm[0],tt[j]+lengths[j][1],'ks')
        
        plt.plot(D['s1'],D['s2'],'s')
        
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((mm[0],tt[j]),D['stripe'][j][0],D['stripe'][j][1], fill=None, alpha=1))

        
######################

def project_virtual_det(m,t,n,nperp,ncross,Lj,L,e3, Nz):
    #point over stripe
    p0 = Nz[0] - Lj * n[0] + t * ncross[0] + m * nperp[0]
    p1 = Nz[1] - Lj * n[1] + t * ncross[1] + m * nperp[1]
    p2 = Nz[0] - Lj * n[2] + t * ncross[2] + m * nperp[2]
    
    den = (Nz * e3).sum() - Lj * ((n * e3).sum()) + t * ((ncross*e3).sum()) + m * ((nperp*e3).sum())
    
    #projected points: grid 
    const = (-L/den)
    x = p0 * const
    y = p1 * const
    z = p2 * const
    
    return x,y,z


def detector_at_virtual_plane(J,P,M, RX, RZ, RY, L, normal, shift, symbol, plot, Ox, Oy, center, typedet):

    e3 = numpy.array( build_normal( normal ) ).reshape([3,1])
    #e3 =(numpy.array([normal[0],normal[1],normal[2]]).reshape([3,1]))
    
    L = L * 1000 #(microns)
    
    D = parameters(J, P, M, shift, typedet )
    
    t       = D['t']
    m       = D['m']    
    delta   = D['delta']
    colours = D['colours']
    lengths = D['stripe']
    
    ###
    
    bottomleft = numpy.zeros([D['P'],2])
    topright   = numpy.zeros([D['P'],2])
    bottomright= numpy.zeros([D['P'],2])
    topleft    = numpy.zeros([D['P'],2])
     
    stripes = []

    Ljs = numpy.zeros([P,])
    
    nperp_list =  []
    ncross_list = []
    n_list      = []
    
    m_module = D['m']
    t_module = D['t']
    m_bdry   = D['m_bdry']
    t_bdry   = D['t_bdry']

    z = numpy.array([0,0,0]).reshape([3,1]) 
    
    for j in range(P):
        #################################
        # normal detector to each stripe
        
        n = numpy.array([0,0,1]).reshape([3,1])
        
        RX_ = (RX[j]*numpy.pi/180.)
        RX_array = numpy.array([[1,0,0],[0,numpy.cos(RX_), -1*numpy.sin(RX_)],[0,numpy.sin(RX_),numpy.cos(RX_)]])
        
        #
        RZ_ = (RZ[j]*numpy.pi/180.)
        RZ_array = numpy.array([[numpy.cos(RZ_), -1*numpy.sin(RZ_), 0],[numpy.sin(RZ_),numpy.cos(RZ_), 0],[0,0,1]])

        RY_ = (RY[j]*numpy.pi/180.)
        RY_array = numpy.array([[numpy.cos(RY_), 0, numpy.sin(RY_)],[0,1,0],[-numpy.sin(RY_),0,numpy.cos(RY_)]])

        n = numpy.dot(RY_array,n)
        n = numpy.dot(RZ_array,n)
        n = numpy.dot(RX_array,n)

        nperp = numpy.array([[n[2,0]],[0],[-n[0,0]]])

        ncross = numpy.array([[ -n[1,0]*n[0,0]],[ n[2,0]**2+n[0,0]**2],[-n[1,0]*n[2,0]]])

        dpnz = n[0] * z[0] + n[1] * z[1] + n[2] * z[2]  
        Nz = numpy.array([ n[0] * dpnz , n[1] * dpnz , n[2] * dpnz ]).reshape([3,1])

        ############
        
        ncross_list.append( ncross )
        nperp_list.append( nperp )
        n_list.append( n )
        
        ###########
        
        Lj = L / numpy.cos( numpy.abs(RX_) ) - j * delta
        
        Ljs[j] = Lj
      
        ##
        sx    =  center[0] * 55
        sy    =  center[1] * 55
        Gz    = [ (n[0]*n[0] + 1)*z[0] + n[0]*n[1]*z[1] + n[0]*n[2]*z[2], n[1]*n[0]*z[0] + (n[1]*n[1]+1)*z[1] + n[1]*n[2]*z[2] ] 
        px    = Gz[0] - Lj * n[0]
        py    = Gz[1] - Lj * n[1]
        det   = ncross[0] * nperp[1] - ncross[1] * nperp[0] 
        tref  = 1*(( nperp[1] * ( sx - px )  - nperp[0] * (sy - py) ) / det)
        mref  = 1*((-ncross[1] * ( sx - px ) + ncross[0] * (sy - py) ) / det)
        
        m = m_module + mref + Ox[j]
        t = t_module[j*J: min( J*P,(j+1)*J )] + tref + Oy[j]
        
        M_,T_ = numpy.meshgrid( m, t )

        #grid points over the virtual detector
        xMesh, yMesh, _ = project_virtual_det(M_,T_,n,nperp,ncross,Lj,L,e3, Nz)
        
        #----------------------
        #stripe boundary meshes

        M_ht,T_ht = numpy.meshgrid( m, numpy.array([ t[0] ]) )  #horizontal/top
        M_hb,T_hb = numpy.meshgrid( m, numpy.array([ t[len(t)-1] ]) ) #horizontal/bottom
        
        M_vl, T_vl = numpy.meshgrid( m_bdry[0:-1:2] + mref + Ox[j],  t ) #vertical/left
        M_vr, T_vr = numpy.meshgrid( m_bdry[1:-1:2] + mref + Ox[j],  t ) #vertical/right
        
        xMesh_ht, yMesh_ht, _ = project_virtual_det(M_ht,T_ht,n,nperp,ncross,Lj,L,e3, Nz)
        xMesh_hb, yMesh_hb, _ = project_virtual_det(M_hb,T_hb,n,nperp,ncross,Lj,L,e3, Nz)

        xMesh_vl, yMesh_vl, _ = project_virtual_det(M_vl,T_vl,n,nperp,ncross,Lj,L,e3, Nz)
        xMesh_vr, yMesh_vr, _ = project_virtual_det(M_vr,T_vr,n,nperp,ncross,Lj,L,e3, Nz)
        
        #corner: top/left
        x, y, _ = project_virtual_det( m[0],t[len(t)-1],n,nperp,ncross,Lj,L,e3,Nz)
        topleft[j,:] = [float(x),float(y)]

        #corner: bottom/left
        x, y, _ = project_virtual_det( m[0],t[0],n,nperp,ncross,Lj,L,e3,Nz)
        bottomleft[j,:] = [float(x),float(y)]

        #corner: bottom/right
        x, y, _ = project_virtual_det( m[len(m)-1],t[0],n,nperp,ncross,Lj,L,e3,Nz)
        bottomright[j,:] = [float(x),float(y)]

        #corner: top/right
        x, y, _ = project_virtual_det( m[len(m)-1],t[len(t)-1],n,nperp,ncross,Lj,L,e3,Nz)
        topright[j,:] = [float(x),float(y)]

        stripes.append( [  xMesh,  yMesh, xMesh_ht, yMesh_ht, xMesh_hb, yMesh_hb, xMesh_vl, yMesh_vl, xMesh_vr, yMesh_vr   ] ) 

        if plot:
            plt.plot(xMesh,yMesh,'{}'.format(colours[j]+symbol) )
            
            plt.plot(bottomleft[j,0], bottomleft[j,1], 'ko', topright[j,0], topright[j,1], 'ko')
            plt.plot(bottomright[j,0], bottomright[j,1], 'ko', topleft[j,0], topleft[j,1], '{}'.format(colours[j]+'o'))
            
            plt.xlabel('x')
            plt.ylabel('y')
    
    minx = min( bottomleft[:,0].min(), topleft[:,0].min() )
    maxx = max( bottomright[:,0].max(), topright[:,0].max() )
    
    miny = min( bottomleft[:,1].min(), bottomright[:,1].min() )
    maxy = max( bottomright[:,1].max(), topright[:,1].max() )
    
    xbox = [ minx, maxx, maxx-minx ]
    ybox = [ miny, maxy, maxy-miny ]

    ###
    
    if plot:
        plt.plot( xbox[0], ybox[0], 'sr' )
        plt.plot( xbox[1], ybox[1], 'sr' )
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((xbox[0],ybox[0]),xbox[2],ybox[2], fill=None, alpha=1))
    
    D['bottomright'] = bottomright
    D['bottomleft'] = bottomleft
    D['topright'] = topright
    D['topleft'] = topleft
    D['ybox'] = ybox
    D['xbox'] = xbox
    D['stripes'] = stripes
    D['L'] = L
    D['n'] = n_list
    D['nperp'] = nperp_list
    D['ncross'] = ncross_list
    D['Ljs'] = Ljs
    
    return D


#################


def pointcloud_virtualplane_135D( params ):
   
    J = 256
    P = 6
    M = 6
        
    L = params['distance']
    susp = params['s']
    v = params['normal'][1]
    a  = params['a']
    z  = params['z']
    dL = params['offset']
    offsetx = params['ox']
    offsety = params['oy']
    center = params['center']
    typedet = params['typedet']
    
    if typedet=="nonplanar":
        RZ = numpy.array( params['rotz'] )
        RY = numpy.array( params['roty'] )
        RX = numpy.array( params['rotx'] )
    else:
        RZ = numpy.zeros([6,1])
        RY = numpy.zeros([6,1])
        RX = numpy.zeros([6,1])
        

    ####

    NOTHREADS=24
    symbol="*"
    plot=True
    shift=False
    
    #xbox, ybox = get_bounding_box(v, [gx, gy, hx, hy], RxM, RyM, RzM, L, dL, a )
    
    detector_at_virtual_plane(J,P,M, RX, RZ, RY, L, v, shift, symbol, plot, offsetx, offsety, center, typedet )

    plt.show()

    return True


##################

def build_geometry_135D(J,P,M,L,normal,Rx,Rz,Ry, Ox, Oy, center, typedet, z):

    start = time.time()

    shift=False
    plot=False
    symbol="*"
    D = detector_at_virtual_plane(J,P,M,Rx,Rz,Ry,L,normal,shift,symbol, plot, Ox, Oy, center, typedet)
    
    #print('Generating mesh/strip: {} sec'.format(round(time.time()-start,3)))
    
    #
    HL = J*M
    VL = J*P
    
    ybox = D['ybox']
    xbox = D['xbox']
    L    = D['L']  #take the correct L (in microns, not the input in mm)

    #xbox & ybox (microns)
    dx = ((xbox[1] - xbox[0])) / (HL - 1 )
    dy = ((ybox[1] - ybox[0])) / (VL - 1 )
    
    #pixelsize @ virtual detector
    step = max( dx, dy )
    
    #millimeters (pixel size)
    medipix = 55
    
    SCALE_FACTOR = z * (medipix / min(dx, dy))    
    
    xx = (SCALE_FACTOR) * numpy.array([xbox[0] + k * step for k in range(HL)]) 
    yy = (SCALE_FACTOR) * numpy.array([ybox[0] + k * step for k in range(VL)])
    
    ########
    
    interp = []

    gaps = 0
    bdry_hor = 0
    bdry_ver = 0
    
    stripes = D['stripes']

    ixGpu = numpy.zeros([6,256,1536], dtype=numpy.int32)
    iyGpu = numpy.zeros([6,256,1536], dtype=numpy.int32)

    start = time.time()
    
    for j in range(P):    
        
        xMesh     = stripes[j][0]
        yMesh     = stripes[j][1]
        xMesh_ht  = stripes[j][2]
        yMesh_ht  = stripes[j][3]
        xMesh_hb  = stripes[j][4]
        yMesh_hb  = stripes[j][5]
        xMesh_vl  = stripes[j][6]
        yMesh_vl  = stripes[j][7]
        xMesh_vr  = stripes[j][8]
        yMesh_vr  = stripes[j][9]
        
        #
        #nearest neighbour interp: assuming a regular mesh at the device
        #        
        ix = ( numpy.floor( (xMesh - xx[0]) /(xx[1]-xx[0]) )).astype(int)
        iy = VL - ( numpy.floor( (yMesh - yy[0]) /(yy[1]-yy[0]) )).astype(int) 
        
        mix0 = ( ix < 0 )
        mix1 = ( ix >= HL - 1)
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
        
        interp.append( [iy, ix, iy_ht, ix_ht, iy_hb, ix_hb, iy_vl, ix_vl, iy_vr, ix_vr ])

        ixGpu[j][:,:] = ix
        iyGpu[j][:,:] = iy
        
        #
        #
        
        iyP1 = iy + 1
        iyM1 = iy - 1
        ixP1 = ix + 1
        ixM1 = ix - 1

        iyP1[ iyP1 >= VL -1 ] = VL-1
        iyM1[ iyM1 < 0 ] = 0
        ixP1[ ixP1 >= HL -1 ] = HL-1
        ixM1[ ixM1 < 0 ] = 0
        
        z = numpy.zeros([ VL, HL] )
        z[iy, ix] = 1
        
        z[iyP1, ix] = 1
        z[iyM1, ix] = 1
        z[iy, ixP1] = 1
        z[iy, ixM1] = 1
        
        gaps += z

        '''
        zz = numpy.zeros([ VL, HL] )
        zz[iy_ht, ix_ht] = 1
        bdry_hor += zz
        
        zzz = numpy.zeros([ VL, HL] )
        zzz[iy_vl, ix_vl] = 1
        zzz[iy_vr, ix_vr] = 1
        bdry_ver += zzz
        '''
        
        ########

    #print('Generating LUT(index)/strip: {} sec'.format(round(time.time()-start,3)))

    ### mask update
        
    start = time.time()

    mask = (gaps == 2).astype(numpy.double)

    struct = ndimage.generate_binary_structure(2,1)
    
    mask = ndimage.binary_dilation(mask, structure=struct).astype(mask.dtype)

    #print('Dilation: {} sec'.format( round( time.time()-start,3 )))    

    ##################
    
    geometry = {
        'interp': interp,
        'LUT': [iyGpu, ixGpu],
        'geom': D,
        'pxlsize': (xx[1] - xx[0]), #microns,
        'Nx': HL,
        'Ny': VL,
        'overlap': mask,
        'P': 6,
        'M': 6,
        'J': 256
        #'bdry': [ bdry_hor, bdry_ver ]
         }
    
    return geometry


def build_geometry_135D_planar(Ox, Oy, Gaps):
    
    H              = 1536
    J              = 256
    ASIC_BUMP_BOND = 3

    def get_index_withgap(end):
        gapchip = ASIC_BUMP_BOND
        arr = []
        for k in range(6):
            start_ = end - 256 * (6-k) - (5 - k) * gapchip
            chip   = numpy.arange(start_,start_ + 256)
            arr.append( chip )    
        return numpy.array(arr).flatten()

    def get_shape_pi135D_planar( ox, oy, g, cr, cc):

        _fromto_ = []
    
        _row_ = []
        _col_ = []
        
        for k in range(6):
            _start_ = cr - (k+1)*J - g[0:k].sum() - oy[k]
            _end_   = _start_ + J
            idx     = get_index_withgap(cc + ox[k] + H)

            _row_.append([_start_,_end_])
            _col_.append([idx.min(), idx.max()])
        
        _row_ = numpy.array(_row_, dtype=int).flatten()
        _col_ = numpy.array(_col_, dtype=int).flatten()
    
        nrows = max(_row_) - min(_row_)
        ncols = max(_col_) - min(_col_)

        rowv = [ min(_row_), max(_row_) ]
        colv = [ min(_col_), max(_col_) ]
    
        return (nrows,ncols), rowv, colv

    N = 4000 #adjust: TODO
    cr = N
    cc = N//2
    
    shape, rowv, colv = get_shape_pi135D_planar(Ox, Oy, Gaps, cr, cc)

    new = -1 * numpy.ones( [N,N])
    
    LUT = ( numpy.zeros([6,J,H]), numpy.zeros([6,J,H]) ) 
    
    for k in range(6): #six 'hexa' modules

        _start_ = cr - (k+1)*J - Gaps[0:k].sum() - Oy[k]
        _end_   = _start_ + J
    
        #module
        _ix_ = get_index_withgap(cc + Ox[k] + H) 
        _iy_ = numpy.arange(_start_,_end_)
        [ix,iy] = numpy.meshgrid(_ix_, _iy_)
        LUT[0][k] = ix
        LUT[1][k] = iy
        
    geometry = {}
    geometry['geom'] = {}
    geometry['geom']['LUT'] = [ LUT[0], LUT[1] ]
    geometry['geom']['shape'] = [N, shape, rowv, colv]
    geometry['geom']['P'] = 6
    geometry['geom']['M'] = 6
    geometry['geom']['J'] = 256
    
    return geometry


################

def build_normal( tilt ):

    tilt_ = numpy.array(tilt).flatten()
    
    RX_ = (tilt[0]) * numpy.pi / 180.
    RY_ = (tilt[1]) * numpy.pi / 180.
    RZ_ = (tilt[2]) * numpy.pi / 180.
    
    e3 = numpy.array([0,0,1]).reshape([3,1])

    RX_array = numpy.array([[1,0,0],[0,numpy.cos(RX_), -1*numpy.sin(RX_)],[0,numpy.sin(RX_),numpy.cos(RX_)]])    
    RZ_array = numpy.array([[numpy.cos(RZ_), -1*numpy.sin(RZ_), 0],[numpy.sin(RZ_),numpy.cos(RZ_), 0],[0,0,1]])
    RY_array = numpy.array([[numpy.cos(RY_), 0, numpy.sin(RY_)],[0,1,0],[-numpy.sin(RY_),0,numpy.cos(RY_)]])
    
    e3 = numpy.dot(RX_array,e3)
    e3 = numpy.dot(RY_array,e3)
    e3 = numpy.dot(RZ_array,e3)    
    
    return e3.flatten()


################

def correct_image_backward_135D(frame, geometry):

    SUSP135D = -10
    
    J = geometry['geom']['J']
    P = geometry['geom']['P']
    M = geometry['geom']['M']

    frame2 = numpy.flipud( frame )
    
    new = -1 * numpy.ones(frame2.shape)
    
    for j in geometry['hexa']:
        
        iy, ix, _, _, _, _, _, _, _, _ = geometry['interp'][j]
        
        stripe = frame2[j*J: j*J + J, :]
        
        new[iy, ix] = stripe

        #improving interpolation for missing values
        if geometry['fill'] == True:
            
            projectedHexa = new[ iy.min():iy.max(), ix.min():ix.max() ]
            
            missing = (projectedHexa == -1)
            
            projectedHexa2 = interpolate_missing_pixels(projectedHexa, missing, 'nearest', SUSP135D)
                    
            new[ iy.min():iy.max(), ix.min():ix.max()] = projectedHexa2


    new[ new < 0 ] = -1
            
    return new

def correct_image_backward_135D_planar(frame, geometry):
         
    J = geometry['geom']['J']
    P = geometry['geom']['P']
    M = geometry['geom']['M']

    N, shape, rowv, colv = geometry['geom']['shape']

    #print(N, shape, rowv, colv)
    
    new = -1 * numpy.ones([N,N])
    
    for j in geometry['hexa']:
    
        ix = geometry['geom']['LUT'][0][j].astype(int)
        iy = geometry['geom']['LUT'][1][j].astype(int)
        
        s      = 5 - j
        stripe = frame[s*J: (s+1)*J, :]
        new[iy, ix] = stripe

    if geometry['crop'] == True:
        arr = new[ rowv[0]:rowv[1], colv[0]:colv[1] ]
    else:
        arr = new
        
    arr[ arr < 0 ] = -1
    
    return arr


#################

def correct_image_forward_135D(frame, geometry):

    J = 256
    P = geometry['geom']['P']
    M = geometry['geom']['M']
    HL = J*M
    VL = J*P

    new     = numpy.zeros( [VL, HL])
    new_gap = numpy.zeros( [VL, HL])
    
    gaps = geometry['overlap']

    for j in geometry['hexa']:
    
        iy, ix, _, _, _, _, _, _, _, _  = geometry['interp'][j]
                
        nearest = frame[ iy, ix ]

        nearest_gap = gaps[iy, ix]

        #-------------------------------------------------
        # assuming that overlap will not exceed half strip
        nearest_gap[0:J//2,:] = 0
        #-------------------------------------------------
        
        new[j*J: j*J+J , :] = ( nearest )
        
        new_gap[j*J: j*J+J, :] = ( nearest_gap )
        
    new [ new_gap == 1 ] = -1
        
    return numpy.flipud( new )


#################

def correct_image_forward_135D_planar(frame, geometry):

    def squarify(M,val):
        (a,b)=M.shape
        if a>b:
            padding=((0,0),(0,a-b))
        else:
            padding=((0,b-a),(0,0))

        return numpy.pad(M,padding,mode='constant',constant_values=val)
        
    J = 256
    P = geometry['geom']['P']
    M = geometry['geom']['M']

    #forcing susp=0 for forward simulation
    geometry['susp'] = 0
    
    back = backward135D ( numpy.ones([1536,1536]), geometry )

    mask = (back < 0)

    signal = mask[:,10]
    signal[0:5] = 0
    signal[len(signal)-5:len(signal)] = 0

    ridx = numpy.argwhere( signal == 0 )

    signal = mask[10,:]
    signal[0:5] = 0
    signal[len(signal)-5:len(signal)] = 0

    cidx = numpy.argwhere( signal == 0 )

    cy = back.shape[0]//2
    cx = back.shape[1]//2
    back[cy - 768 : cy + 768, cx - 768 : cx + 768 ] = frame
    back[mask]  = -1

    r,c = numpy.meshgrid(ridx, cidx)
    tmp = back[r,c].T
    tmp = squarify(tmp, 0)

    new = numpy.zeros([1536, 1536])

    er = int( (1536 - tmp.shape[0])/2 )
    ec = int( (1536 - tmp.shape[1])/2 )

    new[er:er+tmp.shape[0], ec:ec+tmp.shape[1]] = tmp
        
    return new


#################

def move_center( img, hor, ver ):
    translation = numpy.copy(img)
    translation = numpy.roll(translation, ver, 0)
    translation = numpy.roll(translation, hor, 1)
    return translation

#################

def get_geometry135D( params, *args ):

    J = 256
    P = 6
    M = 6
    
    susp = params['susp']
    z = params['z']
    L = params['distance']
    RZ = params['rotz']      
    RY = params['roty']
    RX = params['rotx']
    v =  params['normal']
    Ox = params['ox']
    Oy = params['oy']
    center = params['center']
    typedet = params['typedet']
    
    geometry = build_geometry_135D(J, P, M, L, v, RX, RZ, RY, Ox, Oy, center , typedet, z)

    geometry['hexa'] = params['hexa']
    geometry['mode'] = params['mode']
    geometry['typedet'] = params['typedet']
    geometry['fill'] = params['fill']
    geometry['susp'] = params['susp']
    geometry['crop'] = params['crop']
    
    return geometry

def get_geometry135D_planar( params, *args ):
    
    Ox    = numpy.array( params['ox'] )
    Oy    = numpy.array( params['oy'] )
    Gaps  = numpy.array( params['gaps']  )

    geometry = build_geometry_135D_planar( Ox, Oy, Gaps )
    
    geometry['susp'] = params['susp']        
    geometry['typedet'] = params['typedet']
    geometry['mode'] = params['mode']
    geometry['hexa'] = params['hexa']
    geometry['crop'] = params['crop']
    
    return geometry

def geometry135D ( params, *args ):

    """ Function to compute prior geometrical information related to planar and nonplanar pimega/135D detector. 
    
    Args:
        params: input parameters 
        args: extra arguments

    Returns:
        (dict): Geometrical information
    
    """
     
    if params['typedet'] == "nonplanar":
        
        geometry = get_geometry135D( params, *args )

    elif params['typedet'] == "planar":
        
        if params['mode'] == "virtual":

            # 04/Jan/2023
            # # temporary solution: remove virtual images for real/planar 135D
            # E.Miqueles

            geometry = get_geometry135D_planar( params, *args )
        else:
            
            geometry = get_geometry135D_planar( params, *args )
            
    return geometry


#################

def get_backward135D(  matrix, geometry ):

    if matrix.shape[0]!=1536 or matrix.shape[1]!=1536:
        print('Input error: go find a 1536 x 1536 image!')
        return None
    else:
        J    = 256
        P    = geometry['geom']['P']
        M    = geometry['geom']['M']
        susp = geometry['susp']

        new = set_suspicious_pixels( matrix, J, P, M, susp )        
        new = correct_image_backward_135D( new , geometry)
    
        return new

def get_backward135D_planar(  matrix, geometry ):

    if matrix.shape[0]!=1536 or matrix.shape[1]!=1536:
        print('Input error: go find a 1536 x 1536 image!')
        return None
    else:
        J    = 256
        P    = geometry['geom']['P']
        M    = geometry['geom']['M']

        susp = geometry['susp']
        new = set_suspicious_pixels( matrix, J, P, M, susp )
        new = correct_image_backward_135D_planar( new , geometry)
        
        return new    

def backward135D( matrix, geometry):

    """ Function to restore a given frame using a measured pimega/pi135D data
    
    Args:
        matrix: digital 1536x1536 measured matrix from pimega/pi135D.
        geometry: geometrical data related to pimega/pi135D (see ``dictionary135D()``). 

    Returns:
        (ndarray): restored matrix
    
    """
    
    if geometry['typedet'] == "nonplanar":
        
        new = get_backward135D(matrix, geometry )

    else:
        
        if geometry['mode'] == "virtual":

            # 04/Jan/2023
            # temporary solution: zoom out virtual/planar images for the backend visualizer
            # E.Miqueles

            #forcing to be real/planar
            geometry['mode'] = "real"

            new_real = get_backward135D_planar( matrix, geometry )

            #zoom out
            from scipy import interpolate

            def zoom_feature(img,a):
                d = abs(img.shape[1] - img.shape[0])
                if min(img.shape[1], img.shape[0]) == img.shape[0]:
                    img = numpy.vstack((img, -1 * numpy.ones([d,img.shape[1]] )))
                else:
                    img = numpy.hstack((img, -1 * numpy.ones([img.shape[0],d] )))

                xx = numpy.linspace(-1.0,1.0,max(img.shape[1], img.shape[0]) )
                yy = numpy.linspace(-1.0,1.0,max(img.shape[1], img.shape[0]))
                fun =  interpolate.RectBivariateSpline(xx, yy, img, kx=1, ky=1, s=0)
                x = numpy.linspace(-a,a,1536)
                y = numpy.linspace(-a,a,1536)
                return fun(x,y)

            new = zoom_feature( new_real, 1)

        else:

            new = get_backward135D_planar( matrix, geometry )

    return new
    
##################
    
def forward135D( matrix, geometry ):

    """ Function to simulate a given frame using Pi135D with a simulated image
    
    Args:
        matrix: digital 1536x1536 input matrix.
        geometry: geometrical data related to Pimega/Pi135D. See function geometry135D(). 

    Returns:
        (ndarray): (simulated) measured matrix
    
    """

    if matrix.shape[0]!=1536 or matrix.shape[1]!=1536:
        print('ssc-pimega error: go find a 1536 x 1536 image!')
        return None
    else:    
        if geometry['typedet'] == "nonplanar":

            new = correct_image_forward_135D( matrix , geometry)

        else:
            new = correct_image_forward_135D_planar( matrix , geometry)

    return new

#############

def polar2cart(r, theta):
    return r*numpy.cos(theta), r*numpy.sin(theta)

def cart2polar(x, y):
    return numpy.arctan2(y,x), numpy.sqrt(x**2 + y**2)

def toPolar(cart,ac,ap,R,V):
    n       = cart.shape[0]
    dxy     = (2*ac)/float(n-1)
    m_angle = numpy.linspace(0,2*numpy.pi,V)
    m_ray   = numpy.linspace(0,ap,R)
    theta,R = numpy.meshgrid(m_angle, m_ray)
    X,Y     = polar2cart(R, theta)
    iX      = ((X+ac)/dxy).astype(int)
    iY      = ((Y+ac)/dxy).astype(int)

    iX[ iX >= n] = n-1
    iX[ iX < 0 ] = 0
    iY[ iY >= n] = n-1
    iY[ iY < 0]  = 0
    
    #nearest neighbourhood
    nearest = cart[iY,iX]
    polar   = nearest

    return polar

def toCartesian(polar,ac,ap,n):

    rays    = polar.shape[0]
    views   = polar.shape[1]
    dth     = (2*numpy.pi)/float(views-1)
    dR      = ap/float(rays-1)
    xx      = numpy.linspace(-ac,ac,n)
    x,y     = numpy.meshgrid(xx,xx)
    Theta,R = cart2polar(x,numpy.flipud(y))
    iTheta  = numpy.fliplr( ((Theta + numpy.pi)/dth).astype(int) )
    iR      = ((R)/dR).astype(int)

    iTheta[ iTheta >= views ] = views-1
    iTheta[ iTheta <  0 ]     = 0
    iR[ iR >= rays ] = rays-1
    iR[ iR <  0 ]    = 0
    
    #nearest neighbourhood
    nearest = polar[iR, iTheta] 
    cart    = nearest
    return cart

