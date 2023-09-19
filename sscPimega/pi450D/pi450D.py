import os
import sys
import ctypes
import numpy
import time
import gc
import h5py

import math
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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

from skimage.measure import EllipseModel, LineModelND
from scipy.optimize import minimize

from scipy import ndimage
#from skimage.morphology import skeletonize
#from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.morphology import convex_hull_image

import pickle

MEDIPIX    = 55.0



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


def device450D( status, typedet ):

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

def parameters450D(J, P, M, shift, typedet):
    
    const = device450D( shift, typedet )

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

    ct = t[ J*P//2 ]
    cm = m[ J*P//2 ]

    t = t - ct
    m = m - cm

    t_bdry = t_bdry - ct
    m_bdry = m_bdry - cm

    mref = mref - cm
    tref = tref - ct

    if const['shift'] == True:

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

        
    ## rearranging for the 450D with no guard-rings
    ## there are no large pixels on the boundary, only within stripes (hexa) 
        
    t[0]  = t[1] - 55
    t[J-1] = t[J-2] + 55
    m[0]  = m[1] - 55
    m[M*J-1]  = m[M*J-2] + 55
        
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

 

def project_virtual_det(m,t,n,nperp,ncross,f,v,L):
    #point over stripe
    q0 = f[0] + t * ncross[0] + m * nperp[0]
    q1 = f[1] + t * ncross[1] + m * nperp[1]
    q2 = f[2] + t * ncross[2] + m * nperp[2]

    den = ((f * v).sum()) + t * ((ncross*v).sum()) + m * ((nperp*v).sum())
    
    #projected points: grid 
    const = (L/den)
    x = q0 * const
    y = q1 * const
    z = q2 * const
    
    return x,y,z

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


def get_image_stripe_detector( I, mod, s):
    if s > 0:
        return I[0:256, mod * 1536: (mod+1)*1536]
    else:
        return I[256:512, mod * 1536: (mod+1)*1536]

def get_block_stripe_detector( I, mod, s):
    if s > 0:
        return I[:,0:256, mod * 1536: (mod+1)*1536]
    else:
        return I[:,256:512, mod * 1536: (mod+1)*1536]

    
def set_image_stripe_detector( I, hexa, mod, s):
    if s > 0:
        I[0:256, mod * 1536: (mod+1)*1536] = hexa
    else:
        I[256:512, mod * 1536: (mod+1)*1536] = hexa
    return I
    

def set_suspicious_pixels_450D( image, epsilon):

    SUSP450D = -10 
    J = 256
    M = 6
    
    img = numpy.copy(image)
    
    #
    #remove columns
    #
    if epsilon > 0:
        for x in range(M-1):

            left = (x+1)*J-1
            right = min( (x+1)*J, J*M-1 )

            img[:,left - epsilon: left+1] = SUSP450D
            img[:,right: right + epsilon+1] = SUSP450D

        return img
    else:
        return img


def plot_colortable(colors, *, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    #fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    #fig.subplots_adjust(margin/width, margin/height,
    #                   (width-margin)/width, (height-margin)/height)
    #ax.set_xlim(0, cell_width * 4)
    #ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    #ax.yaxis.set_visible(False)
    #ax.xaxis.set_visible(False)
    #ax.set_axis_off()

    count = 0
    colour_names = []
    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        if i % 7 == 1:
            colour_names.append(name)
            #print(name)
            #count += 1
        
        #ax.text(text_pos_x, y, name, fontsize=14,
        #        horizontalalignment='left',
        #        verticalalignment='center')

        #ax.add_patch(
        #    Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
        #              height=18, facecolor=colors[name], edgecolor='0.7')
        #)
    
    return colour_names

    
def get_bounding_box(dic,plot):

    if plot:
        cnames = plot_colortable(mcolors.CSS4_COLORS)

    D = dic['parameters']
        
    tmesh450D = D['t_'][0:256]
    mmesh450D = D['m_']
    
    theta = numpy.array( [ dic['aperture'] * k / 20.0 + dic['aperture']/40.0 for k in range(20) ] ) - dic['aperture']/2.0 + numpy.pi/2.0
    theta = numpy.flipud( theta ) #clockwise orientation, according to DET/Pitec
    
    Lf = 1.01 * dic['earc']

    T = (tmesh450D.max() - tmesh450D.min())/2.0
    M = (mmesh450D.max() - mmesh450D.min())/2.0

    def ud(s):
        if s==0:
            return 'up'
        else:
            return 'down'
    
    mods = numpy.array(range(20)) % 2
    mod = numpy.zeros([20,])
    for k in range(10):
        mod[2*k] = k
        mod[2*k+1] = k

    _ryhead_ = dic['angle']

    v = numpy.array([0,0,1]).reshape([3,1])

    xvalues = []
    yvalues = []

    if plot:
        fig1, ax1 = plt.subplots(figsize=(20,20))
        fig2, ax2 = plt.subplots(figsize=(20,20))

    stripes = []
        
    for k in range(len(theta)): 

        g = numpy.array([ dic['earc'] * numpy.cos(theta[k] + _ryhead_), 0, dic['earc'] * numpy.sin(theta[k]+_ryhead_) ] ).reshape([3,1])

        rx = dic['rotx'][k]
        ry = dic['roty'][k]
        rz = dic['rotz'][k]

        RX_ = ( (rx)  * numpy.pi/180.)
        RX_array = numpy.array([[1,0,0],[0,numpy.cos(RX_), -1*numpy.sin(RX_)],[0,numpy.sin(RX_),numpy.cos(RX_)]])

        RZ_ = ( (rz) * numpy.pi/180.)
        RZ_array = numpy.array([[numpy.cos(RZ_), -1*numpy.sin(RZ_), 0],[numpy.sin(RZ_),numpy.cos(RZ_), 0],[0,0,1]])

        RY_ = ( (ry) * numpy.pi/180.)
        RY_array = numpy.array([[numpy.cos(RY_), 0, numpy.sin(RY_)],[0,1,0],[-numpy.sin(RY_),0,numpy.cos(RY_)]])

        #basis
        _g_ = g / numpy.sqrt( (g*g).sum() )
        n = numpy.dot(RX_array,_g_)
        n = numpy.dot(RY_array,n)
        nperp  = numpy.array([[n[2,0]],[0],[-n[0,0]]])
        ncross = numpy.array([[ -n[1,0]*n[0,0]],[ n[2,0]**2+n[0,0]**2],[-n[1,0]*n[2,0]]])

        sign   = (-1)**(k)
        updown = sign  * (dic['gap']/2 + T)

        f = g + updown * ncross

        distVimage = Lf

        xtl,ytl,_ = project_virtual_det(mmesh450D[0]-M,tmesh450D[255]-T, n,nperp,ncross,f,v, distVimage)
        xtr,ytr,_ = project_virtual_det(mmesh450D[1535]-M,tmesh450D[255]-T, n,nperp,ncross,f,v, distVimage)
        xbl,ybl,_ = project_virtual_det(mmesh450D[0]-M,tmesh450D[0]-T, n,nperp,ncross,f,v, distVimage)
        xbr,ybr,_ = project_virtual_det(mmesh450D[1535]-M,tmesh450D[0]-T, n,nperp,ncross,f,v, distVimage)
                
        xtl = xtl[0] #float(xtl)
        ytl = ytl[0] #float(ytl)
        xtr = xtr[0] #float(xtr)
        ytr = ytr[0] #float(ytr)
        xbl = xbl[0] #float(xbl)
        ybl = ybl[0] #float(ybl)
        xbr = xbr[0] #float(xbr)
        ybr = ybr[0] #float(ybr)
        
        _angle_ = numpy.arccos( (_g_*v).sum() / (sum(_g_*_g_) * sum(v*v) )) * 180/numpy.pi 

        if abs(_angle_) < 80:
            
            stripes.append(k) 
            
            xvalues.append(xtl)
            xvalues.append(xtr)
            xvalues.append(xbl)
            xvalues.append(xbr)
            yvalues.append(ytl) 
            yvalues.append(ytr) 
            yvalues.append(ybl)
            yvalues.append(ybr)
            
            if plot:        
                ax1.plot(float(g[0]),float(g[2]),'o',color=cnames[k] ) #label='mod:{}/{}'.format(int(mod[k]),ud(mods[k])))
                ax1.annotate("{}/{}".format(int(mod[k]),k), (float(g[0]),float(g[2])  ) )
                ax1.plot(0,0,'dk')
                _r_ = Lf * numpy.sin(90*numpy.pi/180)
                _x_ = numpy.linspace(-_r_, _r_, 100)
                ax1.plot(_x_,numpy.ones(100)*Lf, 'k-')

            if plot: 
                #print((theta[k] + ryhead)*180/numpy.pi, xbl, ybl )
                #print((theta[k] + ryhead)*180/numpy.pi, xtl, ytl)

                ax2.plot( (xbl, xtl), (ybl, ytl), color=cnames[k])
                ax2.plot( (xtl, xtr), (ytl, ytr), color=cnames[k])
                ax2.plot( (xtr, xbr), (ytr, ybr), color=cnames[k])
                ax2.plot( (xbr, xbl), (ybr, ybl), color=cnames[k])

                ax2.plot(xbl,ybl,'o',color=cnames[k])
                ax2.plot(xbr,ybr,'o',color=cnames[k])
                ax2.plot(xtr,ytr,'s',color=cnames[k])
                ax2.plot(xtl,ytl,'s',color=cnames[k])
        else:

            if plot:        
                ax1.plot(float(g[0]),float(g[2]),'kx')
                ax1.plot(0,0,'ok')
        
    xvalues = numpy.array(xvalues)
    yvalues = numpy.array(yvalues)

    xbox = [ xvalues.min(), xvalues.max() ]
    ybox = [ yvalues.min(), yvalues.max() ]

    if plot:    
        #ax1.legend() #,ncol=20)
        _r_ = Lf * numpy.sin(90*numpy.pi/180)
        _x_ = numpy.linspace(xbox[0], xbox[1], 100)
        ax1.plot(_x_,numpy.ones(100)*Lf,'k')
        ax1.plot(xbox[0],Lf, 'sk')
        ax1.plot(xbox[1],Lf, 'sk')
        ax1.plot(-Lf,Lf, 'ok')
        ax1.plot(Lf,Lf, 'ok')
        ax1.plot(0,Lf,'or')
        ax1.plot((0,0),(0,Lf),'r--')
        ax1.set_aspect('equal')
        ax2.set_xlim(xbox[0],xbox[1])
        ax2.set_ylim(ybox[0],ybox[1])

    return xbox, ybox, stripes

def rings450D( dic,  nrings):
    """ Function to simulate diffraction rings at Pimega/pi450D.
    
    Args:
        dic: parameters (see ``dictionary450D()``)
        nrings: number of rings per stripe 

    Returns:
        (ndarray): simulated matrix ( 512 x 15360 ) 
    
    """

    
    xbox , ybox, stripes = get_bounding_box(dic, False)

    D = dic['parameters']

    tmesh450D = D['t_'][0:256]
    mmesh450D = D['m_']

    theta = numpy.array( [ dic['aperture'] * k / 20.0 + dic['aperture']/40.0 for k in range(20) ] ) - dic['aperture']/2.0 + numpy.pi/2.0
    theta = numpy.flipud(theta)

    Lf = 1.01 * dic['earc']

    T = (tmesh450D.max() - tmesh450D.min())/2.0
    M = (mmesh450D.max() - mmesh450D.min())/2.0

    simNx = 10 * 1536
    simNy = 512

    Nx = 20*1536
    Ny = 512

    xmax = xbox[1]
    xmin = xbox[0]
    ymax = ybox[1]
    ymin = ybox[0]

    dx = (xmax-xmin)/(Nx-1)
    dy = (ymax-ymin)/(Ny-1)

    v = numpy.array( [0,0,1] ).reshape([3,1]) 

    Mmesh450D, Tmesh450D = numpy.meshgrid( mmesh450D, tmesh450D )

    mod = numpy.zeros([20,],dtype=int)
    for k in range(10):
        mod[2*k] = k
        mod[2*k+1] = k

    imgSim = numpy.zeros([simNy, simNx])

    ryhead = dic['angle']

    for k in range(len(theta)):

        g = numpy.array([ dic['earc'] * numpy.cos(theta[k]+ryhead), 0, dic['earc'] * numpy.sin(theta[k]+ryhead) ] ).reshape([3,1])

        rx = dic['rotx'][k]
        ry = dic['roty'][k]
        rz = dic['rotz'][k]

        RX_ = ( (rx)  * numpy.pi/180.)
        RX_array = numpy.array([[1,0,0],[0,numpy.cos(RX_), -1*numpy.sin(RX_)],[0,numpy.sin(RX_),numpy.cos(RX_)]])

        RZ_ = ( (rz) * numpy.pi/180.)
        RZ_array = numpy.array([[numpy.cos(RZ_), -1*numpy.sin(RZ_), 0],[numpy.sin(RZ_),numpy.cos(RZ_), 0],[0,0,1]])

        RY_ = ( (ry) * numpy.pi/180.)
        RY_array = numpy.array([[numpy.cos(RY_), 0, numpy.sin(RY_)],[0,1,0],[-numpy.sin(RY_),0,numpy.cos(RY_)]])

        _g_ = g / numpy.sqrt( (g*g).sum() )
        n = numpy.dot(RX_array,_g_)
        n = numpy.dot(RY_array,n)
        nperp  = numpy.array([[n[2,0]],[0],[-n[0,0]]])
        ncross = numpy.array([[ -n[1,0]*n[0,0]],[ n[2,0]**2+n[0,0]**2],[-n[1,0]*n[2,0]]])

        _angle_ = numpy.arccos( (_g_*v).sum() / (sum(_g_*_g_) * sum(v*v) )) * 180/numpy.pi 

        sign   = (-1)**(k)
        updown = sign  * ( dic['gap']/2 + T )

        f = g + updown * ncross

        xMesh,yMesh,_ = project_virtual_det(Mmesh450D-M,Tmesh450D-T, n,nperp,ncross,f,v,Lf)

        xMesh = numpy.fliplr(xMesh)
        
        #-----------
        #build rings
        if nrings > 0:

            rf = 0.90 * max( abs(xMesh.max() ), abs(xMesh.min()) )
            r0 = 1.01 * min( abs(xMesh.max() ), abs(xMesh.min()) )
            eps = numpy.linspace(r0,rf, nrings )
            hexa = 0
            for i in range(len(eps)):
                hexa += numpy.fliplr( (xMesh**2 + yMesh**2 < (1.005 * (eps[i]))**2) & (xMesh**2 + yMesh**2 > (eps[i])**2) )
        else:
            hexa = numpy.zeros([256,1536])
        

        imgSim =  set_image_stripe_detector( imgSim, numpy.flipud(hexa), mod[k], 1-sign) 

    
    return imgSim, stripes


def get_project_values_geometry( *args ):

    if not args:
        ref = {'geo':'arc','opt':True,'mode': 'virtual'}
    else:    
        ref = args[0]
           
    if ref['opt'] == True:

        aperture = numpy.array( [108.92 * numpy.pi/180.]) #radian
        earc = numpy.array([890 * 1e+3]) #um # from sample to pi450D external boundary
        iarc = numpy.array([235 * 1e+3]) #um # from sample to pi450D internal boundary
        gap = numpy.array( [4.07 * 1e+3]) #um     
        rx = numpy.zeros( [20,] )
        ry = numpy.zeros( [20,] )
        rz = numpy.zeros( [20,] )
        dL = numpy.zeros( [20,] )
        ox = numpy.zeros( [20,] )
        oy = numpy.zeros( [20,] )
        z  = 1
        
        x = numpy.array( list( numpy.hstack((rx,ry,rz,dL,ox,oy,gap,z,aperture,earc,iarc)) ) )
            
    else:

        aperture = numpy.array( [108.92 * numpy.pi/180.]) #radian
        earc = numpy.array([890 * 1e+3]) #um # from sample to pi450D external boundary
        iarc = numpy.array([235 * 1e+3]) #um # from sample to pi450D internal boundary
        gap = numpy.array( [4.07 * 1e+3]) #um     
        rx = numpy.zeros( [20,] )
        ry = numpy.zeros( [20,] )
        rz = numpy.zeros( [20,] )
        dL = numpy.zeros( [20,] )
        ox = numpy.zeros( [20,] )
        oy = numpy.zeros( [20,] )
        z  = 1
        
        x = numpy.array( list( numpy.hstack((rx,ry,rz,dL,ox,oy,gap,z,aperture,earc,iarc)) ) )
        
    return x


def dictionary450D( a0, *args ):

    """ Get default values for geometrical image restoration at the pimega/pi450D.

    Args:
        a0: angle wrt to the incident beam
        args: extra arguments

    Returns:
        (dict): Dictionary with informations about the geometrical setup 

    * The output dictionary is given below:

    .. code-block:: python 
        
       det = {
        'rotz': [ [_], [_], [_], [_] ], 
        'roty': [ [_], [_], [_], [_] ],
        'rotx': [ [_], [_], [_], [_] ],
        'angle': a0,
        'z': _,
        'gap': _,
        'offset':  [ [_], [_], [_], [_] ] , 
        'ox': [ [_], [_], [_], [_] ],
        'oy': [ [_], [_], [_], [_] ],
        'shift': _,
        'typedet': 'arc',
        'mode': 'virtual',
        'hexa': _
        }

    * ``det['rotx']``, ``det['roty']``, ``det['rotz']`` 

        These are angle lists for each stripe (clockwise orientation) at row-major order. Each orthornomal basis :math:`\{n, n^\perp, n^{\times}\}` 
        is defined as a rotation of the vector :math:`g` lying at the circle centered at the sample, with respect to ``rotx``, ``roty`` and ``rotz``, respectively.
        Angles are defined in degrees. 

    * ``det['angle']`` 

        This is a Ry angle (degrees) for the 450D pimega head.
 
    * ``det['gap']`` 

        Gap between stripes for 450D geometry.
 
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
    
        The detector type string ``arc``, for the 450D case.
        
    * ``det['hexa']``  

        Integer sequence indicating the hexa's that we want to restore. As an example,
        ``[3,9,15,21]`` indicates that only the fourth hexa of each module will be restored.
        On the other hand, ``range(24)`` indicates that everything will be restored.

    * ``det['mode']``

        Flag indicating ``vìrtual`` restored images. Pimega/450D only
        provide virtual images.

    * Extra arguments are given below. Once the geometrical optimization is 
      performed, the API include optimal values for offset, angles and gap. In 
      this case, the flat 'opt' will be 'True', otherwise 'False'.

    .. code-block:: python
  
       args[0] = {'geo': 'arc', 'opt': True/False, 'mode': 'virtual', 'x': xdet}

       xdet = numpy.array( list(numpy.hstack((rx,ry,rz,dL,ox,oy,v,a,z])) ))

    """
    
    if not args:
        x = get_project_values_geometry( *args )
        ref = {'geo': 'arc', 'opt': False, 'mode': 'virtual'}
    else:            
        ref = args[0]
        if 'x' in ref.keys():
            dic = ref['x']
            x = numpy.array( list(numpy.hstack((dic['rx'],dic['ry'],dic['rz'],dic['dL'],dic['ox'],dic['oy'],dic['gap'],dic['z'],dic['aperture'],dic['earc'],dic['iarc'])) ))            
        else:
            x = get_project_values_geometry( *args )
    
    rx  = x[0:20]
    ry  = x[20:40]
    rz  = x[40:60]
    L   = x[60:80]
    ox  = x[80:100]
    oy  = x[100:120]
    gap = x[120]
    z   = x[121]
    aperture = x[122]
    earc = x[123]
    iarc = x[124]

    if 'susp' in ref.keys():
        susp = ref['susp']
    else:
        susp = 0

    if 'fill' in ref.keys():
        interp = ref['fill']
    else:
        interp = False

        
    if 'hexa' in ref.keys():
        hexa = ref['hexa']
    else:
        hexa = range(20)

    J = 256
    P = 6
    M = 6
    devparameters = parameters450D(J, P, M, 'False', 'planar')
        
    det = {
        'rotz': rz , 
        'roty': ry ,
        'rotx': rx ,
        'angle': a0 * numpy.pi/180.,
        'gap': gap,
        'z': z,
        'dL': L , 
        'ox': ox,
        'oy': oy,
        'shift': False,
        'typedet': 'arc',
        'mode': 'virtual',
        'hexa': hexa,
        'aperture': aperture,
        'earc': earc,
        'iarc': iarc,
        'parameters': devparameters,
        'susp': susp,
        'fill': interp
        }

    return det


def build_geometry_450D(dic, xbox, ybox, stripes):
    
    H = 1536
    J = 256

    D = dic['parameters']
    
    tmesh450D = D['t_'][0:256]
    mmesh450D = D['m_']

    theta = numpy.array( [ dic['aperture'] * k / 20.0 + dic['aperture']/40.0 for k in range(20) ] ) - dic['aperture']/2.0 + numpy.pi/2.0
    theta = numpy.flipud(theta)

    Lf = 1.01 * dic['earc']

    T = (tmesh450D.max() - tmesh450D.min())/2.0
    M = (mmesh450D.max() - mmesh450D.min())/2.0

    rNx = 20 * 1536
    rNy = 1024

    Nx = len(stripes) * 1536
    Ny = 512

    xmax = xbox[1]
    xmin = xbox[0]
    ymax = ybox[1]
    ymin = ybox[0]

    dx = (xmax-xmin)/(Nx-1)
    dy = (ymax-ymin)/(Ny-1)

    v = numpy.array( [0,0,1] ).reshape([3,1]) 

    Mmesh450D, Tmesh450D = numpy.meshgrid( mmesh450D, tmesh450D )

    mod = numpy.zeros([20,],dtype=int)
    for k in range(10):
        mod[2*k] = k
        mod[2*k+1] = k

    ryhead = dic['angle']

    LUT = ( numpy.zeros([20,J,H]), numpy.zeros([20,J,H]), numpy.zeros([20,J,H]), numpy.zeros([20,J,H]), numpy.zeros([20,]), numpy.zeros([20,J,H]), numpy.zeros([20,J,H]) ) 
    
    for k in stripes: #range(len(theta)):

        g = numpy.array([ dic['earc'] * numpy.cos(theta[k]+ryhead), 0, dic['earc'] * numpy.sin(theta[k]+ryhead) ] ).reshape([3,1])

        rx = dic['rotx'][k]
        ry = dic['roty'][k]
        rz = dic['rotz'][k]

        RX_ = ( (rx)  * numpy.pi/180.)
        RX_array = numpy.array([[1,0,0],[0,numpy.cos(RX_), -1*numpy.sin(RX_)],[0,numpy.sin(RX_),numpy.cos(RX_)]])

        RZ_ = ( (rz) * numpy.pi/180.)
        RZ_array = numpy.array([[numpy.cos(RZ_), -1*numpy.sin(RZ_), 0],[numpy.sin(RZ_),numpy.cos(RZ_), 0],[0,0,1]])

        RY_ = ( (ry) * numpy.pi/180.)
        RY_array = numpy.array([[numpy.cos(RY_), 0, numpy.sin(RY_)],[0,1,0],[-numpy.sin(RY_),0,numpy.cos(RY_)]])

        _g_ = g / numpy.sqrt( (g*g).sum() )
        n = numpy.dot(RX_array,_g_)
        n = numpy.dot(RY_array,n)
        nperp  = numpy.array([[n[2,0]],[0],[-n[0,0]]])
        ncross = numpy.array([[ -n[1,0]*n[0,0]],[ n[2,0]**2+n[0,0]**2],[-n[1,0]*n[2,0]]])

        _angle_ = numpy.arccos( (_g_*v).sum() / (sum(_g_*_g_) * sum(v*v) )) * 180/numpy.pi 

        sign   = (-1)**(k)
        updown = sign  * ( dic['gap']/2 + T )

        f = g - updown * ncross

        xMesh,yMesh,_ = project_virtual_det(Mmesh450D-M,Tmesh450D-T, n,nperp,ncross,f,v,Lf)

        #
        #nearest neighbour interp: assuming a regular mesh at the device
        #        
        ix = Nx - (numpy.floor( (xMesh - xmin) /( dx ) )).astype(int)
        iy = Ny - ( numpy.floor( (yMesh - ymin) /( dy ) )).astype(int) 
        ixp1 = ix + 1
        ixm1 = ix - 1

        ix[  ix < 0 ] = 0
        ix[  ix >= Nx-1 ] = Nx-1

        ixp1[  ixp1 < 0] = 0
        ixp1[  ixp1 >= Nx-1 ] = Nx-1

        ixm1[  ixm1 < 0] = 0
        ixm1[  ixm1 >= Nx-1 ] = Nx-1

        iy[ iy <0 ] = 0
        iy[ iy >= Ny-1 ] = Ny-1

        LUT[0][ k, : ] = ix
        LUT[1][ k, : ] = iy
        LUT[2][ k, : ] = ixp1
        LUT[3][ k, : ] = ixm1
        LUT[4][ k ]    = 1-sign
        LUT[5][ k, : ] = xMesh
        LUT[6][ k, : ] = yMesh

    
    geometry = {}
    geometry['LUT'] = [ LUT[0], LUT[1], LUT[2], LUT[3], LUT[4], LUT[5], LUT[6]]
    geometry['shape'] = [[Ny,Nx], [rNy,rNx]]    
    return geometry


def geometry450D ( params, *args ):
    """ Function to compute prior geometrical information related to arc pimega/450D detector. 
    
    Args:
        params: input parameters 
        args: extra arguments

    Returns:
        (dict): Geometrical information
    
    """
    
    J = 256
    P = 6
    M = 6
            
    a  = params['angle']
    z  = params['z']
    offsetx = params['ox']
    offsety = params['oy']

    RxM = numpy.array( [ params['rotx'][0], params['rotx'][1], params['rotx'][2], params['rotx'][3] ] )
    RyM = numpy.array( [ params['roty'][0], params['roty'][1], params['roty'][2], params['roty'][3] ] )
    RzM = numpy.array( [ params['rotz'][0], params['rotz'][1], params['rotz'][2], params['rotz'][3] ] )
    Ox = numpy.array( [ params['ox'][0], params['ox'][1], params['ox'][2], params['ox'][3] ] )
    Oy = numpy.array( [ params['oy'][0], params['oy'][1], params['oy'][2], params['oy'][3] ] )
    dL = numpy.array( [ params['dL'][0], params['dL'][1], params['dL'][2], params['dL'][3] ] )
       
    if not args:
        boxinfo = get_bounding_box( params, False )  
    else:
        boxinfo = args[0]
        
    xbox, ybox, stripes = boxinfo
    #shift = 'False' #original bump bond position , otherwise True
    #typedet = 'planar' #chuncho: usando coisas ja prontas pro 135D / 540D
    #D = parameters450D(J, P, M, shift, typedet)
    
    geometry = build_geometry_450D(params, xbox, ybox, stripes)
    
    geometry['boxinfo'] = boxinfo
    geometry['hexa'] = params['hexa']
    geometry['susp'] = params['susp']
    geometry['fill'] = params['fill']
    
    return geometry


def backward450D(frame, geometry):
    """ Function to restore a given frame using a measured pimega/pi450D data
    
    Args:
        matrix: digital 512x15360 measured matrix from pimega/pi450D.
        geometry: geometrical data related to pimega/pi450D (see ``dictionar450D()``). 

    Returns:
        (ndarray): restored matrix
    
    """
    
    SUSP450D = -10
    
    Nx = 20 * 1536
    Ny = 512
    
    mod = numpy.zeros([20,],dtype=int)
    for k in range(10):
        mod[2*k] = k
        mod[2*k+1] = k

    ##
    
    if len(frame.shape)==3:
        print('ssc-pimega error! Function not implemented for image blocks!')
    
    if len(frame.shape)==2:

        new = -1 * numpy.ones( [Ny, Nx] )
                
        for k in range(20):
            if k in geometry['hexa']:

                ix   = geometry['LUT'][0][k].astype(int)
                iy   = geometry['LUT'][1][k].astype(int)
                ixp1 = geometry['LUT'][2][k].astype(int)
                ixm1 = geometry['LUT'][2][k].astype(int)
                sign = geometry['LUT'][4][k]
                
                stripe = get_image_stripe_detector( frame, mod[k], sign )
                stripe = set_suspicious_pixels_450D( stripe, geometry['susp'] )
                
                new[iy, ix] =  stripe 

                #improving interpolation for missing values
                if geometry['fill'] == True:
                        
                    projectedHexa = new[ iy.min():iy.max(), ix.min():ix.max() ]
                        
                    missing = (projectedHexa == -1)
                    
                    projectedHexa2 = interpolate_missing_pixels(projectedHexa, missing, 'nearest', SUSP450D)
                    
                    new[ iy.min():iy.max(), ix.min():ix.max()] = projectedHexa2

                #new[iy, ixp1] =  stripe 
                #new[iy, ixm1] =  stripe 

    new[new < 0] = -1
    
    return new
