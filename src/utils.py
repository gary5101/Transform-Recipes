import cv2
import numpy as np
from params import wSize, sigma
import os
from params import k
from sklearn.linear_model import Lasso
from imresize import imresize
def float2uint8(R):
    '''Clamping image block from float to uint8.'''

    return np.uint8(np.clip(np.round(R),0,255))

def convertRGB_YCbCr(im):
    '''Converting image to YCbCr space from RGB'''

    im = im.astype(np.float32)
    YCbCr = np.zeros(im.shape)
    YCbCr[:,:,0] = 0.299*im[:,:,0] + 0.587*im[:,:,1] + 0.114*im[:,:,2]
    YCbCr[:,:,1] = 128 - 0.168736*im[:,:,0] - 0.331264*im[:,:,1] + 0.5*im[:,:,2]
    YCbCr[:,:,2] = 128 + 0.5*im[:,:,0] - 0.418688*im[:,:,1] - 0.081312*im[:,:,2]

    YCbCr = float2uint8(YCbCr)
    return YCbCr

def convertYCbCr_RGB(im):
    '''Converting image to RGB space from YCbCr.'''

    im = im.astype(np.float32)
    RGB = np.zeros(im.shape)
    RGB[:,:,0] = im[:,:,0] + 1.402*(im[:,:,2]-128)
    RGB[:,:,1] = im[:,:,0] - 0.34414*(im[:,:,1]-128) - 0.71414*(im[:,:,2]-128)
    RGB[:,:,2] = im[:,:,0] + 1.772*(im[:,:,1]-128)

    RGB = float2uint8(RGB)

    return RGB

def laplacianStack(I, nLevels= -1, minSize = 1, useStack = True):
    ''' Builds laplacian stack or pyramid based on the value of `useStack`'''

    if nLevels == -1:
        nLevels = int(np.log2(I.shape[0]))+1

    pyramid = nLevels*[None]
    pyramid[0] = I
    if len(pyramid[0].shape) < 3:
        pyramid[0].shape += (1,)
    # All levels have the same resolution
    if useStack:
        # Gaussian pyramid
        for i in range(nLevels-1):
            srcSz = pyramid[i].shape[0:2]
            newSz = tuple([a/2 for a in pyramid[i].shape[0:2]])
            newSz = (newSz[1],newSz[0])
            pyramid[i+1] = cv2.pyrDown(pyramid[i])
            if len(pyramid[i+1].shape) < 3:
                pyramid[i+1].shape += (1,)

        # Make a stack
        for lvl in range(0,nLevels-1):
            for i in range(nLevels-1,lvl,-1):
                newSz = pyramid[i-1].shape[0:2]
                up = cv2.pyrUp(pyramid[i],dstsize=(newSz[1],newSz[0]))
                if len(up.shape) < 3:
                    up.shape += (1,)
                pyramid[i] = np.array(up)

        lapl = nLevels*[None]
        lapl[nLevels-1] = np.copy(pyramid[nLevels-1])
        for i in range(0,nLevels-1):
            lapl[i] = pyramid[i].astype(np.float32) - pyramid[i+1].astype(np.float32)
        pyramid = lapl
        return pyramid

    else:
        for i in range(nLevels-1):
            srcSz = pyramid[i].shape[0:2]
            newSz = tuple([a/2 for a in pyramid[i].shape[0:2]])
            newSz = (newSz[1],newSz[0])
            pyramid[i+1] = cv2.pyrDown(pyramid[i])
            if len(pyramid[i+1].shape) < 3:
                pyramid[i+1].shape += (1,)

        for i in range(nLevels-1):
            newSz = pyramid[i].shape[0:2]
            up = cv2.pyrUp(pyramid[i+1],dstsize=(newSz[1],newSz[0])).astype(np.float32)
            if len(up.shape) < 3:
                up.shape += (1,)
            pyramid[i] = pyramid[i].astype(np.float32) - up
        
        return pyramid

def get_lowpass_image(I):
    """ Downsamples the image """

    lp_ratio   = wSize
    lp_sz = [s/lp_ratio for s in I.shape[0:2]]
    lowpassI   = imresize(I,lp_sz)
    return lowpassI

def get_multiscale_luminance(I):
    """ Build the maps for multiscale luminance features """

    II = np.copy(I)[:,:,0]
    n_ms_levels = int(np.log2(wSize)-1)
    ms = np.zeros((I.shape[0],I.shape[1],n_ms_levels))
    L = laplacianStack(II,nLevels = n_ms_levels+1, useStack = True)
    ms = np.zeros((I.shape[0],I.shape[1],n_ms_levels))
    L.pop() # Remove lowpass-residual
    for i,p in enumerate(L):
        ms[:,:,i] = p[:,:,0]
    return ms

def get_patch_features(I, i_rng, j_rng):
     """ Reshape patch data to feature vectors """

     patch = I[i_rng[0]:i_rng[1], j_rng[0]:j_rng[1]]
     sz = patch.shape
     X  = np.reshape(patch,(sz[0]*sz[1],sz[2]))
     X  = np.float64(X)
     return X

def extend_features( X, Xr, X_degraded = None, i_rng = None, j_rng = None,  ms_levels = None):
    """ Add features for luminance prediction"""

    luma_band_thresh = k-1
    Xl = Xr[:,0]
    if X_degraded is not None:
        mini = np.amin(X_degraded[:,0])
        maxi = np.amax(X_degraded[:,0])
    else:
        mini = np.amin(Xl)
        maxi = np.amax(Xl)
    l_step = (maxi-mini)/k
    for il in range(1,k):
        bp = mini+il*l_step
        if l_step < luma_band_thresh:
            Xl1 = np.zeros(Xl.shape)
        else:
            Xl1 = (Xl>=bp).astype(np.float64)
            Xl1 = Xl1*(Xl-bp)
        Xl1.shape += (1,)
        X = np.concatenate((X,Xl1), axis = 1)
    
    Xd = get_patch_features(ms_levels,i_rng, j_rng)
    X = np.concatenate((X,Xd),axis=1)
    return X

def get_model(  ):
    """ Fetch the regression model to use """

    return Lasso(alpha = 1e-3, fit_intercept = True, precompute = True,  max_iter = 1e4)

def buildGaussianPyramid(I, nLevels= -1, minSize = 16):
    """Builds Gaussian Pyramid."""

    if nLevels == -1:
        nLevels = getNlevels(I,minSize)

    pyramid = nLevels*[None]
    pyramid[0] = I
    for i in range(nLevels-1):
        pyramid[i+1] = cv2.pyrDown(pyramid[i])

    return pyramid

def reconstructFromLaplacianPyramid(pyramid):
    """Collapses Laplacian pyramid to form the original image."""
    
    nLevels = len(pyramid)
    out = pyramid[-1]
    if len(pyramid) == 1:
        return out

    useStack = False
    if pyramid[0].shape[0:2] == pyramid[-1].shape[0:2]:
        useStack = True

    dtp = out.dtype
    for i in range(nLevels-2,-1,-1):
        newSz = pyramid[i].shape[0:2]
        if useStack:
            up = out
        else:
            up = cv2.pyrUp(out,dstsize=(newSz[1],newSz[0]))
        if len(up.shape) < 3:
            up.shape += (1,)
        out =  up + pyramid[i]
        out = out.astype(dtp)

    return out
