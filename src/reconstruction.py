import numpy as np
from utils import laplacianStack, convertRGB_YCbCr, convertYCbCr_RGB
from utils import get_lowpass_image, get_multiscale_luminance, get_patch_features, extend_features, get_model
from params import wSize, k, sigma, step
import cv2
from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from imresize import imresize
import numbers

simplefilter("ignore", category=ConvergenceWarning)

def reconstruct_lowpass_residual(lowpassI, recipe_lp, outputImageRef):
    """ Reconstruct lowpass part of the image from recipe """

    lowpassO = recipe_lp.astype(np.float64)
    if lowpassO.shape[2] == 1:
        lI = lowpassI[:,:,0]
        lI.shape += (1,)
        lowpassO = lowpassO * (lI+1) - 1
    else:
        lowpassO = lowpassO * (lowpassI+1) - 1
    lowpassO = imresize(lowpassO,outputImageRef.shape[0:2])
    return lowpassO


def reconstruct(inputImageRef,inputImage, outputImageRef,rcp_channels, recipe_hp_shape, recipe_lp, recipe_hp):
    """ Reconstruct and approximation of the output image, from recipe"""

    Ideg = imresize(inputImage, inputImageRef.shape[0:2])

    # We reconstruct the output based on the reference input
    I = np.copy(inputImageRef).astype(np.float64)
    
    
    recipe_h = recipe_hp.shape[0]
    recipe_w = recipe_hp.shape[1]

    # Construct four images, in case of overlap, which we'll later linearly
    # interpolate
    R = []
    for i in range(4):
        R.append(np.zeros((I.shape[0],I.shape[1],outputImageRef.shape[2])))
    

    lowpassI = get_lowpass_image(I)

    # Reconstruct lowpass
    lowpassO = reconstruct_lowpass_residual(lowpassI, recipe_lp, outputImageRef)
    for i in range(len(R)):
        R[i] = np.copy(lowpassO)

    # Multiscale features
    ms_luma = get_multiscale_luminance(I)

    # High pass component
    lowpassI = imresize(lowpassI,I.shape[0:2])
    highpassI = I - lowpassI

    idx = 0
    for imin in tqdm(range(0,I.shape[0],step),desc="Reconstruction"):
        for jmin in range(0,I.shape[1],step):
            idx += 1
            # Recipe indices
            patch_i = imin//step
            patch_j = jmin//step

            patch_i = min(patch_i, recipe_hp.shape[0]-1)
            patch_j = min(patch_j, recipe_hp.shape[1]-1)
            
            r_index = (patch_i % 2)*2 + (patch_j % 2)

            # Patch indices in the full-res image
            i_rng = (imin, min(imin+wSize,I.shape[0]))
            j_rng = (jmin, min(jmin+wSize,I.shape[1]))
            X          = get_patch_features(highpassI,i_rng,j_rng)
            Xr         = get_patch_features(inputImageRef,i_rng,j_rng)
            X_degraded = get_patch_features(Ideg, i_rng, j_rng)

            X = extend_features(X,Xr,X_degraded = X_degraded, i_rng = i_rng, j_rng = j_rng, ms_levels = ms_luma)

            # Traversing each recipe channel
            rcp_chan = 0
            for chanO in range(outputImageRef.shape[2]):
                rcp_stride = rcp_channels[chanO]
                reg        = get_model()
                coefs = recipe_hp[patch_i, patch_j,rcp_chan:rcp_chan+rcp_stride]

                # hack because sklearn needs to be fitted to initialize coef_, intercept_
                reg.fit(np.zeros((2,2)), np.zeros((2,)))

                reg.coef_      = coefs[0:-1]
                reg.intercept_ = coefs[-1]

                recons         = reg.predict(X[:,0:rcp_stride-1])
                R[r_index][i_rng[0]:i_rng[1], j_rng[0]:j_rng[1],chanO] += np.reshape(recons, (i_rng[1]-i_rng[0],j_rng[1]-j_rng[0]))
                rcp_chan += rcp_stride

    R = linear_interpolate(R)
    return R

def linear_interpolate(R):
    """ Linearly interpolate pixel values of overlapping patches"""

    res  = np.array(R[0])
    res2 = np.array(R[2])

    sz = R[0].shape
    s  = step

    x = np.linspace(0,1,s)
    x = np. reshape(x,(1,s))
    x = np.concatenate((x,np.fliplr(x)),axis=1)
    x = np.tile(x,(sz[0],sz[1]//(s*2)+1))
    x = x[:,0:sz[1]]
    x[:,0:s] = 1
    x.shape += (1,)
    
    res  = np.multiply(res,x) + np.multiply((1-x),R[1])
    res2 = np.multiply(res2,x) + np.multiply((1-x),R[3])

    x = np.linspace(0,1,s)
    x = np. reshape(x,(s,1))
    x = np.concatenate((x,np.flipud(x)),axis=0)
    x = np.tile(x,(sz[0]//(s*2)+1,sz[1]))
    x = x[0:sz[0],:]
    x[0:s,:] = 1
    x.shape += (1,)

    res = np.multiply(x,res) + np.multiply((1-x),res2)
    return res
