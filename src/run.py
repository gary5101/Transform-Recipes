import os
import sys
import cv2
import numpy as np
from utils import convertRGB_YCbCr, float2uint8, convertYCbCr_RGB
from utils import laplacianStack, get_lowpass_image, get_multiscale_luminance, get_patch_features, extend_features, get_model
from reconstruction import reconstruct
import matplotlib.pyplot as plt
from params import wSize, k, step, _nbytes
from imresize import imresize
from tqdm import tqdm
import signal 
import sys
from save_get import save_recipe, get_recipe
import argparse

def set_recipe_shape(I,O):
    """ Setup the number of recipes channel for the given parameters"""
    
    # Linear terms + affine offset
    recipe_channels  = I.shape[2] + 1

    # Additional luma channels
    add_luma_channels  = 0
        
    add_luma_channels += (np.log2(wSize)-1).astype(int)
    
    add_luma_channels += k-1

    sz = O.shape
    recipe_h   = int(np.ceil((1.0*sz[0])/(step)))
    recipe_w   = int(np.ceil((1.0*sz[1])/(step)))
    recipe_d   = sz[2]*recipe_channels + add_luma_channels

    # Number of regression channels per output channels
    rcp_channels      = sz[2]*[recipe_channels]
    rcp_channels[0]  += add_luma_channels

    # High pass parameters
    recipe_hp_shape  = [recipe_h, recipe_w, recipe_d]
    return rcp_channels, recipe_hp_shape,

def get_lowpass_recipe(lowpassI, lowpassO):
    """ Computes the low frequency part of the recipe """
    if lowpassO.shape[2] == 1:
        lI = lowpassI[:,:,0]
        lI.shape += (1,)
        lO = (lowpassO+1)/(lI+1)
    else:
        lO = (lowpassO+1)/(lowpassI+1)
    lowpassO = lO
    lowpassO = lowpassO.astype(np.float16)

    return lowpassO

def main(args):
    global _nbytes
    # image load
    stepSize = wSize//2

    inputImage = cv2.imread(args[0])
    outputImage = cv2.imread(args[1])

    # Converting RGB image to YCbCr space

    inputImageYCbCr = convertRGB_YCbCr(inputImage)
    outputImageYCbCr = convertRGB_YCbCr(outputImage)

    recipeStatus = True
    reconstructStatus = True
    # Create Recipe

    if recipeStatus:

        I = np.copy(inputImageYCbCr).astype(np.float64)
        O = np.copy(outputImageYCbCr).astype(np.float64)


        rcp_channels, recipe_hp_shape = set_recipe_shape(inputImageYCbCr,outputImageYCbCr)
        recipe_hp = np.zeros(recipe_hp_shape,dtype = np.float64)
        recipe_h  = recipe_hp.shape[0]
        recipe_w  = recipe_hp.shape[1]
        lowpassI  = get_lowpass_image(I)
        lowpassO  = get_lowpass_image(O)

        # Lowpass residual
        recipe_lp = get_lowpass_recipe(lowpassI, lowpassO)

        # Multiscale features
        ms_luma = get_multiscale_luminance(I)

        # High pass component
        lowpassI  = imresize(lowpassI,I.shape[0:2])

        lowpassO  = imresize(lowpassO,O.shape[0:2])
        highpassI = I - lowpassI
        highpassO = O - lowpassO

        idx = 0
        sys.stdout.write("\n")
        for imin in tqdm(range(0,I.shape[0],step),desc="Recipe Making"):
            for jmin in range(0,I.shape[1],step):
                idx += 1

                # Recipe indices
                patch_i = imin//step
                patch_j = jmin//step

                # Patch indices in the full-res image
                i_rng    = (imin, min(imin+wSize,I.shape[0]))
                j_rng    = (jmin, min(jmin+wSize,I.shape[1]))

                # Slicing the input image to get wSize patches
                X  = get_patch_features(highpassI,i_rng,j_rng)
                Y  = get_patch_features(highpassO,i_rng,j_rng)
                Xr = get_patch_features(inputImageYCbCr,i_rng,j_rng)

                # Converting input image into appropriate form for regression
                X = extend_features(X,Xr,i_rng = i_rng,j_rng = j_rng, ms_levels = ms_luma)

                # Applying regression channel-wise
                rcp_chan = 0
                for chanO in range(O.shape[2]):
                    rcp_stride = rcp_channels[chanO]
                    reg        = get_model()
                    reg.fit(X[:, 0:rcp_stride-1],Y[:,chanO])
                    coefs = np.append(reg.coef_, reg.intercept_)

                    # Fill in the recipe
                    recipe_hp[patch_i, patch_j,rcp_chan:rcp_chan+rcp_stride] = coefs

                    rcp_chan += rcp_stride
        sys.stdout.write("\n")
        _nbytes   = recipe_hp.nbytes
        _nbytes  += recipe_lp.nbytes

        save_recipe(recipe_lp,recipe_hp)
        print("Recipe Saved")

    
    if reconstructStatus:
        recipe_hp1, recipe_lp1 = get_recipe()
        print("Recipe Loaded")

        # Reading the input image, simulating behaviour at client
        orig = cv2.imread(args[2])

        # Converting input image to YCbCr space
        orig = convertRGB_YCbCr(orig)
        rcp_channels, recipe_hp_shape = set_recipe_shape(inputImageYCbCr,outputImageYCbCr)

        # Reconstructing output image based on recipe
        output_image = reconstruct(orig,inputImageYCbCr, outputImageYCbCr, rcp_channels, recipe_hp_shape, recipe_lp1, recipe_hp1 )

        # Converting output image in YCbCr space back to RGB space
        output_image = convertYCbCr_RGB(output_image)
        output_image = float2uint8(output_image)
        print("Reconstruction Done")
        output_image = np.squeeze(output_image)
        cv2.imwrite('../outputs/out.png',output_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original')
    inputs = parser.parse_args()
    inputImageLocation = '../transformed/inputRef.png'
    outputImageLocation = '../transformed/outRef.png'
    originalInputLocation = inputs.original
    args = [inputImageLocation, outputImageLocation, originalInputLocation]
    main(args)