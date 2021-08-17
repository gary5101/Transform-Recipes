from params import _nbytes, enc_maxs, enc_mins, recipe_lp_shape, recipe_hp_shape
import numpy as np
from PIL import Image, TiffImagePlugin
import os
import imageio

def remodel_save(im):
    """Remodels recipe to store-able format."""

    sz = im.shape
    new_im = np.zeros((sz[0],sz[1]*sz[2]))
    for c in range(sz[2]):
        new_im[:,c*sz[1]:(c+1)*sz[1]] = im[:,:,c]
    return new_im

def remodel_get(im, sz):
    """Remodels the retrieved recipe to a usable format."""
    
    new_im = np.zeros(sz)
    for c in range(sz[2]):
        new_im[:,:,c] = im[:,c*sz[1]:(c+1)*sz[1]]
    return new_im

def save_recipe(recipe_lp,recipe_hp):
    """Saves the computed recipe."""

    global _nbytes, enc_maxs, enc_mins, rec, recipe_hp_shape, recipe_lp_shape
    
    # Encodes recipe_hp to 0 - 255 space
    rcp   = recipe_hp
    sz    = rcp.shape[0:2]
    nChan = rcp.shape[2]
    mins  = nChan*[None]
    maxs  = nChan*[None]

    nbins = 2**8-1
    result = np.zeros(rcp.shape, dtype = np.uint8)
    _nbytes -= rcp.nbytes
    for c in range(nChan):
        vals    = rcp[:,:,c].astype(np.float32)
        mins[c] = np.amin(vals)
        maxs[c] = np.amax(vals)
        rng = maxs[c] - mins[c]
        if rng <= 0:
            rng = 1

        vals -= mins[c]
        vals /= rng
        vals *= nbins
        vals += 0.5

        result[:,:,c] = vals
    
    recipe_hp = result
    enc_mins = mins
    enc_maxs = maxs
    _nbytes += recipe_hp.nbytes

    # Paths to store the recipe
    fname_lp = "../Compressions/recipe_lp.tif"
    fname_hp = "../Compressions/recipe_hp.png"

    recipe_hp_shape = recipe_hp.shape
    recipe_lp_shape = recipe_lp.shape

    hp_img = remodel_save(recipe_hp)
    lp_img = remodel_save(recipe_lp)

    Image.fromarray(np.uint8(hp_img)).save(fname_hp)
    fSize = os.stat(fname_hp).st_size

    h,w = lp_img.shape
    # Recipe saved
    imageio.imsave(fname_lp, lp_img)
    fSize += os.stat(fname_lp).st_size

    _nbytes = fSize
    


def get_recipe():
    """Loads the recipe"""

    global enc_mins, enc_maxs, recipe_hp_shape, recipe_lp_shape

    # Paths to stored recipe
    fname_lp = "../Compressions/recipe_lp.tif"
    fname_hp = "../Compressions/recipe_hp.png"

    im_file = Image.open(fname_hp)
    recipe_hp = remodel_get(np.array(im_file), recipe_hp_shape).astype('uint8')


    im = imageio.imread(fname_lp)
    recipe_lp = im
    recipe_lp = remodel_get(recipe_lp, recipe_lp_shape).astype('float16')

    # Decodes recipe_hp from 0-255 space
    rcp = recipe_hp
    sz    = rcp.shape[0:2]
    nChan = rcp.shape[2]
    mins  = enc_mins
    maxs  = enc_maxs

    nbins = 2**8-1
    result = np.zeros(rcp.shape, dtype = np.float32)
    for c in range(nChan):
        vals    = rcp[:,:,c].astype(np.float32)
        rng = maxs[c] - mins[c]
        if rng <= 0:
            rng = 1

        vals /= nbins
        vals *= rng
        vals += mins[c]
        result[:,:,c] = vals
    recipe_hp = result

    return recipe_hp, recipe_lp

USE_ADOBE_QTABLES = False

def get_img(path):
    """Loads image from the path"""

    name, ext = os.path.splitext(path)
    if ext == '.jp2':
        fname = name+"_temp.png"
        cmd = "convert %s %s" % (path,fname)
        os.system(cmd)
        fSize = os.stat(path).st_size
        im_file = Image.open(fname)
        im = np.array(im_file)
        os.remove(fname)
    else:
        im_file = Image.open(path)
        fSize = os.stat(path).st_size
        im = np.array(im_file)

    return im, fSize


def select_qtable(q):
    """JPEG compression utility"""

    if q <= 10:
        qtable = {
            0: array('b', [27, 26, 26, 41, 29, 41, 65, 38, 38, 65, 66, 47, 47, 47, 66, 39, 28, 28, 28, 28, 39, 34, 23, 23, 23, 23, 23, 34, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
            1: array('b', [29, 41, 41, 52, 38, 52, 34, 24, 24, 34, 20, 14, 14, 14, 20, 20, 14, 14, 14, 14, 20, 17, 12, 12, 12, 12, 12, 17, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
        }
    elif q <= 20:
        qtable = {
            0: array('b', [20, 17, 17, 26, 18, 26, 41, 24, 24, 41, 51, 39, 32, 39, 51, 39, 28, 28, 28, 28, 39, 34, 23, 23, 23, 23, 23, 34, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
            1: array('b', [21, 26, 26, 33, 29, 33, 34, 24, 24, 34, 20, 14, 14, 14, 20, 20, 14, 14, 14, 14, 20, 17, 12, 12, 12, 12, 12, 17, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
        }
    elif q <= 30:
        qtable = {
            0: array('b', [18, 14, 14, 22, 16, 22, 35, 21, 21, 35, 44, 34, 27, 34, 44, 39, 28, 28, 28, 28, 39, 34, 23, 23, 23, 23, 23, 34, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
            1: array('b', [20, 22, 22, 29, 25, 29, 34, 24, 24, 34, 20, 14, 14, 14, 20, 20, 14, 14, 14, 14, 20, 17, 12, 12, 12, 12, 12, 17, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
        }
    elif q <= 40:
        qtable = {
            0: array('b', [12, 8, 8, 13, 9, 13, 21, 12, 12, 21, 26, 20, 16, 20, 26, 32, 27, 26, 26, 27, 32, 34, 23, 23, 23, 23, 23, 34, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
            1: array('b', [13, 13, 13, 17, 14, 17, 27, 17, 17, 27, 20, 14, 14, 14, 20, 20, 14, 14, 14, 14, 20, 17, 12, 12, 12, 12, 12, 17, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
        }
    elif q <= 50:
        qtable = {
            0: array('b', [10, 7, 7, 11, 8, 11, 18, 10, 10, 18, 22, 17, 14, 17, 22, 27, 23, 22, 22, 23, 27, 34, 23, 23, 23, 23, 23, 34, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
            1: array('b', [11, 14, 14, 31, 19, 31, 34, 24, 24, 34, 20, 14, 14, 14, 20, 20, 14, 14, 14, 14, 20, 17, 12, 12, 12, 12, 12, 17, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
        }
    elif q <= 60:
        qtable = {
            0: array('b', [8, 6, 6, 9, 6, 9, 14, 8, 8, 14, 17, 13, 11, 13, 17, 21, 18, 17, 17, 18, 21, 28, 23, 23, 23, 23, 23, 28, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
            1: array('b', [9, 9, 9, 11, 10, 11, 18, 11, 11, 18, 20, 14, 14, 14, 20, 20, 14, 14, 14, 14, 20, 17, 12, 12, 12, 12, 12, 17, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
        }
    elif q <= 70:
        qtable = {
            0: array('b', [4, 3, 3, 4, 3, 4, 7, 4, 4, 7, 9, 7, 5, 7, 9, 11, 9, 9, 9, 9, 11, 14, 12, 12, 12, 12, 12, 14, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
            1: array('b', [4, 6, 6, 12, 8, 12, 22, 12, 12, 22, 20, 14, 14, 14, 20, 20, 14, 14, 14, 14, 20, 17, 12, 12, 12, 12, 12, 17, 17, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
        }
    elif q <= 80:
        qtable = {
            0: array('b', [2, 2, 2, 3, 2, 3, 4, 2, 2, 4, 5, 4, 3, 4, 5, 6, 5, 5, 5, 5, 6, 8, 7, 7, 7, 7, 7, 8, 11, 9, 9, 9, 9, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
            1: array('b', [3, 3, 3, 7, 4, 7, 13, 7, 7, 13, 15, 13, 13, 13, 15, 15, 14, 14, 14, 14, 15, 15, 12, 12, 12, 12, 12, 15, 15, 12, 12, 12, 12, 12, 12, 15, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]),
        }
    elif q <= 90:
        qtable = {
            0: array('b', [1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 5, 6, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]),
            1: array('b', [1, 2, 2, 4, 2, 4, 7, 4, 4, 7, 8, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]),
        }
    else:
        qtable = {
            0: array('b', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            1: array('b', [1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
        }
    return qtable


def save_and_get_img(im, path, quality = -1, subsampling = 1):
    """stores and retrieves image given path, simulating transfer between client and server."""

    name, ext = os.path.splitext(path)
    if ext == '.jpg' or ext == '.jpeg':
        if quality == -1:
            quality = 100
        ii = Image.fromarray(im)
        if USE_ADOBE_QTABLES:
            qtable = select_qtable(quality)
            ii.save(path, subsampling = subsampling,  qtables = qtable)
        else:
            ii.save(path, quality = quality, subsampling = subsampling)
        fSize = os.stat(path).st_size
        im_file = Image.open(path)
        im = np.array(im_file)
    elif ext == '.jp2':
        if quality == -1:
            quality = 1
        # because pillow jp2000 doesn work for now...
        fname = name+"_temp.png"
        jp2 = name+".jp2"
        Image.fromarray(np.uint8(im)).save(fname)
        cmd = "opj_compress -i %s -o %s -r %s  >/dev/null 2>&1" % (fname,jp2,quality)
        if not os.system(cmd):
            os.remove(fname)
            cmd = "convert %s %s" % (jp2,fname)
            os.system(cmd)
            fSize = os.stat(jp2).st_size
        else:
            print ("ERROR: couldnt make jpeg2000")
            fSize = os.stat(fname).st_size
        im_file = Image.open(fname)
        im = np.array(im_file)
        os.remove(fname)
    elif ext == '.png':
        Image.fromarray(np.uint8(im)).save(path)
        fSize = os.stat(path).st_size
        im_file = Image.open(path)
        im = np.array(im_file)
    elif ext == '.tif':
        h,w = im.shape
        iii = Image.frombytes("I;16",(w,h), im.tostring())
        im2 = np.array(iii)
        im2 = im2.view('float16')
        iii.save(path, compression = "tiff_deflate")
        fSize = os.stat(path).st_size
        im_file = Image.open(path)
        im = np.array(im_file)
        im = im.view('float16')
    else:
        print ("ERROR: unrecognized format")
        im = np.zeros((1,1))
        fSize = 0

    return im, fSize