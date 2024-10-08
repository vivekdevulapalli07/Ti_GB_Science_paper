/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on 17.01.23 at 21:09 by c.liebscher

"""

#Libraries
import numpy as np
from skimage.morphology import white_tophat, disk


#Normalize image
def normalize_image(image):

    normalized_image = (image - np.min(image)) / ((np.max(image) - np.min(image)))

    return normalized_image


#Remove polynomial background
def polynomial_background(image, poly_order: int):

    # Image size
    (Nx, Ny) = image.shape

    # Generate array for fitting background coefficients
    x = np.arange(0, Nx)
    y = np.arange(0, Ny)
    xa, ya = np.meshgrid(y, x)

    # Cast into vectors for fitting
    x = xa.flatten()
    y = ya.flatten()
    b = image.flatten()

    # Determine background for chosen polynomial order (up to 3)
    if poly_order == 1:
        A1 = np.stack((np.ones((Nx*Ny)), x, y, x*y), axis=1)
        beta1 = np.linalg.lstsq(A1, b, rcond=None)
        bkg = beta1[0][0] + beta1[0][1]*xa + beta1[0][2]*ya + beta1[0][3]*xa*ya

    elif poly_order == 2:
        A2 = np.stack((np.ones((Nx*Ny)), x, y, x*y, x**2, y**2, x**2*y, x*y**2, x**2*y**2), axis=1)
        beta2 = np.linalg.lstsq(A2, b, rcond=None)
        bkg = beta2[0][0] + beta2[0][1]*xa + beta2[0][2]*ya + beta2[0][3]*xa*ya + beta2[0][4]*xa**2 + \
              beta2[0][5]*ya**2 + beta2[0][6]*xa**2*ya + beta2[0][7]*xa*ya**2 + beta2[0][8]*xa**2*ya**2

    elif poly_order == 3:
        A3 = np.stack((np.ones((Nx*Ny)), x, y, x*y, x**2, y**2, x**2*y, x*y**2, x**2*y**2,
                           x**3, y**3, x**3*y, x*y**3, x**3*y**2, x**2*y**3), axis=1)
        beta3 = np.linalg.lstsq(A3, b, rcond=None)
        bkg = beta3[0][0] + beta3[0][1]*xa + beta3[0][2]*ya + beta3[0][3]*xa*ya + beta3[0][4]*xa**2 + \
              beta3[0][5]*ya**2 + beta3[0][6]*xa**2*ya + beta3[0][7]*xa*ya**2 + beta3[0][8]*xa**2*ya**2 + \
              beta3[0][9]*xa**3 + beta3[0][10]*ya**3 + beta3[0][11]*xa**3*ya + beta3[0][12]*xa*ya**3 + \
              beta3[0][12]*xa**3*ya**2 + beta3[0][13]*xa**2*ya**3

    elif poly_order > 3:
        print('Higher order than 3 is not yet implemented!')

    image_bkgcorr = image - bkg

    return image_bkgcorr

#Remove rolling ball background
def tophat_background(image, radius):

        # Define kernel
        kernel = disk(radius)

        # Make background image
        image_bkgcorr = white_tophat(image, kernel)

        ## Make backgground corrected image
        #image_bkgcorr = image - tophat_bkg

        return image_bkgcorr