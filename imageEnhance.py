#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on 30.11.21 at 21:09 by c.liebscher

"""

#Libraries
import numpy as np

#Normalize image
def normImg(image):

    normalized_image = (image - np.min(image)) / ((np.max(image) - np.min(image)))

    return normalized_image


#Auto adjust image threshold
def autoThresh(image, stddev):

    image = normImg(image)
    image_mean = image.mean()
    image_std = image.std()
    image[image > image_mean+stddev*image_std] = image_mean+stddev*image_std
    image[image < image_mean-stddev*image_std] = image_mean-stddev*image_std
    image = normImg(image)

    return image


#Apply Butterworthfiler
def bwfilter(image, inc, order):

    # Subtract mean value from image
    (Nx_init, Ny_init) = image.shape
    Ny = Nx_init
    Nx = Ny_init
    image = image - np.mean(image)

    # Generate Fourier coordinates in 2D
    psize = 1 / inc
    lsize = np.array([Nx, Ny]) * psize

    if Nx % 2 == 0:
        qx = np.roll((np.arange(-Nx / 2, Nx / 2) / lsize[0]), np.rint(-Nx / 2).astype(int, casting='unsafe'),
                     axis=0)
    else:
        qx = np.roll((np.arange(-Nx / 2 + .5, Nx / 2 - .5) / lsize[0]),
                     np.rint(-Nx / 2 + .5).astype(int, casting='unsafe'), axis=0)
    if Ny % 2 == 0:
        qy = np.roll((np.arange(-Ny / 2, Ny / 2) / lsize[0]), np.rint(-Ny / 2).astype(int, casting='unsafe'),
                     axis=0)
    else:
        qy = np.roll((np.arange(-Ny / 2 + .5, Ny / 2 - .5) / lsize[0]),
                     np.rint(-Ny / 2 + .5).astype(int, casting='unsafe'), axis=0)

    # Apply low pass Butterworth filter
    qxa, qya = np.meshgrid(qx, qy, sparse=True)
    q2 = np.square(qxa) + np.square(qya)
    wfilt = 1 - 1 / np.sqrt(1 + np.power(q2, order) / np.power(.5, 16))

    butterfilt = np.real(np.fft.ifft2(np.fft.fft2(image) * wfilt))

    return butterfilt


#Remove polynomial background
def removePolyBackground(image, poly_order: int):

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
