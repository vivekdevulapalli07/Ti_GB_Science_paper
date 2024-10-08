#!/usr/bin/python3.7

"""
This function computes the Fast Fourier Transform (FFT) of a high resolution image, applies a gaussian filter,
adjusts the intensity threshold and cuts out the central region of interest. This function can be called from any
other script.
Input: a 2d image, the kernel size of the gaussian filter (needs to be odd) and the sigma value. k_size is typically
7 or 9, sigma 4.
Output: the smoothed FFT of the image fft_blur
"""

# Libraries
import numpy as np
from numpy import square, abs, fft, log, sqrt, max, arange, rint, meshgrid
from skimage.filters import window
from skimage.filters import gaussian
from imageEnhance import normImg
import matplotlib.pyplot as plt


def fft2_raw(image):

    # Take FFT of image
    fft2 = fft.fft2(image)
    fft2_shift = fft.fftshift(fft2)
    fft2 = fft2_shift

    return fft2


def fft2_blur_peaks(image, gauss_sigma: float, hann: bool):

    #If hanning window is selected
    if hann:
        image = image * window('hann', image.shape)

    # Take FFT of image
    fft2 = fft.fft2(image)
    fft2_shift = fft.fftshift(fft2)
    fft2 = abs(fft2_shift)

    # Smooth FFT with gaussian filter: k_size has to be odd!
    fft2_blur = gaussian(fft2, sigma=gauss_sigma, mode='wrap', truncate=2.0)
    fft2_blur = normImg(fft2_blur)

    # Make mask to boost contrast and only show central part of FFT
    (Ny, Nx) = fft2.shape
    x = arange(-Nx / 2, Nx / 2, 1)
    y = arange(-Ny / 2, Ny / 2, 1)
    xv, yv = meshgrid(x, y, sparse=False)
    mask_radius = sqrt(square(xv) + square(yv))
    mask = mask_radius > 20
    fft2_masked = fft2_blur * mask
    fft2_masked_max = max(fft2_masked)
    fft2_blur = fft2_blur / fft2_masked_max
    fft2_blur[fft2_blur > 1] = 1

    # # Display FFT (using matplotlib)
    # plt.clf()
    # plt.figure(1)
    # plt.imshow(fft2_blur, cmap='plasma')
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    # plt.close()

    return fft2_blur


def fft2_blur(image, gauss_sigma: float, hann: bool, crop: float, scale: bool, cont_boost: float):

    #If hanning window is selected
    if hann:
        image = image * window('hann', image.shape)

    # Take FFT of image
    fft2 = fft.fft2(image)
    fft2_shift = fft.fftshift(fft2)

    # Make mask to boost contrast and only show central part of FFT
    if scale in ['log', 'Log']:
        fft2 = log(abs(fft2_shift))

        # Smooth FFT with gaussian filter: k_size has to be odd!
        fft2_blur = gaussian(fft2, sigma=gauss_sigma, mode='wrap', truncate=2.0)

        (Ny, Nx) = fft2.shape
        x = arange(-Nx / 2, Nx / 2, 1)
        y = arange(-Ny / 2, Ny / 2, 1)
        xv, yv = meshgrid(x, y, sparse=False)
        mask_radius = sqrt(square(xv) + square(yv))
        mask = mask_radius > cont_boost
        fft2_masked = fft2_blur * mask
        fft2_masked_max = max(fft2_masked)
        fft2_blur = fft2_blur / fft2_masked_max
        fft2_blur[fft2_blur > 1] = 1

    if scale in ['lin', 'Lin']:
        fft2 = abs(fft2_shift)

        # Smooth FFT with gaussian filter: k_size has to be odd!
        fft2_blur = gaussian(fft2, sigma=gauss_sigma, mode='wrap', truncate=2.0)

        (Ny, Nx) = fft2.shape
        fft2_blur = fft2_blur / (max(fft2_blur) * cont_boost/100)
        fft2_blur[fft2_blur > 1] = 1

    fft2_blur = fft2_blur[rint(Ny * crop).astype(int, casting='unsafe'):rint(Ny * (1-crop)).astype(int,
                                                                                                  casting='unsafe'),
                rint(Nx * crop).astype(int, casting='unsafe'):rint(Nx * (1-crop)).astype(int, casting='unsafe')]

    # # Display FFT (using matplotlib)
    # plt.clf()
    # plt.figure(1)
    # plt.imshow(fft2_blur, cmap='plasma')
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    # plt.close()

    return fft2_blur


def fourier_mask(image, peaks, mask_radius: int, gauss_sigma: float):

    (Ny, Nx) = np.shape(image)

    #Make coordinates for mask
    ym, xm = meshgrid(np.arange(0, Nx, 1), np.arange(0, Ny, 1), sparse=False)
    mask = np.zeros((Ny, Nx))

    #Mask each selected peak
    num_peaks = np.shape(peaks)
    num_peaks = num_peaks[0]
    for a0 in range(0, num_peaks, 1):
        #Peak and mask coordinates
        x_p = peaks[a0, 0]
        y_p = peaks[a0, 1]
        mask_peak = (xm-x_p) ** 2 + (ym-y_p) ** 2 < mask_radius
        mask = mask + mask_peak

    #Smooth mask
    mask = gaussian(mask, sigma=gauss_sigma, mode='wrap', truncate=2.0)
    mask = normImg(mask)

    return mask


def inverse_fft2(image):

    inv_fft2 = fft.ifftshift(image)
    inv_fft2 = fft.ifft2(inv_fft2)
    inv_fft2 = np.real(inv_fft2)
    inv_fft2 = normImg(inv_fft2)

    return inv_fft2



