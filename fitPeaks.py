#!/usr/bin/python3.7

"""
This function reads in an atomic resolution (S)TEM image, or takes it as input through calling this function in another
script. It then applies a Butterworth and Gaussian filter to smooth out noise and does a simple peak fitting. The
user has then the option to refine the peak positions by non-linear Gaussian fitting.
Input: atomic resolution (S)TEM image
Output: peaks
        - normal peak fitting: [x, y, peak intensity]
"""

# Libraries
import numpy as np
from skimage.filters import gaussian, butterworth


def fitPeaks(image, min_peak_dist: int, min_peak_int: bool,
             smooth_gauss: int):

    # # Normalize image
    # image = image - np.mean(image)
    # image = normImg(image)

    # Determine image properties
    # shape no. rows (Nx) x no. columns (Ny) x no. channels (Nz)
    (Ny, Nx) = image.shape

    # Apply Gaussian filter
    image_filt = gaussian(image, sigma=smooth_gauss, mode='wrap')

    # Determine lattice peaks with indices x_p, y_p
    image_mid = image_filt[1:-1, 1:-1]
    peaks = np.logical_and.reduce((image_mid > image_filt[0:-2, 0:-2],
                                   image_mid > image_filt[1:-1, 0:-2],
                                   image_mid > image_filt[2:, 0:-2],
                                   image_mid > image_filt[0:-2, 1:-1],
                                   image_mid > image_filt[2:, 1:-1],
                                   image_mid > image_filt[0:-2, 2:],
                                   image_mid > image_filt[1:-1, 2:],
                                   image_mid > image_filt[2:, 2:]))

    (x_p, y_p) = np.nonzero(np.multiply(peaks, image_mid))
    num_peaks = int(np.size(x_p))

    x_p = x_p.reshape(num_peaks, 1)
    x_p += 1
    y_p = y_p.reshape(num_peaks, 1)
    y_p += 1
    I_p = image_filt[x_p, y_p]

    peaks = np.zeros((num_peaks, 3))
    peaks[:, :3] = np.hstack((x_p, y_p, I_p))
    peaks = peaks[np.argsort(1*peaks[:, 2])]
    x_p = peaks[:, 0]
    y_p = peaks[:, 1]

    # Remove peaks too close together
    del_peak = np.ones(num_peaks, dtype=bool)
    min_dist = min_peak_dist
    for a0 in range(0, num_peaks - 1, 1):
        d2 = (x_p[a0] - x_p[a0 + 1:]) ** 2 + (y_p[a0] - y_p[a0 + 1:]) ** 2

        if np.min(d2) < (min_dist ** 2):
            del_peak[a0] = False

    peaks = peaks[del_peak, :]

    # Remove low intensity peaks
    min_int = min_peak_int
    min_peaks = peaks[:, 2] > min_int
    peaks = peaks[min_peaks, :]

    return peaks


def fitPeaks2DGaussian(image, peaks):

    # Determine image properties
    # shape no. rows (Nx) x no. columns (Ny) x no. channels (Nz)
    (Nx, Ny) = image.shape

    # Inital parameters for non-linear peak fitting
    peaks_Nx, peaks_Ny = np.shape(peaks)
    num_peaks = peaks_Nx

    Nsub_pix_iterations = 3     # number of sub-pixel iterations
    d_xy = 0.5                  # max allowed shift for fitting
    rCut = 5                    # size of cutting area around inital peak
    rFit = 4                    # size of fitting radius
    sigma0 = 5
    sigmaMin = 2
    sigmaMax = 9
    #damp = 2/3                  # Damping rate

    # Fitting coordinates
    x_coord = np.arange(0, Ny, 1)
    y_coord = np.arange(0, Ny, 1)
    x_a, y_a = np.meshgrid(x_coord, y_coord, sparse=False)

    # Define 2D Gaussian function
    def func(c, x_func, y_func, int_func):
        return c[0] * np.exp(-1/2 / c[1] ** 2 * ((x_func - c[2]) ** 2 + (y_func - c[3]) ** 2)) + c[4] - int_func

    # Loop through inital peaks and fit by non-linear Gaussian functions
    peaks_refine = []
    for p0 in range(0, num_peaks, 1):
        # Initial peak positions
        x = peaks[p0, 0]
        xc = np.rint(x).astype(int, casting='unsafe')
        y = peaks[p0, 1]
        yc = np.rint(y).astype(int, casting='unsafe')

        # Cut out subsection around the peak
        x_sub = np.arange(np.max((xc-rCut, 0)), np.min((xc + rCut, Ny)) + 2, 1)
        y_sub = np.arange(np.max((yc-rCut, 0)), np.min((yc+rCut, Ny)) + 2, 1)

        # Make indices of subsection
        x_cut = x_a[y_sub[0]: y_sub[-1], x_sub[0]: x_sub[-1]]
        y_cut = y_a[y_sub[0]: y_sub[-1], x_sub[0]: x_sub[-1]]
        cut = image[x_cut, y_cut]

        # Inital values for least-squares fitting
        k = np.min(cut)
        int_0 = np.max(cut)-k
        sigma = sigma0

        # Sub-pixel iterations
        for s0 in range(0, 3, 1):
            sub = (x_cut - x) ** 2 + (y_cut - y) ** 2 < rFit ** 2

            # Fitting coordinates
            x_fit = x_cut[sub]
            y_fit = y_cut[sub]
            int_fit = cut[sub]

            # Initial guesses and bounds of fitting function
            c0 = [int_0, sigma, x, y, k]
            lower_bnd = [int_0*.8, max(sigma*0.8, sigmaMin), x-d_xy, y-d_xy, k-int_0*0.5]
            upper_bnd = [int_0*1.2, min(sigma*1.2, sigmaMax), x+d_xy, y+d_xy, k+int_0*0.5]

            # Linear least squares fitting
            peak_fit = least_squares(func, c0, args=(x_fit, y_fit, int_fit), bounds=(lower_bnd, upper_bnd))
            image_fit = peak_fit.x[0] * np.exp(-1/2 / peak_fit.x[1] ** 2 * ((x_cut - peak_fit.x[2]) ** 2 + (y_cut -
                                                                                                            peak_fit.x[3]) ** 2)) + peak_fit.x[4]


        # Refined peak positions
        int_0 = peak_fit.x[0]
        sigma = peak_fit.x[1]
        x = peak_fit.x[2]
        y = peak_fit.x[3]
        k = peak_fit.x[4]

        # Write refined peak array: sigma (of Gaussian), x, y, k (fitted background level), peak_int (integrated peak
        # intensity without background)
        peak_int = np.sum(cut * sub)  # integrated intensity without background
        peaks_refine.append([sigma, x, y, k, peak_int])

        # Display fitting progress
        if p0 % 50 == 0:
            comp =  (p0 + 1) / num_peaks
            print('Fitting is ' + str("{:.2f}".format(comp*100)) + '% complete!')

    # Reshape refined peak array
    peaks_refine = np.asarray(peaks_refine)
    peaks_refine = peaks_refine.reshape(num_peaks, 5)

    # Combine peaks from simple fitting with refined peak positions
    # Column meaning of peaks:
    # Col 1: x coordinate of original peak
    # Col 2: y coordinate of original peak
    # Col 3: peak intensity of original peak
    # Col 4: sigma (standard deviation) of Gaussian function
    # Col 5: refined x coordinate
    # Col 6: refined y coordinate
    # Col 7: background value determined by fit
    # Col 8: integrated peak intensity without background

    return peaks_refine









