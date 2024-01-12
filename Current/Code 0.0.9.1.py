import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, fmin
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline, interp1d
from scipy.signal import find_peaks, savgol_filter
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from numpy.polynomial.polynomial import Polynomial


# Alternate between Clockwise and Counterclockwise rotation


def load_data(*filenames):

    dataset_split = []
    data_list = [np.loadtxt(filename, delimiter=",") for filename in filenames]
    data = np.array(np.concatenate(data_list))

    time = data[:, 0]
    index = data[:, 1]
    xdata = data[:, 2]
    ydata = data[:, 3]
    ydata = ydata / np.max(ydata, axis=0, keepdims=True) # Normalizes the amplitude of the data to 0 - 1

    index_change = np.where(np.abs(np.diff(index)) > 500)[0] + 1 # the spot at which the index changes drasticly(like from 4096 to 0)

    ydata = np.split(ydata, index_change)
    xdata = np.split(xdata, index_change)

    for x in xdata:
        for i in range(1, len(x)):
            if abs(x[i] - x[i - 1]) > 180:  # Threshold can be adjusted
                direction = np.sign(get_slope(x))  # Determine direction
                adjustment = 360 * direction
                x[i:] += adjustment  # Adjust subsequent data

    C_xdata = []
    CC_xdata = []
    C_ydata = []
    CC_ydata = []

    for j in range(len(xdata)):
        if get_slope(xdata[j]) > 0:
            C_xdata.append(xdata[j])
            C_ydata.append(ydata[j])
        elif get_slope(xdata[j]) < 0:
            CC_xdata.append(xdata[j])
            CC_ydata.append(ydata[j])

    # Sort each subarray independently based on the corresponding xdata values
    def sort(xdata, ydata):
        for i in range(len(xdata)):
            sorted_indices = np.argsort(xdata[i])
            xdata[i] = xdata[i][sorted_indices]
            ydata[i] = ydata[i][sorted_indices]
        return xdata, ydata

    dataset_split.append((sort(C_xdata, C_ydata)))
    dataset_split.append((sort(CC_xdata, CC_ydata)))
    return dataset_split


def get_slope(xdata):
    y = [float(value) for value in xdata]
    x = list(range(len(y)))

    # looking for the transition from 0 to 360°, or the other way
    change_point =  np.where(np.abs(np.diff(y)) > 50)[0] + 1
    change_points = [0, change_point[0], len(x)] if change_point.size > 0 else [0, len(x)]

    slopes = [stats.linregress(x[change_points[i]:change_points[i+1]], y[change_points[i]:change_points[i+1]]).slope
            for i in range(len(change_points) - 1)]
    
    return np.mean(slopes)


def unique(xdata, ydata):

    xdata_unique, pos = np.unique(xdata, return_index=True)
    summed_ydata = np.add.reduceat(ydata, pos)
    counts = np.diff(np.append(pos, len(ydata)))
    ydata_unique = summed_ydata / counts

    return xdata_unique, ydata_unique



def spline_fit(xdata, ydata):    
    xdata, ydata = unique(xdata, ydata)

    spline = UnivariateSpline(xdata, ydata, k=4, s=6.25/len(xdata))

    fit = spline(xdata)
    res = np.zeros_like(xdata)
    for i in range(len(ydata)):
        res[i] = ydata[i] - fit[i]

    rms = np.sqrt(np.mean(res**2))
    
    spline_prime = spline.derivative()
    spline_sec_prime = spline_prime.derivative()

    maxima = []
    minima = []
    roots = spline_prime.roots()

    for root in roots:
        extrema = spline_sec_prime(root)
        if extrema > 0:
            minima.append(root)
        elif extrema < 0:
            maxima.append(root)

    return fit, spline_prime(xdata), res, rms, maxima, minima

def gradient_colorbar(colormap, ax, orientation, smoothing):

    num_smoothing = len(smoothing)
    max_ticks = 10
    num_ticks = min(max_ticks, num_smoothing)

    # Create a colorbar for smoothing values
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colormap(np.linspace(0, 1, num_smoothing)), N=num_smoothing)
    tick_positions = np.linspace(0, num_smoothing - 1, num_ticks)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_smoothing - 1)),
                        ax=ax, orientation=orientation, fraction=0.05, pad=0.07, aspect=40, ticks=tick_positions)
    cbar.set_label('Smoothing')

    cbar.set_ticklabels([f'{smoothing[int(position)]:.5f}' for position in tick_positions])


def process_data(xdata, ydata):
    extrema_collected = []
    splines_collected, derivative_collected, residuals_collected = [], [], []

    for loop_ind, loop_ydata in enumerate(ydata):      
        extrema = [[], [], [], []]
        spline, derivative, res_spline, rms_res_spline, maxima, minima = spline_fit(
            xdata[loop_ind], loop_ydata)
        
        extrema[0].append(minima[0] if minima else None)
        extrema[1].append(minima[1] if minima else None)
        extrema[2].append(maxima[0] if maxima else None)
        extrema[3].append(maxima[1] if maxima else None)
        
        splines_collected.append(spline)
        derivative_collected.append(derivative)
        residuals_collected.append(res_spline)
        extrema_collected.append(extrema)

        print(loop_ind)

    return extrema_collected, splines_collected, derivative_collected, residuals_collected, rms_res_spline

def plotting(rotation, ydata, xdata, ydata_cc, xdata_cc):

    extrema, splines, derivatives, residuals, rms = process_data(xdata, ydata)
    '''
    xdata_complete = [datum for x in xdata for datum in x]
    ydata_complete = [datum for y in ydata for datum in y]

    fig_raw, ax_raw = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig_res, ax_res = plt.subplots(1, 1, figsize=(15, 8))

    for i in range(len(xdata)):
        ax_raw[0].plot(np.unique(xdata[i]), splines[i], label='Spline', color='blue')
        ax_raw[1].plot(np.unique(xdata[i]), derivatives[i], label='Derivative',color='blue')
    
        ax_res.scatter(np.unique(xdata[i]), residuals[i] ,s=5, label='Residuals',color='red')

    ax_raw[0].scatter(xdata_complete, ydata_complete, label='Data', s=5, color='red')
    ax_raw[1].plot(xdata_complete, np.zeros(len(xdata_complete)), color='black')

    ax_raw[0].grid()
    ax_raw[1].grid()
    ax_res.grid()

    fig_raw.suptitle(f'Raw Data with Spline Fit {rotation}', fontsize=16)
    fig_res.suptitle(f'Residuals of spline {rotation}', fontsize=16)     


    # Plotting the resulting extrema from the spline fit
    fig_extr, ax_extr = plt.subplots(2, 2, figsize=(15, 8))
    ax_extr = ax_extr.flatten()


    names = ["minima_1", "minima_2", "maxima_1", "maxima_2"]

    indexes = list(range(len(extrema)))
    for i in range(4):  

        extrema_points = [extrema_set[i] for extrema_set in extrema]

        ax_extr[i].scatter(indexes, extrema_points, color='red')

        intercept = np.mean(extrema_points)
        residuals = extrema_points - intercept
        rms = np.sqrt(np.mean(residuals**2))
        
        ax_extr[i].set_title(names[i])
        ax_extr[i].grid()

    fig_extr.suptitle(f'Extrema of Splines {rotation}', fontsize=16)'''



    colormap = plt.cm.cividis
    fig_dat, ax_dat = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
    fig_ext, ax_ext = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
    fig_cen, ax_cen = plt.subplots(1, 1, figsize=(15, 8), sharex=True) 
    fig_rms, ax_rms = plt.subplots(1, 1, figsize=(15, 8), sharex=True)

    sigma_values = np.linspace(15, 40, 200)  

    # Calculate centers using the modified center_of_gravity function
    results = center_of_gravity(extrema, sigma_values, xdata, ydata)

    # Plotting loop
    ax_dat.scatter(results[-1]['xdata'], results[-1]['ydata'], s=5, color='blue')
    for index, extremum in enumerate(extrema):
        ax_ext.scatter(index, extremum[0][0], s=16, color='blue')
        ax_dat.axvline(x=extremum[0][0], color='red', linestyle='--')

        print(index)

    # Plotting loop for centers
    for i, result in enumerate(results):
        color = colormap(result['sigma'] / max(sigma_values))

        # Plot center lines
        ax_dat.axvline(x=result['center'], color=color, linestyle='--')

        # Plot center values
        ax_cen.scatter(result['extremum_index'], result['center'], s=16, color=color)

        
        
    sigmas, rmss = [], []
    for i, sigma in enumerate(sigma_values):
        filtered_results = [res for res in results if res['sigma'] == sigma]
        center_points = [res['center'] for res in filtered_results]

        color = colormap(i / (len(sigma_values) - 1))

        intercept = np.mean(center_points)
        residuals = center_points - intercept
        rms = np.sqrt(np.mean(residuals**2))

        sigmas.append(sigma)
        rmss.append(rms)
        ax_rms.scatter(sigma, rms, s=16, color=color)

    # Find the minimum sigma using the updated function
    min_sigma, polynomial, = find_minimum_sigma(sigmas, rmss)

    dense_sigmas = np.linspace(min(sigmas), max(sigmas), 500)
    fitted_rmss = polynomial(dense_sigmas)

    # Add polynomial fit and minimum sigma line to the existing ax_rms plot
    ax_rms.plot(dense_sigmas, fitted_rmss, color='purple', label='Polynomial Fit')
    ax_rms.axvline(x=min_sigma, color='green', linestyle='--', label=f'Minimum at {min_sigma:.2f}')

    print('minimum sigma:', min_sigma)


    # Set plot titles
    ax_dat.set_title('Data')
    ax_ext.set_title('Extrema Plot')
    ax_cen.set_title('Center Plot')
    ax_rms.set_title('RMS Plot')

def find_minimum_sigma(sigmas, rmss):
    # Fit a polynomial to the data
    polynomial_degree = 9  # Or the degree that best fits your data
    coeffs = np.polyfit(sigmas, rmss, polynomial_degree)
    p = np.poly1d(coeffs)  # Represent the polynomial using np.poly1d

    # Find the roots of the derivative of the polynomial (the extrema)
    deriv = np.polyder(p)
    der_roots = np.roots(deriv)

    # Filter the roots to find the real ones within the range of sigma values
    real_roots = der_roots[np.isreal(der_roots)].real
    real_roots_in_range = real_roots[(real_roots >= min(sigmas)) & (real_roots <= max(sigmas))]

    # Assume there are real roots within the range and find the minimum
    real_minima = p(real_roots_in_range)
    min_index = np.argmin(real_minima)
    min_sigma = real_roots_in_range[min_index]

    return min_sigma, p


def center_of_gravity(extrema, sigma_values, xdata_all, ydata_all, boundary_multiplier=4, max_iterations=1000, convergence_tolerance=1e-6):
    """
    Optimized function to calculate the center of gravity for given data.

    Args:
    extrema (list): List of extrema points.
    sigma_values (list): List of sigma values for calculation.
    xdata_all (list): List of all x data points.
    ydata_all (list): List of all y data points.
    boundary_multiplier (int, optional): Multiplier for boundary calculation. Default to 4.
    max_iterations (int, optional): Maximum number of iterations. Default to 1000.
    convergence_tolerance (float, optional): Tolerance for convergence. Default to 1e-6.

    Returns:
    list: A list of dictionaries containing the results for each extremum and sigma value.
    """

    results = []
    for index, extremum in enumerate(extrema):
        
        ydata_original = ydata_all[index]
        xdata_original = xdata_all[index]
        max_ydata = np.max(ydata_original)  # Compute once, outside the loop
        ydata_original = max_ydata - ydata_original
        minimum_value = extremum[0][0]

        for sigma in sigma_values:
            centers = []
            bounds = []
            weighted_values = []

            for iteration in range(max_iterations):
                # Calculate bounds
                if iteration == 0:
                    lower_bound = minimum_value - boundary_multiplier * sigma
                    upper_bound = minimum_value + boundary_multiplier * sigma
                else:
                    lower_bound = centers[-1] - boundary_multiplier * sigma
                    upper_bound = centers[-1] + boundary_multiplier * sigma

                bounds.append((lower_bound, upper_bound))

                # Apply mask to data
                mask = (xdata_original > lower_bound) & (xdata_original < upper_bound)
                masked_xdata = xdata_original[mask]
                masked_ydata = ydata_original[mask]

                # Calculate weighted sum
                if iteration == 0:
                    weighted_sum = masked_ydata
                else:
                    weighted_sum = masked_ydata * np.exp(-(masked_xdata - centers[-1])**2 / (sigma**2))

                center = np.sum(masked_xdata * weighted_sum) / np.sum(weighted_sum)

                # Check for convergence every few iterations to save computation time
                if iteration > 0 and iteration % 5 == 0 and abs(centers[-1] - center) < convergence_tolerance:
                    break

                centers.append(center)
                weighted_values.append(weighted_sum)

            center_std = np.std(centers) if len(centers) > 1 else None

            results.append({'center': centers[-1], 'error': center_std, 'bounds': bounds[-1], 
                            'weighted_values': weighted_values[-1], 'mask': mask, 
                            'extremum_index': index, 'sigma': sigma,
                            'xdata': masked_xdata, 'ydata': masked_ydata})

    return results



def main(*data):
    dataset = load_data(*data)
    rotation = ["Clockwise", "Counterclockwise"]
    xdata_c, ydata_c = dataset[1]
    xdata_cc, ydata_cc = dataset[1]

    plotting(rotation[1], ydata_c, xdata_c, ydata_cc, xdata_cc)

        
    plt.show()

main("data\-9cs1a.txt")

