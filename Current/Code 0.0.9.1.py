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
    splines_collected, derivative_collected, residuals_collected, rms_collected = [], [], [], []

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
        rms_collected.append(rms_res_spline)

        print(loop_ind)

    return extrema_collected, splines_collected, derivative_collected, residuals_collected, rms_collected

def plotting(rotation, ydata, xdata,):

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

    # Calculate centers using the modified find_center function
    results = find_center(extrema, sigma_values, xdata, ydata, rms)

    all_xdata = np.concatenate([res['xdata'] for res in results])
    all_ydata = np.concatenate([res['ydata'] for res in results])

    # Plotting loop
    ax_dat.scatter(all_xdata, all_ydata, s=5, color='blue')
    for index, extremum in enumerate(extrema):
        ax_ext.scatter(index, extremum[0][0], s=16, color='blue')
        ax_dat.axvline(x=extremum[0][0], color='red', linestyle='--')

        print(index)

    # Plotting loop for centers
    '''
    for i, result in enumerate(results):
        color = colormap(result['sigma'] / max(sigma_values))

        # Plot center lines
        ax_dat.axvline(x=result['center'], color=color, linestyle='--')

        # Plot center values
        ax_cen.scatter(result['extremum_index'], result['center'], s=16, color=color)'''

        
        
    sigmas, rmss = [], []
    for i, sigma in enumerate(sigma_values):
        filtered_results = [res for res in results if res['sigma'] == sigma]
        center_points = [res['center'] for res in filtered_results]

        color = colormap(i / (len(sigma_values) - 1))

        intercept = np.mean(center_points)
        residuals = center_points - intercept
        rms_sigma = np.sqrt(np.mean(residuals**2))

        sigmas.append(sigma)
        rmss.append(rms_sigma)
        ax_rms.scatter(sigma, rms_sigma, s=16, color=color)

    # Find the minimum sigma using the updated function
    min_sigma, polynomial, = find_minimum_sigma(sigmas, rmss)

    dense_sigmas = np.linspace(min(sigmas), max(sigmas), 500)
    fitted_rmss = polynomial(dense_sigmas)

    # Add polynomial fit and minimum sigma line to the existing ax_rms plot
    ax_rms.plot(dense_sigmas, fitted_rmss, color='purple', label='Polynomial Fit')
    ax_rms.axvline(x=min_sigma, color='green', linestyle='--', label=f'Minimum at {min_sigma:.2f}')

    print('minimum sigma:', min_sigma)

    result_optm = find_center(extrema, min_sigma, xdata, ydata, rms)

    
    errors = []
    centers = [] 
    for i, result in enumerate(result_optm):
        std_deviation = result['std_deviation']
        error = result['error']
        centers.append(result['center'])
        errors.append(error)

        ax_dat.axvline(x=result['center'], color='green', linestyle='--')
        ax_cen.errorbar(result['extremum_index'], result['center'], yerr=error, fmt='o', color='green', ecolor='lightgreen', capsize=5, label='Center with error')

    center_mean = np.mean(centers)
    error_mean = np.mean(errors)
    print(f"Mean of centers: {center_mean}" )
    print(f"Individual Errors: ±{error_mean}" )

    # Set plot titles
    ax_dat.set_title('Data')
    ax_ext.set_title('Extrema Plot')
    ax_cen.set_title('Center Plot')
    ax_rms.set_title('RMS Plot')



def find_minimum_sigma(sigmas, rmss):

    polynomial_degree = 10
    coeffs = np.polyfit(sigmas, rmss, polynomial_degree)
    p = np.poly1d(coeffs) 

    deriv = np.polyder(p)
    der_roots = np.roots(deriv)


    real_roots = der_roots[np.isreal(der_roots)].real
    real_roots_in_range = real_roots[(real_roots >= min(sigmas)) & (real_roots <= max(sigmas))]


    real_minima = p(real_roots_in_range)
    min_index = np.argmin(real_minima)
    min_sigma = real_roots_in_range[min_index]

    return min_sigma, p


def find_center(extrema, sigma_values, xdata_all, ydata_all, rms, boundary_multiplier=3, max_iterations=1000, convergence_tolerance=1e-6):
    sigma_values = np.atleast_1d(sigma_values)

    results = []
    for index, extremum in enumerate(extrema):
        
        ydata_o = ydata_all[index]
        xdata_o = xdata_all[index]
        max_ydata = np.max(ydata_o)
        ydata_o = max_ydata - ydata_o
        minimum_value = extremum[0][0]

        for sigma in sigma_values:
            centers = [minimum_value]  # Initialize with the first center value
            bounds = []
            gdata = []

            for iteration in range(max_iterations):
                # Calculate bounds
                lower_bound = centers[-1] - boundary_multiplier * sigma
                upper_bound = centers[-1] + boundary_multiplier * sigma

                bounds.append((lower_bound, upper_bound))

                mask = (xdata_o > lower_bound) & (xdata_o < upper_bound)
                x = xdata_o[mask]
                y = ydata_o[mask]
                x = x - centers[-1]

                g = y * np.exp(-((x)**2 / (sigma**2)))
                center = np.sum(x * g) / np.sum(g) + centers[-1]

                if iteration > 0 and abs(centers[-1] - center) < convergence_tolerance:
                    break

                centers.append(center)
                gdata.append(g)

            center_std = np.std(centers) if len(centers) > 1 else None
            error = np.sqrt(np.sum(rms[index]**2 * (x)**2) / (np.sum(g)**2))

            results.append({
                'center': centers[-1], 
                'std_deviation': center_std, 
                'error': error, 
                'bounds': bounds[-1], 
                'gdata': gdata[-1], 
                'mask': mask, 
                'extremum_index': index, 
                'sigma': sigma,
                'xdata': xdata_o[mask], 
                'ydata': y
            })

    return results


def main(*data):
    dataset = load_data(*data)
    rotation = ["Clockwise", "Counterclockwise"]

    k=0
    xdata, ydata = dataset[k]

    plotting(rotation[k], ydata, xdata)

    plt.show()

main( "data\-9c9.txt")

