import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.signal import find_peaks, savgol_filter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy import stats


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

    return extrema_collected, splines_collected, derivative_collected, residuals_collected

def plotting(rotation, ydata, xdata, ydata_cc, xdata_cc):

    extrema, splines, derivatives, residuals = process_data(xdata, ydata)
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



    a = 6
    ydata = ydata[a]
    xdata = xdata[a]

    angle = 110
    phase_shift = np.pi/2
    #xdata = np.linspace(angle, angle + 360, 4096)
    #ydata = np.sin(np.radians(xdata) + phase_shift)**2

    print(extrema[a][2][0], extrema[a][3][0])
    

    ydata = np.max(ydata) - ydata
    #ydata_cc[a] = np.max(ydata_cc[a]) - ydata_cc[a]



    fig_box, ax_box = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
    ax_box.scatter(xdata, ydata, label='Data', s=5, color='red')

    sigma = 27.59
    iterations = 10000
    minimum = extrema[a][0][0]
    #minimum = 271
    print(minimum)

    # Creating a loop for the given process
    sums, bounds, gs, mask = center_of_gravity(minimum, iterations, sigma, xdata, ydata)

    ax_box.scatter(xdata[mask], gs[-1], label=f'final iteration', s=5, color = 'green')

    for bound in bounds:
        ax_box.axvline(x=bound[0], color='r', linestyle='--')
        ax_box.axvline(x=bound[1], color='r', linestyle='--')

    ax_box.axvline(x=minimum, color='red', linestyle='--')

    for sum in sums:
        ax_box.axvline(x=sum, color='purple', linestyle='--')

    ax_box.grid()
    fig_box.suptitle(f'Raw Data', fontsize=16)

    print(sums[-1])

    sigma_values = np.linspace(1, 70, 400)  

    fig_it, ax_it = plt.subplots(1, 1, figsize=(15, 8))
    for i, sum in enumerate(sums):
        ax_it.scatter(i, sum, color='blue')

    fig_sigma, ax_sigma = plt.subplots(1, 1, figsize=(15, 8))

    for sigma in sigma_values:
        sums, bounds, gs, mask = center_of_gravity(minimum, iterations, sigma, xdata, ydata)

        ax_sigma.scatter(sigma, sums[-1], color='blue')
        ax_sigma.axhline(y=minimum, color='r', linestyle='--')


def center_of_gravity(minimum, iterations, sigma, xdata, ydata, b=4, tolerance=1e-7):
    sums = []
    bounds = []
    gs = []
    for i in range(iterations):  
        if i == 0:
            bound_l = minimum - b * sigma
            bound_u = minimum + b * sigma
        else:
            bound_l = sums[-1] - b * sigma
            bound_u = sums[-1] + b * sigma

        bounds.append((bound_l, bound_u))
        mask = (xdata > bound_l) & (xdata < bound_u)

        if i == 0:
            g = ydata[mask] * np.exp(-(xdata[mask] - sums[-1])**2 / (sigma**2)) if sums else ydata[mask]
        else:
            g = ydata[mask] * np.exp(-(xdata[mask] - sums[-1])**2 / (sigma**2))
        sum_g = np.sum(xdata[mask] * g) / np.sum(g)

        if sums and abs(sums[-1] - sum_g) < tolerance:
            break

        sums.append(sum_g)
        gs.append(g)
    return sums, bounds, gs, mask


def main(*data):
    dataset = load_data(*data)
    rotation = ["Clockwise", "Counterclockwise"]
    xdata_c, ydata_c = dataset[0]
    xdata_cc, ydata_cc = dataset[1]

    plotting(rotation[0], ydata_c, xdata_c, ydata_cc, xdata_cc)

        
    plt.show()

main("data\-9c9.txt")

