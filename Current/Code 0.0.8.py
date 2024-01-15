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
    ydata = data[:, 3:]
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



def spline_fit(xdata, ydata, smoothing):    
    xdata, ydata = unique(xdata, ydata)

    spline = UnivariateSpline(xdata, ydata, k=4, s=smoothing/len(xdata))

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

def process_data(xdata, ydata, smoothing):
    extrema, splines_collected, derivative_collected, residuals_collected = [], [], [], []

    for loop_ind, loop_ydata in enumerate(ydata):
        extrema_loop = [[], [], [], []]
        splines_loop, derivatives_loop, residuals_loop = [], [], []
        
        for color_ind in range(loop_ydata.shape[1]):
            for smooth in smoothing:
                spline, derivative, res_spline, rms_res_spline, maxima, minima = spline_fit(
                    xdata[loop_ind], loop_ydata[:, color_ind], smooth)
                
                extrema_loop[0].append(minima[0] if minima else None)
                extrema_loop[1].append(minima[1] if minima else None)
                extrema_loop[2].append(maxima[0] if maxima else None)
                extrema_loop[3].append(maxima[1] if maxima else None)
                
                splines_loop.append(spline)
                derivatives_loop.append(derivative)
                residuals_loop.append(res_spline)

        extrema.append(extrema_loop)
        splines_collected.append(splines_loop)
        derivative_collected.append(derivatives_loop)
        residuals_collected.append(residuals_loop)

        print(loop_ind)

    return extrema, splines_collected, derivative_collected, residuals_collected

def plotting(rotation, ydata, xdata, smoothing):

    extrema, splines, derivatives, residuals = process_data(xdata, ydata, smoothing)


    xdata_complete = [datum for x in xdata for datum in x]
    ydata_complete = [datum for y in ydata for color_data in y for datum in color_data]

    colormap = plt.cm.cividis
        # PLotting the raw data and the corresponding spline fits and dereivatives
    fig_raw, ax_raw = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig_res, ax_res = plt.subplots(1, 1, figsize=(15, 8))

    for i in range(len(xdata)):
        for j, smooth in enumerate(smoothing):
            color = colormap(j / (len(smoothing)-1))
            ax_raw[0].plot(np.unique(xdata[i]), splines[i][j], label='Spline', color=color)
            ax_raw[1].plot(np.unique(xdata[i]), derivatives[i][j], label='Derivative',color=color)
        
            ax_res.scatter(np.unique(xdata[i]), residuals[i][j] ,s=5, label='Residuals',color=color)

    ax_raw[0].scatter(xdata_complete, ydata_complete, label='Data', s=5, color='red')
    ax_raw[1].plot(xdata_complete, np.zeros(len(xdata_complete)), color='black')

    ax_raw[0].grid()
    ax_raw[1].grid()
    ax_res.grid()

    gradient_colorbar(colormap, ax_raw, 'vertical', smoothing)
    gradient_colorbar(colormap, ax_res, 'vertical', smoothing)
    fig_raw.suptitle(f'Raw Data with Spline Fit {rotation}', fontsize=16)
    fig_res.suptitle(f'Residuals of spline {rotation}', fontsize=16)     


    # Plotting the resulting extrema from the spline fit
    fig_extr, ax_extr = plt.subplots(2, 2, figsize=(15, 8))
    ax_extr = ax_extr.flatten()

    # PLotting the Rms from the induvidual extrema and their resulting average line
    fig_rms, ax_rms = plt.subplots(2, 2, figsize=(15, 8))
    ax_rms = ax_rms.flatten()


    names = ["minima_1", "minima_2", "maxima_1", "maxima_2"]
    
    extrema_filtered = [item for item in extrema if any(item)]
    indexes = list(range(len(extrema_filtered)))
    for i in range(4):  # Loop over different types of extrema

        for j, smooth_value in enumerate(smoothing):
            extrema_points = [extrema_set[i][j] for extrema_set in extrema_filtered]
            color = colormap(j / (len(smoothing) - 1))

            ax_extr[i].scatter(indexes, extrema_points, color=color)

            intercept = np.mean(extrema_points)
            # horizontal_line = np.full_like(extrema_points, intercept)
            # ax_extr[i].plot(indexes, horizontal_line, color=color)

            residuals = extrema_points - intercept
            rms = np.sqrt(np.mean(residuals**2))

            ax_rms[i].scatter(smooth_value, rms, color=color)
        

        ax_extr[i].set_title(names[i])
        ax_rms[i].set_title(names[i])
        
        ax_extr[i].grid()
        ax_rms[i].grid()



    gradient_colorbar(colormap, ax_extr, 'horizontal', smoothing)
    gradient_colorbar(colormap, ax_rms, 'horizontal', smoothing)

    fig_extr.suptitle(f'Extrema of Splines {rotation}', fontsize=16)
    fig_rms.suptitle(f'Rms of Extrema {rotation}', fontsize=16)


def main(smoothing, *data):
    dataset = load_data(*data)
    rotation = ["Clockwise", "Counterclockwise"]

    for k, (xdata, ydata) in enumerate(dataset):
        print(rotation[k])

        plotting(rotation[k], ydata, xdata, smoothing)
        


    plt.show()

smoothing = np.linspace(4.5, 6.5, 2)
main(smoothing, "data\-9c9.txt")

