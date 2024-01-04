import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
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
    ydata_dict = {}
    for x, y in zip(xdata, ydata):
        if x in ydata_dict:
            ydata_dict[x].append(y)
        else:
            ydata_dict[x] = [y]

    ydata_unique = {x: np.mean(y) for x, y in ydata_dict.items()}
    xdata_unique = np.array(list(ydata_unique.keys()))
    ydata_unique = np.array(list(ydata_unique.values()))
    return xdata_unique, ydata_unique



def spline_fit(xdata, ydata, smoothing):    
    xdata, ydata = unique(xdata, ydata)
    spline = UnivariateSpline(xdata, ydata, k=4, s=smoothing)
    fit = spline(xdata)
    res = np.zeros_like(xdata)
    for i in range(len(ydata)):
        res[i] = ydata[i] - fit[i]

    rms = np.sqrt(np.mean(res**2))

    peaks, _ = find_peaks(ydata)
    valleys, _ = find_peaks(-ydata)

    
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

def gradient_colorbar(colormap, ax, orientation, smoothing_values):

    num_smoothing_values = len(smoothing_values)
    max_ticks = 10
    num_ticks = min(max_ticks, num_smoothing_values)

    # Create a colorbar for smoothing values
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colormap(np.linspace(0, 1, num_smoothing_values)), N=num_smoothing_values)
    tick_positions = np.linspace(0, num_smoothing_values - 1, num_ticks)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_smoothing_values - 1)),
                        ax=ax, orientation=orientation, fraction=0.05, pad=0.07, aspect=40, ticks=tick_positions)
    cbar.set_label('Smoothing')

    cbar.set_ticklabels([f'{smoothing_values[int(position)]:.5f}' for position in tick_positions])


def main(smoothing, *data):
    dataset = load_data(*data)
    rotation = ["Clockwise", "Counterclockwise"]
    # k determimens the direction of the data, k=0 is clockwise and k=1 is counterclockwise
    for k in range(len(dataset[0])):

        print(rotation[k])

        xdata, ydata = dataset[k]
        
        xdata_complete = [data for j in range(len(ydata)) for data in xdata[j]]
        ydata_complete = [data for j in range(len(ydata)) for i in range(ydata[j].shape[1]) for data in ydata[j][:, i]]

        extrema = []
        splines_collected = []
        derivative_collected = []
        residuals_collected = []

        for j in range(len(ydata)):                 # looping through each loop of the data
            minima_1 = []
            minima_2 = []
            maxima_1 = []
            maxima_2 = []

            splines_j = []
            derivatives_j = []
            residuals_j = []
            
            for i in range(ydata[j].shape[1]):      # looping through each color, when that gets added
                for smooth in smoothing:
                    spline, derivative, res_spline, rms_res_spline, maxima, minima = spline_fit(xdata[j], ydata[j][:, i], smooth)
                    
                    if all((minima, maxima)):
                        minima_1.append(minima[0])
                        minima_2.append(minima[1])
                        maxima_1.append(maxima[0])
                        maxima_2.append(maxima[1])
                    
                    splines_j.append(spline)
                    derivatives_j.append(derivative)
                    residuals_j.append(res_spline)

            extrema.append([minima_1, minima_2, maxima_1, maxima_2])
            splines_collected.append(splines_j)
            derivative_collected.append(derivatives_j)
            residuals_collected.append(residuals_j)

            print(j)
    
        
        colormap = plt.cm.cividis

        # PLotting the raw data and the corresponding spline fits and dereivatives
        fig_raw, ax_raw = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

        for i in range(len(xdata)):
            for j, smoothing_value in enumerate(smoothing_values):
                color = colormap(j / (len(smoothing_values)-1))
                ax_raw[0].plot(np.unique(xdata[i]), splines_collected[i][j], label='Spline', color=color)
                ax_raw[1].plot(np.unique(xdata[i]), derivative_collected[i][j], label='Derivative',color=color)

        ax_raw[0].scatter(xdata_complete, ydata_complete, label='Data', s=5, color='red')
        ax_raw[1].plot(xdata_complete, np.zeros(len(xdata_complete)), color='black')

        ax_raw[0].grid()
        ax_raw[1].grid()

        gradient_colorbar(colormap, ax_raw, 'vertical', smoothing_values)
        fig_raw.suptitle(f'Raw Data with Spline Fit {rotation[k]}', fontsize=16)



        # PLotting the residuals from the spline and raw data
        fig_res, ax_res = plt.subplots(1, 1, figsize=(15, 8))

        for i in range(len(xdata)):
            for j, smoothing_value in enumerate(smoothing_values):
                color = colormap(j / (len(smoothing_values)-1))
                ax_res.scatter(np.unique(xdata[i]), residuals_collected[i][j] ,s=5, label='Residuals',color=color)
        ax_res.grid()

        gradient_colorbar(colormap, ax_res, 'vertical', smoothing_values)
        fig_res.suptitle(f'Residuals of spline {rotation[k]}', fontsize=16)



        # Plotting the resulting extrema from the spline fit
        fig_extr, ax_extr = plt.subplots(2, 2, figsize=(15, 8))
        ax_extr = ax_extr.flatten()

        names = ["minima_1", "minima_2", "maxima_1", "maxima_2"]

        rms = []
        accuracy = []
        
        extrema_filtered = [item for item in extrema if any(item)]
        indexes = list(range(len(extrema_filtered)))

        for i in range(4):
            rms_for_extr = []
            accuracy_for_extr = []
            for j, smoothing_value in enumerate(smoothing_values):
                variable_data = [variables[i][j] for variables in extrema_filtered]
                color = colormap(j / (len(smoothing_values)-1))
                ax_extr[i].plot(indexes, variable_data, color=color)

                intercept = np.mean(variable_data)
                horizontal_line = np.full_like(variable_data, intercept)
                #ax_extr[i].plot(indexes, horizontal_line, color= color)

                res = variable_data - horizontal_line
                rms_per_smooth = np.sqrt(np.mean(res**2))
                accuracy_per_smooth = np.mean(np.abs(res))

                rms_for_extr.append(rms_per_smooth)
                accuracy_for_extr.append(accuracy_per_smooth)
                
            rms.append(rms_for_extr)
            accuracy.append(accuracy_for_extr)

            ax_extr[i].set_title(names[i])
            ax_extr[i].grid()

        gradient_colorbar(colormap, ax_extr, 'horizontal', smoothing_values)
        fig_extr.suptitle(f'Extrema of Splines {rotation[k]}', fontsize=16)
        

        # PLotting the Rms from the induvidual extrema and their resulting average line
        fig_rms, ax_rms = plt.subplots(2, 2, figsize=(15, 8))
        ax_rms = ax_rms.flatten()

        for i in range(4):
            for j, smoothing_value in enumerate(smoothing_values):
                color = colormap(j / (len(smoothing_values)-1))
                ax_rms[i].scatter(smoothing_value, rms[i][j], color=color)
                ax_rms[i].grid()
        gradient_colorbar(colormap, ax_rms, 'horizontal', smoothing_values)
        fig_rms.suptitle(f'Rms of Extrema {rotation[k]}', fontsize=16)

        '''
        fig4, ax4 = plt.subplots(2, 2, figsize=(15, 8))
        ax4 = ax4.flatten()

        for i in range(4):
            for j, smoothing_value in enumerate(smoothing_values):
                color = colormap(j / (len(smoothing_values)-1))
                ax4[i].scatter(smoothing_value, accuracy[i][j], color=color)
                ax4[i].grid()
        gradient_colorbar(colormap, ax4, 'horizontal', smoothing_values)
        fig4.suptitle(f'Accuracy of Extrema {rotation[k]}', fontsize=16)'''

    plt.show()

smoothing_values = np.linspace(0.0011, 0.0018, 4)
main(smoothing_values, "data\-9cw2.txt")

