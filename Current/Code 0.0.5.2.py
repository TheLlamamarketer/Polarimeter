import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, savgol_filter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


# Testing out something to see if it works


def load_data(filename):
    dataset_split = []
    data = np.loadtxt(filename, delimiter=",")

    data = np.array(data)
    time = data[:, 0]
    index = data[:, 1]
    xdata = data[:, 2]
    ydata = data[:, 3:]
    ydata = ydata / np.max(ydata, axis=0, keepdims=True) # Normalizes the amplitude of the data to 0 - 1

    index_change = np.where(np.abs(np.diff(index)) > 2000)[0] + 1 # the spot at which the index changes drastically(like from 4096 to 0)

    ydata_split = np.split(ydata, index_change)
    xdata_split = np.split(xdata, index_change)


    # Sort each subarray independently based on the corresponding xdata values
    for i in range(len(xdata_split)):
        sorted_indices = np.argsort(xdata_split[i])
        xdata_split[i] = xdata_split[i][sorted_indices]
        ydata_split[i] = ydata_split[i][sorted_indices]


    dataset_split.append((xdata_split, ydata_split))
    return dataset_split


def spline_fit(xdata, ydata, smoothing):    

    xdata, index = np.unique(xdata, return_index=True)
    ydata = ydata[index]

    smoothed_data = savgol_filter(ydata, window_length=10, polyorder=3)
    spline = UnivariateSpline(xdata, ydata, k=4, s=smoothing)
    fit = spline(xdata)
    res = np.zeros_like(xdata)
    for i in range(len(ydata)):
        res[i] = xdata[i] - fit[i]

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


    return smoothed_data, fit, spline_prime(xdata), res, rms, maxima, minima, peaks, valleys

def plot(smoothing):
    dataset = load_data("data\-9c2.txt")
    xdata, ydata = dataset[0]
    
    xdata_complete = [data for j in range(len(ydata)) for data in xdata[j]]
    ydata_complete = [data for j in range(len(ydata)) for i in range(ydata[j].shape[1]) for data in ydata[j][:, i]]

    extrema = []
    peaks = []
    valleys = []
    splines_collected = []
    smoothing_collected = []
    derivative_collected = []



    for j in range(len(ydata)):
        minima_1 = []
        minima_2 = []
        maxima_1 = []
        maxima_2 = []
        
        for i in range(ydata[j].shape[1]):
            for smooth in smoothing:
                smooth_Y, spline, spline_prime, res_spline, rms_spline, maxima, minima, peak, valley = spline_fit(xdata[j], ydata[j][:, i], smooth)
                
                if all((minima, maxima)):
                    minima_1.append(minima[0])
                    minima_2.append(minima[1])
                    maxima_1.append(maxima[0])
                    maxima_2.append(maxima[1])

        extrema.append([minima_1, minima_2, maxima_1, maxima_2])
        print(j)
 
    
    fig1, ax1 = plt.subplots(2, 2, figsize=(15, 8))
    ax1 = ax1.flatten()

    names = ["minima_1", "minima_2", "maxima_1", "maxima_2"]
    
    extrema_filtered = [item for item in extrema if any(item)]
    indexes = list(range(len(extrema_filtered)))

    colormap = plt.cm.magma

    for i in range(4):
        for j, smoothing_value in enumerate(smoothing_values):
            variable_data = [variables[i][j] for variables in extrema_filtered]
            color = colormap(j / (len(smoothing_values)-1))
            ax1[i].plot(indexes, variable_data, color=color)
        ax1[i].set_title(names[i])


    num_smoothing_values = len(smoothing_values)
    max_ticks = 10
    num_ticks = min(max_ticks, num_smoothing_values)

    # Create a colorbar for smoothing values
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colormap(np.linspace(0, 1, num_smoothing_values)), N=num_smoothing_values)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_smoothing_values - 1)),
                        ax=ax1, orientation='horizontal', fraction=0.05, pad=0.07, aspect=40, ticks=np.linspace(0, num_smoothing_values - 1, num_ticks))
    cbar.set_label('Smoothing')

    cbar.set_ticklabels([f'{value:.5f}' for value in smoothing_values])

    plt.show()

smoothing_values = np.linspace(0.00118, 0.00135, 30)
plot(smoothing_values)

