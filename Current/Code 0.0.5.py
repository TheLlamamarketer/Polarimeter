import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, savgol_filter


# Better plotting of the minima and general impovements

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
        for i in range(ydata[j].shape[1]):
        
            smooth_Y, spline, spline_prime, res_spline, rms_spline, maxima, minima, peak, valley = spline_fit(xdata[j],  ydata[j][:, i], smoothing)

            if all((minima, maxima)):
                 extrema.append([*minima, *maxima])
            print(j)

            peaks.append(peak)
            valleys.append(valley)

            splines_collected.append(spline)
            smoothing_collected.append(smooth_Y)
            derivative_collected.append(spline_prime)
    
    indexes = list(range(len(extrema)))

    fig1, ax1 = plt.subplots(2, 2, figsize=(10, 8))
    ax1 = ax1.flatten()

    names = ["minima_1", "minima_2", "maxima_1", "maxima_2"]
    for i in range(4):
        variable_data = [variables[i] for variables in extrema]
        ax1[i].plot(indexes, variable_data)
        ax1[i].set_title(names[i])
        intercept = np.mean(variable_data)
        horizontal_line = np.full_like(variable_data, intercept)
        ax1[i].plot(indexes, horizontal_line)

        res = variable_data - horizontal_line
        rms = np.sqrt(np.mean(res**2))

        print(names[i], intercept, rms)


    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(xdata_complete, ydata_complete, label='Data', s=5, color='blue')
    ax2.plot(xdata_complete, np.zeros(len(xdata_complete)), color='black')


    for i in range(len(xdata)):
        #ax2.plot(np.unique(xdata[i]), smoothing_collected[i], label='Smoothed Data', color='black')
        ax2.plot(np.unique(xdata[i]), splines_collected[i], label='Spline', color='limegreen')
        ax2.plot(np.unique(xdata[i]), derivative_collected[i], label='Derivative', color='green')
        #ax2.scatter(xdata(peaks[i]), ydata(peaks[i]), label='Peaks', s=7, color='red')
        #ax2.scatter(xdata(valleys[i]), ydata(valleys[i]), label='Peaks', s=7, color='red')


    plt.tight_layout()
    plt.show()

plot(0.0013)

