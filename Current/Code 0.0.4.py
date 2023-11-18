import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

# Different methods of fitting the modulation curve are used

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


def spline_fit(xdata, ydata):    
    smoothed_data = gaussian_filter(ydata, 3)
    spline = UnivariateSpline(xdata, smoothed_data, k=5, s=0.05)
    fit = spline(xdata)
    res = np.zeros_like(ydata)
    for i in range(len(ydata)):
        res[i] = ydata[i] - fit[i]

    rms = np.sqrt(np.mean(res**2))

    
    spline_prime = spline.derivative()
    x_range = np.linspace(xdata.min(), xdata.max(), 500000)  # Adjust the number of points
    derivative_values = spline_prime(x_range)

    maxima = []
    minima = []
    for i in range(1, len(derivative_values) - 1):
        if derivative_values[i - 1] > 0 and derivative_values[i] < 0:
            maxima.append(x_range[i])
        elif derivative_values[i - 1] < 0 and derivative_values[i] > 0:
            minima.append(x_range[i])

    return smoothed_data, fit, spline_prime(xdata), res, rms, maxima, minima

def plot():
    dataset = load_data("data\-9c2.txt")
    xdata, ydata = dataset[0]
    
    xdata_complete = [data for j in range(len(ydata)) for data in xdata[j]]
    ydata_complete = [data for j in range(len(ydata)) for i in range(ydata[j].shape[1]) for data in ydata[j][:, i]]

    extrema = []
    splines_collected = []
    smoothing_collected = []

    for j in range(len(ydata)):
        for i in range(ydata[j].shape[1]):
        
            #d, a, b, c, cos_fit, error, res_cos, rms_cos = sinus_fit(xdata[j],  ydata[j][:, i])
            smooth_Y, spline, spline_prime, res_spline, rms_spline, maxima, minima = spline_fit(xdata[j],  ydata[j][:, i])

            if all((minima, maxima)):
                 extrema.append([*minima, *maxima])
            print(j)

            splines_collected.append(spline)
            smoothing_collected.append(smooth_Y)

            '''
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

            color = ['blue', 'black', 'limegreen', 'magenta', 'green']
    
            ax1.scatter(xdata,  ydata[:, i], label='ydata', s=5, color=color[0]) 
            ax1.plot(xdata, smooth_Y, color[1], label='Smoothed Data')
            ax1.plot(xdata, xdata*0, color[1]) 

            ax1.plot(xdata, spline, color[2], label='Spline fit')
            ax1.plot(xdata, spline_prime, color[4], label='Derivative of Spline fit')
            ax1.plot(xdata, cos_fit, color[3], label='Cosine² Fit')  
            ax1.legend()

            ax2.scatter(xdata, res_spline, label='Residuals', s=5, color=color[2])

            plt.show()'''
    
    indexes = list(range(len(extrema)))

    fig1, ax1 = plt.subplots(2, 2, figsize=(10, 8))
    ax1 = ax1.flatten()

    names = ["minima_1", "minima_2", "maxima_1", "maxima_2"]
    for i in range(4):
        variable_data = [variables[i] for variables in extrema]
        ax1[i].plot(indexes, variable_data)
        ax1[i].set_title(names[i])


    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(xdata_complete, ydata_complete, label='Data', s=5, color='blue')
    for i in range(len(xdata)):
        ax2.plot(xdata[i], smoothing_collected[i], label='Smoothed Data', color='black')
        ax2.plot(xdata[i], splines_collected[i], label='Spline', color='limegreen')


    plt.tight_layout()
    plt.show()

plot()

