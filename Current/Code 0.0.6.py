import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, savgol_filter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


# Adding gradient to the full code


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

def gradient_colorbar(colormap, ax, orientantion):

    num_smoothing_values = len(smoothing_values)
    max_ticks = 10
    num_ticks = min(max_ticks, num_smoothing_values)

    # Create a colorbar for smoothing values
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colormap(np.linspace(0, 1, num_smoothing_values)), N=num_smoothing_values)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_smoothing_values - 1)),
                        ax=ax, orientation=orientantion, fraction=0.05, pad=0.07, aspect=40, ticks=np.linspace(0, num_smoothing_values - 1, num_ticks))
    cbar.set_label('Smoothing')

    cbar.set_ticklabels([f'{value:.5f}' for value in smoothing_values])


def main(smoothing):
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

        splines_j = []
        derivatives_j = []
        
        for i in range(ydata[j].shape[1]):
            for smooth in smoothing:
                smooth_Y, spline, derivative, res_spline, rms_spline, maxima, minima, peak, valley = spline_fit(xdata[j], ydata[j][:, i], smooth)
                
                if all((minima, maxima)):
                    minima_1.append(minima[0])
                    minima_2.append(minima[1])
                    maxima_1.append(maxima[0])
                    maxima_2.append(maxima[1])
                
                splines_j.append(spline)
                derivatives_j.append(derivative)

        extrema.append([minima_1, minima_2, maxima_1, maxima_2])
        splines_collected.append(splines_j)
        derivative_collected.append(derivatives_j)
        print(j)
 
    
    colormap = plt.cm.cividis


    fig1, (ax2, ax3) = plt.subplots(2, 1, figsize=(15, 8))
    ax2.scatter(xdata_complete, ydata_complete, label='Data', s=5, color='blue')
    ax3.plot(xdata_complete, np.zeros(len(xdata_complete)), color='black')

    for i in range(len(xdata)):
         for j, smoothing_value in enumerate(smoothing_values):
            color = colormap(j / (len(smoothing_values)-1))
            ax2.plot(np.unique(xdata[i]), splines_collected[i][j], label='Spline', color=color)
            ax3.plot(np.unique(xdata[i]), derivative_collected[i][j], label='Derivative',color=color)

    gradient_colorbar(colormap, (ax2, ax3), 'vertical')


    fig2, ax1 = plt.subplots(2, 2, figsize=(15, 8))
    ax1 = ax1.flatten()

    names = ["minima_1", "minima_2", "maxima_1", "maxima_2"]
    rms = []
    
    extrema_filtered = [item for item in extrema if any(item)]
    indexes = list(range(len(extrema_filtered)))

    for i in range(4):
        rms_for_extr = []
        for j, smoothing_value in enumerate(smoothing_values):
            variable_data = [variables[i][j] for variables in extrema_filtered]
            color = colormap(j / (len(smoothing_values)-1))
            ax1[i].plot(indexes, variable_data, color=color)

            intercept = np.mean(variable_data)
            horizontal_line = np.full_like(variable_data, intercept)
            ax1[i].plot(indexes, horizontal_line, color= color)

            res = variable_data - horizontal_line
            rms_per_smooth = np.sqrt(np.mean(res**2))
            rms_for_extr.append(rms_per_smooth)
        rms.append(rms_for_extr)
        ax1[i].set_title(names[i])

    gradient_colorbar(colormap, ax1, 'horizontal')

    fig3, ax1 = plt.subplots(2, 2, figsize=(15, 8))
    ax1 = ax1.flatten()


    for i in range(4):
        for j, smoothing_value in enumerate(smoothing_values):
            color = colormap(j / (len(smoothing_values)-1))
            ax1[i].scatter(smoothing_value, rms[i][j], color=color)


    plt.show()

smoothing_values = np.linspace(0.00124, 0.00128, 100)
main(smoothing_values)

