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
from numpy.fft import fft, ifft, fftfreq


# Alternate between Clockwise and Counterclockwise rotation


def load_data(*filenames):

    dataset_split = []
    data_list = [np.loadtxt(filename, delimiter=",") for filename in filenames]
    data = np.concatenate(data_list)

    data = np.array(data)
    time = data[:, 0]
    index = data[:, 1]
    xdata = data[:, 2]
    ydata = data[:, 3:]
    ydata = ydata / np.max(ydata, axis=0, keepdims=True) # Normalizes the amplitude of the data to 0 - 1

    index_change = np.where(np.abs(np.diff(index)) > 2000)[0] + 1 # the spot at which the index changes drasticly(like from 4096 to 0)

    ydata_split = np.split(ydata, index_change)
    xdata_split = np.split(xdata, index_change)

    def get_slope(xdata):
        y = [float(value) for value in xdata]
        x = list(range(len(y)))

        # looking for the transition from 0 to 360°, or the other way
        change_point =  np.where(np.abs(np.diff(y)) > 100)[0] + 1
        change_points = [0, change_point[0], len(x)] if change_point.size > 0 else [0, len(x)]

        slopes = [stats.linregress(x[change_points[i]:change_points[i+1]], y[change_points[i]:change_points[i+1]]).slope
                for i in range(len(change_points) - 1)]
        
        return np.mean(slopes)


    C_xdata_split = []
    CC_xdata_split = []
    C_ydata_split = []
    CC_ydata_split = []

    for j in range(len(xdata_split)):
        if get_slope(xdata_split[j]) > 0:
            C_xdata_split.append(xdata_split[j])
            C_ydata_split.append(ydata_split[j])
        elif get_slope(xdata_split[j]) < 0:
            CC_xdata_split.append(xdata_split[j])
            CC_ydata_split.append(ydata_split[j])

    # Sort each subarray independently based on the corresponding xdata values
    def sort(xdata, ydata):
        for i in range(len(xdata)):
            sorted_indices = np.argsort(xdata[i])
            xdata[i] = xdata[i][sorted_indices]
            ydata[i] = ydata[i][sorted_indices]
        return xdata, ydata

    dataset_split.append((sort(C_xdata_split, C_ydata_split)))
    dataset_split.append((sort(CC_xdata_split, CC_ydata_split)))
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



def fourier_fit(xdata, ydata):
    # Fourier analysis implementation
    N = len(ydata)
    yf = fft(ydata)
    xf = fftfreq(N, np.mean(np.diff(xdata)))
    
    # You may need to adjust the number of harmonics based on your data
    num_harmonics = 7
    indices = np.argsort(np.abs(yf))[-num_harmonics:]

    # Zeroing out smaller Fourier coefficients
    yf_filtered = np.zeros_like(yf)
    yf_filtered[indices] = yf[indices]

    # Inverse Fourier transform to obtain the filtered signal
    return ifft(yf_filtered).real



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


    for k in range(len(dataset[0])):
        xdata, ydata = dataset[k]
        
        xdata_complete = [data for j in range(len(ydata)) for data in xdata[j]]
        ydata_complete = [data for j in range(len(ydata)) for i in range(ydata[j].shape[1]) for data in ydata[j][:, i]]

        extrema = []
        peaks = []
        valleys = []
        peaks_fourier = []
        valleys_fourier = []
        splines_collected = []
        smoothing_collected = []
        derivative_collected = []

        fourier_collected = []
        fourier_derivative_collected = []


        for j in range(len(ydata)):
            minima_1 = []
            minima_2 = []
            maxima_1 = []
            maxima_2 = []

            splines_j = []
            derivatives_j = []

            fourier_j = []
            fourier_derivative_j = []

            
            for i in range(ydata[j].shape[1]):

                fourier_y = fourier_fit(np.unique(xdata[j]), ydata[j][:, i])
                fourier_j.append(fourier_y)

                distances = np.diff(np.unique(xdata[j]))
                if len(distances) != len(fourier_y) - 1:
                    distances = 0.1 

                # Calculate the gradient
                derivative_fourier = np.gradient(fourier_y, distances, edge_order=2)
                fourier_derivative_j.append(derivative_fourier)


                for smooth in smoothing:
                    smooth_Y, spline, derivative, res_spline, rms_spline, maxima, minima, peak, valley = spline_fit(xdata[j], ydata[j][:, i], smooth)
                    
                    if all((minima, maxima)):
                        minima_1.append(minima[0])
                        minima_2.append(minima[1])
                        maxima_1.append(maxima[0])
                        maxima_2.append(maxima[1])
                    
                    splines_j.append(spline)
                    derivatives_j.append(derivative)

                # Fourier fitting and peak detection
                peak_fourier, _ = find_peaks(fourier_y)
                valley_fourier, _ = find_peaks(-fourier_y)

                peaks_fourier.append(peak_fourier)
                valleys_fourier.append(valley_fourier)


            extrema.append([minima_1, minima_2, maxima_1, maxima_2])
            splines_collected.append(splines_j)
            derivative_collected.append(derivatives_j)

            fourier_collected.append(fourier_j)
            fourier_derivative_collected.append(fourier_derivative_j)
            print(j)
    
        
        colormap = plt.cm.cividis


        fig1, ax1 = plt.subplots(2, 1, figsize=(15, 8))
        ax1[0].scatter(xdata_complete, ydata_complete, label='Data', s=5, color='blue')
        ax1[1].plot(xdata_complete, np.zeros(len(xdata_complete)), color='black')

        for i in range(len(xdata)):
            for j, smoothing_value in enumerate(smoothing_values):
                color = colormap(j / (len(smoothing_values)-1))
                ax1[0].plot(np.unique(xdata[i]), splines_collected[i][j], label='Spline', color=color)
                ax1[1].plot(np.unique(xdata[i]), derivative_collected[i][j], label='Derivative',color=color)
        ax1[0].grid()
        ax1[1].grid()

        gradient_colorbar(colormap, ax1, 'vertical', smoothing_values)
        fig1.suptitle(f'Raw Data with Spline Fit {rotation[k]}', fontsize=16)


        fig2, ax2 = plt.subplots(2, 2, figsize=(15, 8))
        ax2 = ax2.flatten()

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
                ax2[i].plot(indexes, variable_data, color=color)

                intercept = np.mean(variable_data)
                horizontal_line = np.full_like(variable_data, intercept)
                #ax2[i].plot(indexes, horizontal_line, color= color)

                res = variable_data - horizontal_line
                rms_per_smooth = np.sqrt(np.mean(res**2))
                accuracy_per_smooth = np.mean(np.abs(res))

                rms_for_extr.append(rms_per_smooth)
                accuracy_for_extr.append(accuracy_per_smooth)
                
            rms.append(rms_for_extr)
            accuracy.append(accuracy_for_extr)

            ax2[i].set_title(names[i])
            ax2[i].grid()

        gradient_colorbar(colormap, ax2, 'horizontal', smoothing_values)
        fig2.suptitle(f'Extrema of Splines {rotation[k]}', fontsize=16)
        

        fig3, ax3 = plt.subplots(2, 2, figsize=(15, 8))
        ax3 = ax3.flatten()

        for i in range(4):
            for j, smoothing_value in enumerate(smoothing_values):
                color = colormap(j / (len(smoothing_values)-1))
                ax3[i].scatter(smoothing_value, rms[i][j], color=color)
                ax3[i].grid()
        gradient_colorbar(colormap, ax3, 'horizontal', smoothing_values)
        fig3.suptitle(f'Rms of Extrema {rotation[k]}', fontsize=16)


        fig4, ax4 = plt.subplots(2, 2, figsize=(15, 8))
        ax4 = ax4.flatten()

        for i in range(4):
            for j, smoothing_value in enumerate(smoothing_values):
                color = colormap(j / (len(smoothing_values)-1))
                ax4[i].scatter(smoothing_value, accuracy[i][j], color=color)
                ax4[i].grid()
        gradient_colorbar(colormap, ax4, 'horizontal', smoothing_values)
        fig4.suptitle(f'Accuracy of Extrema {rotation[k]}', fontsize=16)



        fig_fourier, ax_fourier = plt.subplots(2, 1, figsize=(15, 8))
        ax_fourier[0].scatter(xdata_complete, ydata_complete, label='Data', s=5, color='blue')
        ax_fourier[1].plot(xdata_complete, np.zeros(len(xdata_complete)), color='black')

        for i in range(len(xdata)):
            for j, fourier in enumerate(fourier_collected[i]):
                #color = colormap(j / (len(fourier_collected[i])-1))
                ax_fourier[0].plot(xdata[i], fourier, label='Fourier Fit' )

                # Assuming fourier_derivative_collected is structured similarly to fourier_collected
                derivative_fit = fourier_derivative_collected[i][j]
                ax_fourier[1].plot(xdata[i], derivative_fit, label='Fourier Derivative')

        ax_fourier[0].set_title('Fourier Transform')
        ax_fourier[1].set_title('Derivative of Fourier Transform')
        ax_fourier[0].grid()
        ax_fourier[1].grid()
        fig_fourier.suptitle(f'Fourier Analysis {rotation[k]}', fontsize=16)

    plt.show()

smoothing_values = np.linspace(0.00118, 0.00128, 40)
main(smoothing_values, "data\-9c5.txt")

