import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

# Different methods of fitting the modulation curve are used

def load_data(filename):
    with open(filename, "r") as data_file:
        lines = data_file.readlines()

    dataset = []
    data = []

    for line in lines:
        line = line.strip()
        if "," in line:
            row = list(map(float, line.split(",")))
            data.append(row)

    data = np.array(data)
    xdata = data[:, 1]
    ydata = data[:, 2:]
    ydata = ydata / np.max(ydata)

    sorted_indices = np.argsort(xdata)
    xdata = xdata[sorted_indices]
    ydata = ydata[sorted_indices]

    dataset.append((xdata, ydata))  # Convert ydata to a NumPy array
    return dataset


def spline_fit(xdata, ydata):    
    smoothed_data = gaussian_filter(ydata, 2)
    spline = UnivariateSpline(xdata, smoothed_data, k=5, s=0.05)
    fit = spline(xdata)
    res = np.zeros_like(ydata)
    for i in range(len(ydata)):
        res[i] = ydata[i] - fit[i]

    rms = np.sqrt(np.mean(res**2))

    
    spline_prime = spline.derivative()
    x_range = np.linspace(xdata.min(), xdata.max(), 10000000)  # Adjust the number of points
    derivative_values = spline_prime(x_range)

    maxima = []
    minima = []
    for i in range(1, len(derivative_values) - 1):
        if derivative_values[i - 1] > 0 and derivative_values[i] < 0:
            maxima.append(x_range[i])
        elif derivative_values[i - 1] < 0 and derivative_values[i] > 0:
            minima.append(x_range[i])

    return smoothed_data, fit, spline_prime(xdata), res, rms, maxima, minima

def func(x, d, A, B, C):
    return d +  A * (np.cos((x + B) / 180 * math.pi)**2)


def residuals(d, A, B, C, x, y):
    return y - func(x, d, A, B, 180)


def sinus_fit(xdata, ydata):
    p0 = [ 10000, 10633.643364493164, 213.48643322729848, 180 ]

    def fixed_func(x, *args):
        d, A, B, C = args
        return func(x, d, A, B, C)

    params, pcov = curve_fit(fixed_func, xdata, ydata, p0=p0, maxfev=100000, bounds=([0, 0, 0, 0], [np.inf, np.inf, 360, 360]))
    d, A, B, C = params[0], params[1], params[2], params[3]
    
    param_errors = np.sqrt(np.diag(pcov))
    res = residuals(d, A, B, C, xdata, ydata)
    rms = np.sqrt(np.mean(res**2))

    return d, A, B, C, func(xdata, d, A, B, C), param_errors, res, rms

def fft_analysis(xdata, ydata):
    # Perform FFT analysis
    n = len(xdata)
    fft_result = np.fft.fft(ydata)
    frequencies = np.fft.fftfreq(n, xdata[1] - xdata[0])

    # Calculate the amplitude spectrum (ignore the DC component) and normalize
    amplitude_spectrum = 2.0 / n * np.abs(fft_result[:n // 2])
    frequencies = frequencies[:n // 2]

    return frequencies, amplitude_spectrum

def inverse_fft(frequencies, amplitude_spectrum, n):
    full_amplitude_spectrum = np.zeros(n, dtype=complex)
    full_amplitude_spectrum[:len(frequencies)] = amplitude_spectrum
    full_amplitude_spectrum[-len(frequencies):] = np.conj(amplitude_spectrum)
    result = np.fft.ifft(full_amplitude_spectrum, n=n).real  

    return result


def plot():
    dataset = load_data("-8c2.txt")

    xdata, ydata = dataset[0]
    for i in range(ydata.shape[1]):

        d, a, b, c, cos_fit, error, res_cos, rms_cos = sinus_fit(xdata,  ydata[:, i])
        smooth_Y, spline, spline_prime, res_spline, rms_spline, maxima, minima = spline_fit(xdata,  ydata[:, i])

        frequencies, amplitude_spectrum = fft_analysis(xdata,  ydata[:, i])
        reconstructed_signal = inverse_fft(frequencies, amplitude_spectrum, len(xdata))

    
        np.set_printoptions(suppress=True)
        print(f"Spline Fit: MINIMA: {minima} , MAXIMA: {maxima}")
        print(f"Cosine Fit: y-Offset: {d:.5f} , Amp: {a:.5f}, Phase: {b:.5f}, Per: {c:.5f}")
        print(f"RMS Spline: {rms_spline:.5f}, RMS Cosine: {rms_cos:.5f}")
        print(f"Errors: {error}")
        print()


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

        color = ['blue', 'black', 'limegreen', 'magenta', 'green']
 
        ax1.scatter(xdata,  ydata[:, i], label='ydata', s=5, color=color[0]) 
        ax1.plot(xdata, smooth_Y, color[1], label='Smoothed Data')
        ax1.plot(xdata, xdata*0, color[1]) 

        ax1.plot(xdata, spline, color[2], label='Spline fit')
        ax1.plot(xdata, spline_prime, color[4], label='Derivative of Spline fit')
        ax1.plot(xdata, cos_fit, color[3], label='Cosine² Fit')  
        ax1.legend()

        #ax2.scatter(xdata, res_cos, label='Residuals', s=5, color=color[3])
        ax2.scatter(xdata, res_spline, label='Residuals', s=5, color=color[2])

        #fig, ax3 = plt.subplots(figsize=(10, 4))
        #ax3.plot(xdata, reconstructed_signal, label='Reconstructed Signal', color='red')
        #ax3.set_xlabel('Frequency')
        #ax3.set_ylabel('Amplitude')
        #ax3.legend()

        plt.show()

plot()

