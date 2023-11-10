import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline

# Calibrated data from the first file is analysed with a spline and that spline is then multiplied by a factor in a sinus fit of a modualtion curve
# Factor * calibrated spline * sin


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

    sorted_indices = np.argsort(xdata)
    xdata = xdata[sorted_indices]
    ydata = ydata[sorted_indices]

    dataset.append((xdata, ydata))  # Convert ydata to a NumPy array
    return dataset


def calibrated_spline():
    dataset = load_data("-6w8.txt")
    xdata, ydata = dataset[0]
    ydata_n = ydata / np.max(ydata)

    smoothed_data = gaussian_filter(ydata_n, 50)
    spline = UnivariateSpline(xdata, smoothed_data, k=5, s=0.001)
    fit = spline(xdata)
    res = np.zeros_like(ydata_n)
    for i in range(len(ydata_n)):
        res[i] = ydata_n[i] - fit[i]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Plot the data
    ax1.scatter(xdata, ydata_n, label='ydata', s=5, color='blue')    
    ax1.plot(xdata, smoothed_data, 'k-', label='Smoothed Data')  
    ax1.plot(xdata, fit, 'g-', label='Spline fit')  
    ax1.legend()

    ax2.scatter(xdata, res, label='Residuals', s=5)

    plt.show()
    return spline

spline = calibrated_spline()

def func(x, d, A, B, C):
    return d + (C * spline(x) + A) * (np.cos((x + B) / 180 * math.pi)**2)


def residuals(d, A, B, C, x, y):
    return y - func(x, d, A, B, C)


def fit_dataset(xdata, ydata):
    p0 = [ 125.52744690313588, 10633.643364493164, 213.48643322729848, 6.504151648200757 ]

    def fixed_func(x, *args):
        d, A, B, C = args
        return func(x, d, A, B, C)

    params, pcov = curve_fit(fixed_func, xdata, ydata, p0=p0, maxfev=100000, bounds=([0, -np.inf, 0, 0], [np.inf, np.inf, 360, np.inf]))
    d, A, B, C = params[0], params[1], params[2], params[3]
    
    param_errors = np.sqrt(np.diag(pcov))
    res = residuals(d, A, B, C, xdata, ydata)
    rms = np.sqrt(np.mean(res**2))

    return d, A, B, C, func(xdata, d, A, B, C), param_errors, res, rms



def main():
    dataset = load_data("-8c1.txt")
    xdata, ydata = dataset[0]

    for i in range(ydata.shape[1]):

        d, a, b, c, fit, error, res, rms = fit_dataset(xdata, ydata[:, i])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

        np.set_printoptions(suppress=True)
        print(f"y-Offset: {d}")
        print(f"Amplitudes: {a}")
        print(f"Phases: {b}")
        print(f"Calibrated Amplitude: {c}")
        print(f"RMS: {rms}")
        print(f"Errors: {error}")
        print()

        # Plot the data
        ax1.scatter(xdata, ydata[:, i], label='ydata', s=5, color='red')     
        ax1.plot(xdata, fit, 'k-', label='fit')  
        ax1.legend()

        ax2.scatter(xdata, res, label='Residuals', s=5)

        plt.show()

main()
