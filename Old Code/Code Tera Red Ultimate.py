import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import tkinter as tk
from tkinter import scrolledtext, IntVar
from matplotlib.widgets import RadioButtons
import mplcursors

# (-6956.00378071 * np.cos(((x + 243.07960733) / 9984.08589119) * 2 * math.pi) + 282.53479691 * np.cos(((x + 89.21990083) / 168.72274363) * 2 * math.pi) + 17.16150463 * np.cos(((x + 88.07843889) / 102.5248014) * 2 * math.pi) - 323.62271109 * np.cos(((x + 144.97397749) / 272.89562898) * 2 * math.pi))


def func(x, d, A, B, C):
    sum = d
    return sum + (C * calibrated(x) + A) * (np.cos((x + B) / 180.15 * math.pi)**2)

def calibrated(x):
    return (353.72040727 * np.cos(((x + 72.53476122) / 357.37452521) * 2 * math.pi) + 31.5977721 * np.cos(((x + 125.48042614) / 118.79852539) * 2 * math.pi) + 3.45634491* np.cos(((x + 151.59277747) / 63.91125385) * 2 * math.pi) )

def residuals(d, A, B, C, x, y):
    return y - func(x, d, A, B, C)


def fit_dataset(xdata, ydata):
    p0 = [ 128.71463977875302, 10630.504491596352, 213.80534764556128, 0.15237119123806733 ]

    def fixed_func(x, *args):
        d, A, B, C = args[0], args[1], args[2], args[3]
        return func(x, d, A, B, C)

    params, pcov = curve_fit(fixed_func, xdata, ydata, p0=p0, maxfev=10000000, bounds=([0, -np.inf, -np.inf, 0], [np.inf, np.inf, 360, np.inf]))
    d, A, B, C = params[0], params[1], params[2], params[3]
    
    param_errors = np.sqrt(np.diag(pcov))
    res = residuals(d, A, B, C, xdata, ydata)
    rms = np.sqrt(np.mean(res**2))

    return d, A, B, C, func(xdata, d, A, B, C), param_errors, res, rms



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


def plot_power_spectrum(xdata, ydata, res, colors, filename):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

    max_amplitude = np.max(np.sqrt(np.absolute(res)))  # Compute the maximum amplitude
    radio_button = RadioButtons(
        ax=plt.axes([0, 0.05, 0.06, 0.05]), labels=["Period", "Frequency"], active=1
    )
    lines = []

    for i, (color, ax) in enumerate(zip(colors, axes)):
        y = res[i]  # Select the column for the current color
        N = len(xdata)

        yf = np.fft.fft(y - np.mean(y))  # Compute the FFT
        amplitude = 2 * np.abs(yf) / N  # Compute the amplitude

        dt = np.mean(np.diff(xdata))  # Time step
        fs = 1 / dt  # Sampling frequency

        freq = np.fft.fftfreq(N, d=1 / fs)[: N // 2]  # Construct frequency axis

        per = np.where(freq != 0, 1 / freq, np.inf)

        (line,) = ax.plot(
            freq, amplitude[: N // 2], c=color, label=f"{color}"
        )  # Plot spectrum vs frequency
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Frequency")
        ax.legend([color])
        ax.grid(True)

        if len(freq) > 1:
            ax.set_xlim(0, np.max(freq[1:]))
        ax.set_ylim(0, max_amplitude)

        lines.append(line)

    def update_axis(label):
        d = {"Period": per, "Frequency": freq}

        for ax, line in zip(axes, lines):
            data = d[label]
            line.set_xdata(data)

            if label == "Period":
                ax.set_xlabel("Period")
                ax.set_xlim(0, np.max(data[1:]))
            elif label == "Frequency":
                ax.set_xlabel("Frequency")
                ax.set_xlim(0, np.max(data[1:]))

        plt.setp(ax.get_xticklabels(), visible=False)

        plt.draw()

    radio_button.on_clicked(update_axis)

    # Enable point selection
    # mplcursors.cursor(hover=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=True)


class PlotHandles:
    def __init__(self):
        self.handles = []  # List to store plot handles


def toggle_color(color, handles):
    plot_handles = handles.handles

    rgb_color = mcolors.to_rgb(color)

    for scatter, line, scatter_res in plot_handles:
        scatter_colors = scatter.get_facecolor()

        # Toggle visibility for scatter plot, line plot, and scatter residuals plot corresponding to the color
        for color in scatter_colors:
            if np.all(color[:3] == rgb_color[:3]):
                scatter_visible = scatter.get_visible()
                line_visible = line.get_visible()
                scatter_res_visible = scatter_res.get_visible()
                scatter.set_visible(not scatter_visible)
                line.set_visible(not line_visible)
                scatter_res.set_visible(not scatter_res_visible)
                if not scatter_visible:
                    line.set_data([], [])  # Remove the line by setting empty data
                else:
                    line.set_data(scatter.get_offsets().T)  # Restore the line

    plt.draw()  # Update the plot


def create_plots(ax1, ax2, xdata, ydata, fits, residuals, colors, filename):
    handles = PlotHandles()  # Create an instance of PlotHandles to store plot handles

    for i, color in enumerate(colors):
        scatter = ax1.scatter(xdata, ydata[:, i], c=color, s=10, label=f"{color}")
        (line,) = ax1.plot(
            xdata, fits[i], c=color
        )  # Store the line as a single element tuple
        scatter_res = ax2.scatter(
            xdata, residuals[i], c=color, s=7, label=f"{color} Residuals"
        )
        handles.handles.append(
            (scatter, line, scatter_res)
        )  # Add plot handles to the handles list

    ax1.set_ylabel("Light intensity in ADU's")
    ax1.set_title(f"Data and fits from {filename}")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel("Rotation in °")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals")
    ax2.grid(True)

    return handles  # Return the PlotHandles object


def plot_data(
    xdata, ydata, fits, error, residuals, rms, D, A, B, C, norm_rms, colors, filename
):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Create the plots and store the handles in 'handles' variable
    handles = create_plots(ax1, ax2, xdata, ydata, fits, residuals, colors, filename)

    # Adjust the layout and show the plot without blocking the code execution
    plt.tight_layout()
    plt.show(block=False)

    # Create a new window to display results
    text_window = tk.Toplevel()
    text_window.title(f"Results for {filename}")
    text_window.geometry("700x100")

    # Create a frame to hold the text area
    text_frame = tk.Frame(text_window)
    text_frame.pack(side=tk.LEFT)

    # Create a scrolled text area to display results
    text_area = scrolledtext.ScrolledText(text_frame, width=70, height=5)
    text_area.pack()

    # Insert the column headers into the text area
    text_area.insert(tk.END, "Color\tPhase\t  Amplitude\tPeriod    \tRMS\tNorm RMS\tError\n")

    # Create a frame to hold the check buttons
    checkbuttons_frame = tk.Frame(text_window)
    checkbuttons_frame.pack(side=tk.RIGHT)

    checkbuttons = []
    for color, d, a, b, c, rm, nrm, err in zip(
        colors, D, A, B, C, rms, norm_rms, error
    ):
        rm = np.atleast_1d(rm)  # Convert rm to 1-D array
        nrm = np.atleast_1d(nrm)  # Convert nrm to 1-D array
        err = np.atleast_1d(err)  # Convert err to 1-D array

        # Insert the result values into the text area
        text_area.insert(
            tk.END,
            f"{color}\t{phase_dilemma(b, a)[0]:.6f}\t{phase_dilemma(b, a)[1]:.2f}\t  {c}\t{rm[0]:.2f}\t{nrm[0]:.2f}\t{err[2]:.6f}\n",
        )

        var = tk.BooleanVar()
        # Create a check button for each color, linking it to the toggle_color function with the corresponding color and handles
        checkbutton = tk.Checkbutton(
            checkbuttons_frame,
            text=color,
            variable=var,
            onvalue=True,
            offvalue=False,
            command=lambda c=color, h=handles: toggle_color(c, h),
        )
        checkbutton.pack(anchor=tk.W)
        checkbuttons.append((checkbutton, var))

    def on_close():
        text_window.destroy()

    # Set the action to be performed when the window is closed
    text_window.protocol("WM_DELETE_WINDOW", on_close)

    # Enable point selection
    # mplcursors.cursor(hover=True)

    # Start the Tkinter event loop
    text_window.mainloop()


def phase_dilemma(phase, amplitude):
    if amplitude > 0:
        phase %= 360
        if phase > 180:
            phase -= 180

    if amplitude < 0:
        phase %= 360
        if phase > 180:
            phase = (phase + 90) % 360
        else:
            phase = (phase + 270) % 360

    return phase, abs(amplitude)


def main(filenames):
    for filename in filenames:
        dataset = load_data(filename)  # Load the dataset from the file
        colors = ["Red"]

        xdata, ydata = dataset[0]  # Access the first element of the dataset list

        A_list = []
        B_list = []
        C_list = []
        d_list = []
        og_residuals_list = []
        fits = []
        error_list = []
        residuals_list = []
        rms_list = []
        norm_rms_list = []

        for i in range(ydata.shape[1]):
            # Fit dataset to cosine with 180° period
            d3, A3, B3, C3, fit, error, res, rms = fit_dataset(xdata, ydata[:, i])

            A_list.append(A3)
            B_list.append(B3)
            C_list.append(C3)
            d_list.append(d3)
            fits.append(fit)

            error_list.append(error)
            residuals_list.append(res)
            rms_list.append(rms)

            og_residuals_list.append(res)

            # Calculate relative RMS
            amplitude = np.max(ydata[:, i]) - np.min(ydata[:, i])
            norm_rms = 1000 * rms / amplitude
            norm_rms_list.append(norm_rms)

        for i in range(len(colors)):
            np.set_printoptions(suppress=True)
            print(f"{colors[i]}:")
            print(f"y-Offset: {d_list[i]}")
            print(f"Amplitude: {A_list[i]}")
            print(f"Phase Offset: {B_list[i]}")
            print(f"Calibrated Amplitude: {C_list[i]}")
            print(f"RMS: {rms_list[i]}")
            print(f"normalized RMS: {norm_rms_list[i]}")
            print(f"Errors: {error_list[i]}")
            print()

        # Plot the data

        plot_data(
            xdata,
            ydata,
            fits,
            error_list,
            residuals_list,
            rms_list,
            d_list,
            A_list,
            B_list,
            C_list,
            norm_rms_list,
            colors,
            filename,
        )
        #plot_power_spectrum(xdata, ydata, residuals_list, colors, filename)


# Files being analyzed
# filenames = ["-3.txt", "-3c2.txt", "-3c3.txt", "-3c4.txt", "-3c5.txt",  "-3c6.txt", "-3c7.txt", "-3c8.txt", "-3c9.txt"]
#filenames = ["-4.txt", "-4c1.txt", "-4c2.txt", "-4c3.txt", "-4c4.txt", "-4c5.txt",  "-4c6.txt", "-4c7.txt", "-4c8.txt"]
filenames = ["-8c5.txt"]
main(filenames)


