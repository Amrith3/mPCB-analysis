"""Plotting: the figure style and the panels the figure scripts share.

The filter figures (scripts/ecg_filter_steps.py, scripts/ppg_vitals.py) build each row up in three steps:
step 1 the signal's spectrum, step 2 the filter's gain, step 3 the spectrum before and after the
filter. figure_style() is the style of the other figures: Arial 8 pt, thin axes, no box.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.signal import welch

from .filters import gain_db
from .recording import output

INK = "#0b0b0b"
INK_2 = "#52514e"
AXIS = "#898781"
HAIRLINE = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def figure_style():
    """Set the figure style for all following figures."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": HAIRLINE, "grid.linewidth": 0.5,
        "lines.linewidth": 1.2, "lines.markersize": 3, "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def letter(ax, text):
    """A bold panel letter (A, B, ...) at the top left of an axis."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(-40, 0), textcoords="offset points",
                fontsize=10, fontweight="bold", va="top")


def save(fig, name, folder=""):
    """Save the figure as PDF and PNG in outputs/, or in a folder inside it (e.g. "isoflurane")."""
    fig.savefig(output(folder, name + ".pdf"))
    fig.savefig(output(folder, name + ".png"))


def plain(value, position):
    """Tick label without trailing zeros (axis formatter)."""
    return f"{value:f}".rstrip("0").rstrip(".")


def frequency_axis(fs):
    """A frequency axis from 0.1 Hz to half the sampling rate, with plain tick labels."""
    plt.xlim(0.1, fs / 2)
    ticks = []
    for f in [0.1, 0.3, 1, 3, 10, 30, 100]:
        if f < 0.8 * fs / 2:
            ticks.append(f)
    labels = [str(f) for f in ticks]
    ticks.append(fs / 2)
    labels.append(str(round(fs / 2)))
    plt.xticks(ticks, labels)
    plt.minorticks_off()
    plt.xlabel("frequency (Hz)")


def step1_amplitude(x, fs, unit):
    """Step 1: the amplitude of each frequency in the signal (Welch spectrum, 8 s pieces), log axes."""
    f, power = welch(x, fs=fs, nperseg=min(len(x), int(8 * fs)), scaling="spectrum")
    plt.loglog(f[1:], np.sqrt(2 * power[1:]))
    frequency_axis(fs)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(plain))
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.title("1. FFT of the signal: amplitude (" + unit + ")")


def power_spectrum(x, fs):
    """The power spectrum of the signal in dB (Welch, 8 s pieces)."""
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(8 * fs)))
    plt.semilogx(f[1:], 10 * np.log10(p[1:]))
    frequency_axis(fs)
    plt.title("power spectrum of the signal (dB)")


def step2_gain(answer, fs):
    """Step 2: the filter's gain in dB at each frequency, from its impulse response (filters.gain_db)."""
    f, gain = gain_db(answer, fs)
    plt.semilogx(f[1:], gain[1:])
    frequency_axis(fs)
    plt.title("2. the filter: gain (dB)")


def step3_before_after(before, after, fs):
    """Step 3: the power spectrum before (grey) and after the filter, in dB."""
    piece = min(len(after), int(8 * fs))
    f, p_before = welch(before, fs=fs, nperseg=piece)
    f, p_after = welch(after, fs=fs, nperseg=piece)
    plt.semilogx(f[1:], 10 * np.log10(p_before[1:]), color="0.7", label="before")
    plt.semilogx(f[1:], 10 * np.log10(p_after[1:] + 1e-20), label="after")
    frequency_axis(fs)
    plt.title("3. power spectrum (dB)")
    plt.legend(fontsize=7)
