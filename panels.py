"""Shared panels: signal spectrum (step 1), filter gain (step 2), before/after (step 3), averaged beat;
and the figure style of hr_agreement.py and isoflurane/: Arial 8 pt, thin axes, no top/right box."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.signal import welch

import filters

INK = "#0b0b0b"
INK_2 = "#52514e"
AXIS = "#898781"
HAIRLINE = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def figure_style():
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
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(-40, 0), textcoords="offset points",
                fontsize=10, fontweight="bold", va="top")


def save(fig, script, name):
    """The figure as PDF and PNG, next to the script that made it."""
    folder = os.path.dirname(os.path.abspath(script))
    fig.savefig(os.path.join(folder, name + ".pdf"))
    fig.savefig(os.path.join(folder, name + ".png"))


def plain(value, position):
    return f"{value:f}".rstrip("0").rstrip(".")


def frequency_axis(fs):
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
    f, power = welch(x, fs=fs, nperseg=min(len(x), int(8 * fs)), scaling="spectrum")
    plt.loglog(f[1:], np.sqrt(2 * power[1:]))
    frequency_axis(fs)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(plain))
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.title("1. the signal: amplitude (" + unit + ")")


def step2_gain(answer, fs):
    f, gain = filters.gain_db(answer, fs)
    plt.semilogx(f[1:], gain[1:])
    frequency_axis(fs)
    plt.title("2. the filter: gain (dB)")


def step3_before_after(before, after, fs):
    piece = min(len(after), int(8 * fs))
    f, p_before = welch(before, fs=fs, nperseg=piece)
    f, p_after = welch(after, fs=fs, nperseg=piece)
    plt.semilogx(f[1:], 10 * np.log10(p_before[1:]), color="0.7", label="before")
    plt.semilogx(f[1:], 10 * np.log10(p_after[1:] + 1e-20), label="after")
    frequency_axis(fs)
    plt.title("3. power spectrum (dB)")
    plt.legend(fontsize=7)


def averaged(x, centres, half):
    """Median of x around each centre, or None if fewer than 3 fit."""
    pieces = []
    for k in centres:
        if k - half >= 0 and k + half <= len(x):
            pieces.append(x[k - half:k + half])
    if len(pieces) < 3:
        return None
    return np.median(pieces, axis=0)
