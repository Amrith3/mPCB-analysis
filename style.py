"""Figure style for the figures: Arial 8 pt, thin axes, no top/right box, PDF text kept as text."""
import os

import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK_2 = "#52514e"
AXIS = "#898781"
HAIRLINE = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def thesis():
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
    """The figure as PDF (for the thesis) and PNG, next to the script that made it."""
    folder = os.path.dirname(os.path.abspath(script))
    fig.savefig(os.path.join(folder, name + ".pdf"))
    fig.savefig(os.path.join(folder, name + ".png"))
