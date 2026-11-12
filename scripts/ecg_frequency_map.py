"""Where the ECG's frequencies lie against what a recorder keeps: human ECG (published), mouse ECG
(published) and mouse ECG (ours), drawn two ways with the same numbering.

ecg_frequency_ranges.png: one frequency axis and one row per item, with a bar from the item's low to
its high frequency for the human (grey), the published mouse (blue) and our mouse (red) values. Our
board's band, from the ECG chip's 0.5 Hz high-pass to its 153.6 Hz low-pass, is shaded: a waveform
band inside it is recorded. Rhythms (heart rate, HRV) are measured from the R-wave times, so they
only need the R wave recorded; their bands are for comparison. All our values come from the ECG.

ecg_frequency_map.png: the same bands as points, as on the supervisor's whiteboard. Across: the low
cut (the high-pass that still keeps the band). Up: the high cut (the low-pass that still keeps it). A
single frequency lies on the diagonal. A recorder is the box from (low, low) to (high, high) and keeps
every band inside it. Filled numbers: ECG waveform; open numbers: rhythms. A published and our mark
too close to tell apart are drawn side by side.

Our bands: every stretch at one dose in each run of stats.RUNS, its last 2 min
(sessions.dose_averages). The waves are cut from the averaged beat of the stretch, read every 1/8192 s,
at its boundaries (sessions.ecg_points): P on to P off, and QRS on to S for the R wave. The straight
line through the ends is taken off, then FFT; the band holds the middle 90 % of the wave's energy, counted from the
heart rate up: a beat repeats at the heart rate, so it holds nothing slower (Buendía-Fuentes et al.
2012). Our ECG's power: the same FFT of the whole averaged beat, and the band from 0 Hz holding 98 %
of its energy, as Ai et al. (1996) give it for the mouse ECG. The median over the stretches is drawn,
with the range over them as a thin line.
Heart rate: from the ECG's R-R intervals every 10 s in every session, 1st to 99th percentile.
Our HRV: low frequency as published. High frequency from 1.5 Hz up to half the heart rate, because
the R-R intervals come one per beat and so hold no rhythm faster than half the heart rate: 2.3 Hz at
our slowest heart, 4.4 Hz at our fastest.
Mains: the lines at 50 Hz and its multiples that stand at least 3 x above the spectrum next to them
(Welch, 4 s pieces); the notches take them out.

Published: Kligfield et al. 2007 (Circulation 115:1306): diagnostic ECG, clinical monitor, slowest
human heart rate. Tereshchenko & Josephson 2015 (J Electrocardiol 48:933): human P wave, QRS and
high-frequency QRS. Buendía-Fuentes et al. 2012 (ISRN Cardiol 706217): no ECG content below the
heart rate. Bhadouria et al. 2025 (Signal Image Video Process 19:231): fastest human heart rate,
mouse heart rate. Shaffer & Ginsberg 2017 (Front Public Health 5:258): human HRV. Tomsits et al. 2023 (Lab
Anim 52:85): mouse HRV. Sedaghat et al. 2017 (J Electrocardiol 50:323): implanted telemetry,
high-frequency QRS. Ai et al. 1996 (Sheng Li Xue Bao 48:512): the mouse ECG's power.
"""
import contextlib
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import CubicSpline
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the main folder
from mpcb import filters, plotting, stats  # noqa: E402
from mpcb.plotting import BLUE, INK, INK_2  # noqa: E402
from mpcb.sessions import SESSIONS, Session, ecg_points  # noqa: E402

BOARD_HZ = (0.5, 153.6)             # the ECG chip's high-pass, and its low-pass as set since 24 Sep 2026
RED = "#c8322b"
GREY = "#8c8c8c"
SHADE = "#f8e6e5"                   # our board's band
SERIES = [("human, published", GREY), ("mouse, published", BLUE), ("mouse, ours", RED)]

# recorders: (name, human, mouse published, mouse ours), each (low, high, what) or None
RECORDERS = [
    ("ECG recorder", (0.05, 150, "diagnostic ECG"), (1, 200, "implanted telemetry"), (*BOARD_HZ, "our board")),
    ("Clinical monitor", (0.5, 40, "clinical monitor"), None, None),
]
# the numbered items: (number, name, waveform or rhythm, human, mouse published); ours below, from the data
ITEMS = [
    (1, "Whole ECG, 98 % of power", "waveform", None, (0, 300)),
    (2, "High-frequency QRS", "waveform", (150, 250), (200, 500)),
    (3, "R wave (mouse), QRS (human)", "waveform", (8, 50), None),
    (4, "P wave", "waveform", (5, 30), None),
    (5, "Heart rate", "rhythm", (0.67, 3.3), (4, 11)),
    (6, "HRV, high frequency", "rhythm", (0.15, 0.4), (1.5, 5)),
    (7, "HRV, low frequency", "rhythm", (0.04, 0.15), (0.15, 1.5)),
]


def wave_spectrum(t_ms, beat, start, end):
    """Frequencies (Hz) and energy spectrum of the averaged beat from start to end (ms).

    How: the beat is read every 1/8192 s (cubic spline), as sessions.ecg_points reads it to find the
    boundaries, so the piece starts and ends exactly there; at 2 ms between samples a mouse R wave
    holds only 3-6 samples, too few to cut it precisely. The straight line through the first and
    the last point is taken off, so the wave starts and ends at zero, and zeros are added to 2^17
    points before the FFT (0.06 Hz steps).
    """
    t = np.arange(t_ms[0], t_ms[-1], 1000 / 8192)
    wave = CubicSpline(t_ms, beat)(t[(t >= start) & (t <= end)])
    wave = wave - np.linspace(wave[0], wave[-1], len(wave))
    return np.fft.rfftfreq(2 ** 17, 1 / 8192), np.abs(np.fft.rfft(wave, 2 ** 17)) ** 2


def band(f, energy, heart_hz, share=(0.05, 0.95)):
    """The band (Hz) holding a wave's energy from the share[0] to the share[1] part (the middle 90 %
    unless told otherwise), counted from the heart rate up."""
    total = np.cumsum(np.where(f < heart_hz, 0, energy))
    total = total / total[-1]
    return f[np.searchsorted(total, share[0])], f[np.searchsorted(total, share[1])]


# ours: the band of each wave in every stretch, the ECG's power, rates, mains
uses = {"P wave": [], "R wave": []}
power98 = []
mains = {f0: [] for f0 in filters.MAINS_HZ}
for run in stats.RUNS:
    with contextlib.redirect_stdout(io.StringIO()):
        doses = Session(run).dose_averages()
    t_ms = doses["t_ecg"]
    for dose in doses["doses"]:
        points = ecg_points(t_ms, dose["beat"])
        heart_hz = 1000 / dose["rr_ms"]
        for name, start, end in [("P wave", points["P on"], points["P off"]), ("R wave", points["QRS on"], points["S"][0])]:
            uses[name].append(band(*wave_spectrum(t_ms, dose["beat"], start, end), heart_hz))
        power98.append(band(*wave_spectrum(t_ms, dose["beat"], t_ms[0], t_ms[-1]), 0, (0, 0.98))[1])
        t, ecg, rate = dose["raw"]
        f, power = welch(ecg, fs=rate, nperseg=int(4 * rate), scaling="spectrum")
        for f0 in mains:
            near = (np.abs(f - f0) > 2) & (np.abs(f - f0) < 6)
            line = np.sqrt(2 * power[np.argmin(np.abs(f - f0))])
            mains[f0].append(line if line > 3 * np.sqrt(2 * np.median(power[near])) else np.nan)
mains = {f0: np.nanmedian(sizes) for f0, sizes in mains.items() if np.sum(~np.isnan(sizes)) > len(sizes) / 2}
with contextlib.redirect_stdout(io.StringIO()):
    every = [Session(name).vitals() for name in SESSIONS]
heart = np.nanpercentile(np.concatenate([v["hr"] for v in every]), [1, 99]) / 60
print("ours, median band over %d stretches (lowest to highest):" % len(uses["R wave"]))
for name, found in uses.items():
    found = np.array(found)
    print("  %-10s %.1f-%.0f Hz (low cut %.1f to %.1f, high cut %.0f to %.0f)"
          % (name, *np.median(found, axis=0), *np.percentile(found[:, 0], [0, 100]), *np.percentile(found[:, 1], [0, 100])))
print("  heart rate %.2f-%.1f Hz, mains %s" % (*heart, ", ".join("%.0f µV at %d Hz" % (s, f0) for f0, s in mains.items())))
print("  98 %% of the beat's power below %.0f Hz (%.0f to %.0f)" % (np.median(power98), min(power98), max(power98)))


def spread(bands):
    """Median band of the stretches, and the range of their low and of their high cuts."""
    b = np.array(bands)
    return tuple(np.median(b, axis=0)), (b[:, 0].min(), b[:, 0].max()), (b[:, 1].min(), b[:, 1].max())


# ours: number -> (band, range of the low cut or None, range of the high cut or None)
OURS = {
    1: ((0, np.median(power98)), None, (min(power98), max(power98))),
    3: spread(uses["R wave"]),
    4: spread(uses["P wave"]),
    5: (tuple(heart), None, None),
    6: ((1.5, heart[0] / 2), None, tuple(heart / 2)),
    7: ((0.15, 1.5), None, None),
}
SPREAD = "lowest to highest over our %d recordings" % len(power98)      # what the thin red lines show


def rounded(value):
    """Two significant figures below 10, whole numbers above."""
    return "%g" % float("%.2g" % value) if value < 10 else "%.0f" % value


def rows():
    """The rows of the range chart: (group title, [(number, row name, [human, published, ours])]),
    each of the three None or (band, thin line from-to or None, text)."""
    recorders = [("", name, [(b[:2], None, "%g-%g (%s)" % b) if b else None for b in bands]) for name, *bands in RECORDERS]
    out = [("Recorders", recorders)]
    for kind, title in [("waveform", "ECG waveform: must lie inside the recorded band"),
                        ("rhythm", "Rhythms: measured from the R-wave times")]:
        found = []
        for n, name, k, human, pub in ITEMS:
            if k != kind:
                continue
            series = [(b, None, "%g-%g" % b) if b else None for b in (human, pub)]
            if n in OURS:
                (low, high), lows, highs = OURS[n]
                line = ((lows or (low,))[0], (highs or (0, high))[1]) if lows or highs else None
                text = "%s-%s" % (rounded(low), rounded(high))
                if name.startswith("HRV"):
                    text = text + ", up to %s" % rounded(highs[1]) if highs else text
                series.append(((low, high), line, text))
            else:
                series.append(None)
            found.append((str(n), name, series))
        out.append((title, found))
    return out


def log_axis(axis, ticks):
    axis.set_ticks(ticks, [plotting.plain(tick, None) for tick in ticks])


def range_chart():
    """The bands as bars on one frequency axis, our board's band shaded. Each row's bars sit one under
    the other (human, published, ours, as far as they exist) with the row's number and name centred
    on them; the values stand in a column on the right."""
    left, right = 0.02, 1000
    slot, gap = 0.45, 0.55                    # height of one bar's place, space between rows
    names_at = -132                           # points left of the axes where the number column starts

    def write(text, y, x_points, axes_x=0, **style):
        ax.annotate(text, (axes_x, y), xycoords=("axes fraction", "data"), xytext=(x_points, 0),
                    textcoords="offset points", va="center", **style)

    fig, ax = plt.subplots(figsize=(16 / 2.54, 12.5 / 2.54), layout="constrained")
    ax.axvspan(*BOARD_HZ, color=SHADE, linewidth=0, zorder=0)
    for f0 in BOARD_HZ:
        ax.axvline(f0, color=RED, linewidth=0.6, zorder=0.5)
    for f0 in mains:
        ax.axvline(f0, color=GREY, linestyle=":", linewidth=0.8, zorder=0.5)
    y = 0.0
    for title, found in rows():
        y -= 0.6
        ax.axhspan(y - 0.45, y + 0.45, color="white", linewidth=0, zorder=0.7)     # a clean strip for the title
        write(title, y, names_at, fontweight="bold", fontsize=7.5, color=INK)
        y -= 0.6
        for number, name, series in found:
            bars = [(s, color) for s, (_, color) in zip(series, SERIES) if s]
            middle = y - slot * len(bars) / 2
            write(number, middle, names_at, color=INK_2)
            write(name, middle, names_at + 14, color=INK_2)
            for k, (((low, high), line, text), color) in enumerate(bars):
                at = y - slot * (k + 0.5)
                if line:
                    ax.plot([max(line[0], left), line[1]], [at, at], color=color, linewidth=0.9, solid_capstyle="butt", zorder=1)
                ax.barh(at, high - max(low, left), left=max(low, left), height=0.3, color=color, linewidth=0, zorder=2)
                write(text, at, 6, axes_x=1, fontsize=6.5, color=color)
            y -= slot * len(bars) + gap
    ax.text(BOARD_HZ[0] * 1.08, 0.85, "our board records\n%g-%g Hz" % BOARD_HZ, color=RED, fontsize=7, va="center",
            linespacing=1.1)
    ax.text(min(mains) / 1.08, y + 0.25, "mains %s Hz (notched)" % ", ".join("%d" % f0 for f0 in mains),
            color=GREY, fontsize=6.5, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(left, right)
    ax.set_ylim(y - 0.1, 1.8)
    log_axis(ax.xaxis, [0.1, 1, 10, 100, 1000])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("frequency (Hz)")
    fig.legend([Patch(color=c) for _, c in SERIES] + [Line2D([], [], color=INK_2, linewidth=0.9)],
               [n for n, _ in SERIES] + [SPREAD], loc="outside upper center", ncol=4, frameon=False)
    plotting.save(fig, "ecg_frequency_ranges")


def number(ax, x, y, n, color, filled):
    """An item as its number in a circle: filled for the ECG waveform, open for a rhythm."""
    ax.text(x, y, str(n), ha="center", va="center", fontsize=5.5, fontweight="bold", color="white" if filled else color,
            zorder=5, bbox={"boxstyle": "circle,pad=0.2", "facecolor": color if filled else "white", "edgecolor": color,
                            "linewidth": 1})


def box(ax, low, high, what, color, style, shade=None, below=False):
    """A recorder: the box from (low, low) to (high, high), named in its corner below the diagonal
    (or under the box)."""
    if shade:
        ax.fill([low, low, high], [low, high, high], color=shade, linewidth=0, zorder=0)
    ax.plot([low, low, high, high, low], [low, high, high, low, low], linestyle=style, color=color, linewidth=1, zorder=1)
    ax.text(high / 1.1, low / 1.15 if below else low * 1.15, "%s %g-%g Hz" % (what, low, high), ha="right",
            va="top" if below else "bottom", fontsize=6.5, color=color)


def apart(a, b):
    """Whether two marks (x, y) are far enough apart on the log axes not to overlap."""
    return np.hypot(*np.log10(np.array(a) / np.array(b))) >= 0.16


def map_figure():
    """The bands as points (low cut across, high cut up) with the recorders as boxes: human and mouse."""
    fig, (hum, mou) = plt.subplots(1, 2, figsize=(16 / 2.54, 10.5 / 2.54), layout="constrained")
    for ax, title, low, note in [(hum, "a   Human ECG, published", 0.02, 10), (mou, "b   Mouse ECG, published and ours", 0.1, 20)]:
        ax.plot([low, 1000], [low, 1000], color="0.82", linewidth=0.6, zorder=0.5)
        ax.text(note, note / 1.4, "low cut = high cut", rotation=45, ha="center", va="center", fontsize=6,
                color=GREY, style="italic")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, 1000)
        ax.set_ylim(low, 1000)
        ax.set_box_aspect(1)
        log_axis(ax.xaxis, [0.1, 1, 10, 100, 1000])
        log_axis(ax.yaxis, [0.1, 1, 10, 100, 1000])
        ax.grid(False)
        ax.set_xlabel("low cut: high-pass (Hz)")
        ax.set_ylabel("high cut: low-pass (Hz)")
        ax.set_title(title, loc="left", fontweight="bold")
    for name, human, pub, ours in RECORDERS:
        if human:
            box(hum, *human, INK_2, "--" if name == "ECG recorder" else ":", below=name != "ECG recorder")
        if pub:
            box(mou, *pub, BLUE, "--")
        if ours:
            box(mou, *ours, RED, "-", SHADE, below=True)
    for n, name, kind, human, pub in ITEMS:
        filled = kind == "waveform"
        if human:
            number(hum, max(human[0], 0.02 * 1.4), human[1], n, GREY, filled)
        pub_x = max(pub[0], 0.1 * 1.4) if pub else None
        if n in OURS:
            (low, high), lows, highs = OURS[n]
            x = max(low, 0.1 * 1.4)
            if pub and not apart((pub_x, pub[1]), (x, high)):              # too close: drawn side by side
                pub_x, x = pub_x / 1.18, x * 1.18
            if lows:
                mou.plot(lows, [high, high], color=RED, linewidth=0.9, zorder=4)
            if highs:
                mou.plot([x, x], highs, color=RED, linewidth=0.9, zorder=4)
            number(mou, x, high, n, RED, filled)
        if pub:
            number(mou, pub_x, pub[1], n, BLUE, filled)
    for f0 in mains:
        mou.plot(f0, f0, "P", color=GREY, markersize=5, zorder=4)
    marks = [Line2D([], [], linestyle="none", marker="o", markersize=6, color=INK),
             Line2D([], [], linestyle="none", marker="o", markersize=6, color=INK, markerfacecolor="white"),
             *[Patch(color=c) for _, c in SERIES], Line2D([], [], color=INK_2, linewidth=0.9),
             Line2D([], [], linestyle="none", marker="P", markersize=5, color=GREY)]
    fig.legend(marks, ["ECG waveform", "rhythm (from the R-wave times)", *[n for n, _ in SERIES],
                       SPREAD, "mains (notched out)"], loc="outside upper center", ncol=4, frameon=False)
    fig.legend([Line2D([], [], linestyle="none", marker="$%d$" % n, markersize=5, color=INK) for n, *_ in ITEMS],
               [name for _, name, *_ in ITEMS], loc="outside lower center", ncol=4, frameon=False)
    plotting.save(fig, "ecg_frequency_map")


plotting.figure_style()
range_chart()
map_figure()
plt.show()
