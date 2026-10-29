"""One isoflurane session, one column per dose: the averaged ECG beat, the averaged PPG pulse,
20 s of breathing and the skin temperature, all from the last 2 min recorded at each dose.

The averaged beat and pulse come from sessions.dose_averages; the R wave and the baseline (P onset)
from sessions.ecg_points, the same ones table.py measures. Only R is marked: a mouse QRS starts
with R (no Q wave), and the dip before R is the ECG chip's filter ringing.
Breathing: the PPG breathing band, 20 s around the last clear breathing value at that dose.
"""
import matplotlib.pyplot as plt
import numpy as np

import sessions
import style

SESSION = "26 Oct, 30 g"
BREATH_S = 20

d = sessions.dose_averages(SESSION)
v = d["v"]


def note(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=style.INK_2)


style.thesis()
n = len(d["doses"])
fig, axes = plt.subplots(4, n, figsize=(6.3, 7.4), sharex="row", sharey="row", layout="constrained")
for col, c in enumerate(d["doses"]):
    ax = axes[0, col]
    points = sessions.ecg_points(d["t_ecg"], c["beat"])
    ax.plot(d["t_ecg"], c["beat"] - points["base"], color=style.BLUE)
    x, y = points["R"]
    ax.plot(x, y, "o", color=style.INK, markersize=2.5)
    ax.annotate("R %.0f" % y, (x, y), xytext=(-4, 2), textcoords="offset points", ha="right", fontsize=6.5,
                color=style.INK_2)
    ax.set_title("%g %%\n%.0f beats/min" % (c["level"], 60000 / c["rr_ms"]))
    ax.set_xlabel("Time from R (ms)")

    ax = axes[1, col]
    pulse = c["pulse"] - np.median(c["pulse"])
    ax.plot(d["t_ppg"], pulse, color=style.BLUE)
    after = d["t_ppg"] >= 0
    x, y = d["t_ppg"][after][np.argmax(pulse[after])], np.max(pulse[after])
    ax.plot(x, y, "o", color=style.INK, markersize=2.5)
    ax.annotate("%.0f ms" % x, (x, y), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=style.INK_2)
    ax.set_xlabel("Time from R (ms)")

    ax = axes[2, col]
    idx = c["last"]
    clear = idx[~np.isnan(v["breathing"][idx])]
    middle = v["t"][clear[-1] if len(clear) else idx[-1]]
    near = (d["t_breath"] >= middle - BREATH_S / 2) & (d["t_breath"] < middle + BREATH_S / 2)
    ax.plot(d["t_breath"][near] - d["t_breath"][near][0], d["breath"][near], color=style.BLUE, linewidth=0.8)
    rate = sessions.steady(v, "breathing", c["start"], c["end"])
    note(ax, "-" if rate is None else "%.0f breaths/min" % rate[0])
    ax.set_xlabel("Time (s)")

    ax = axes[3, col]
    lo, hi = v["t"][idx[0]] - sessions.STEP_S / 2, v["t"][idx[-1]] + sessions.STEP_S / 2
    near = (d["t_temp"] >= lo) & (d["t_temp"] < hi)
    second = np.floor(d["t_temp"][near] - lo).astype(int)
    count = np.bincount(second)
    have = count > 0
    ax.plot(np.flatnonzero(have) / 60, (np.bincount(second, d["temp"][near]) / np.maximum(count, 1))[have],
            color=style.BLUE)                      # 1 s averages
    note(ax, "%.1f °C" % np.mean(d["temp"][near]))
    ax.set_xlabel("Time (min)")
    for ax in axes[:2, col]:
        ax.axvline(0, color=style.AXIS, linewidth=0.5, zorder=0)

axes[0, 0].margins(y=0.12)
axes[2, 0].margins(y=0.2)
axes[3, 0].margins(y=0.25)
axes[0, 0].set_ylabel("ECG (µV)")
axes[1, 0].set_ylabel("PPG pulse\n(% of light level)")
axes[2, 0].set_ylabel("Breathing\n(% of light level)")
axes[3, 0].set_ylabel("Skin temperature\n(°C)")
for row, text in enumerate("ABCD"):
    style.letter(axes[row, 0], text)
fig.align_ylabels(axes[:, 0])

style.save(fig, __file__, "doses")
plt.show()
