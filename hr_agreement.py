"""Heart rate from the PPG against heart rate from the ECG (the reference), in every recording in
data/ with a usable PPG: a Bland-Altman plot (Bland and Altman 1986, Lancet 327:307-310).

Recordings where the ECG's beat repeats clearly in fewer than half the 30 s pieces are left out
(the ECG is the reference). One value pair per 30 s piece in which the beat repeats clearly:
ECG: 60 / median R-R. PPG: the strongest frequency in the heart band (Welch), as in ppg_vitals.py.
Bias = mean of PPG - ECG; limits of agreement = bias ± 1.96 SD.
"""
import contextlib
import glob
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import filters
import style
from recording import DATA, load, unzip

for z in glob.glob(os.path.join(DATA, "*.zip")):
    unzip(os.path.basename(z)[:-4])

ecg_hr, ppg_hr, session = [], [], []
print("recording: pieces with a clear beat, PPG within 5 % of ECG")
for f in sorted(glob.glob(os.path.join(DATA, "*", "**", "*.csv"), recursive=True), key=os.path.basename):
    day = os.path.basename(f)[17:19] + " Oct"
    mouse = os.path.basename(os.path.dirname(f))
    label = day + (", " + mouse[6:-1] + " g" if mouse.startswith("mouse_") and mouse.endswith("g") else "")
    pieces = []
    with contextlib.redirect_stdout(io.StringIO()):
        rec = load(f)
        r = filters.r_waves(rec["ecg"], rec["fs_ecg"], pieces)
    if "ir_nA" not in rec or sum(c for _, _, c in pieces) < len(pieces) / 2:
        continue                                         # no PPG, or the ECG reference is unclear in most of it
    r_times = rec["t_ecg"][r]
    fp, tp = rec["fs_ppg"], rec["t_ppg"]
    heart = filters.heart_band(filters.ppg_lowpass(rec["ir_nA"], fp), fp)
    pairs = []
    for first, end, clear in pieces:
        t0, t1 = rec["t_ecg"][first], rec["t_ecg"][end - 1]
        near = (tp >= t0) & (tp < t1)
        rr = np.diff(r_times[(r_times >= t0) & (r_times < t1)])
        rr = rr[(rr >= 0.08) & (rr <= 0.30)]
        if not clear or len(rr) < 10 or t1 - t0 < 20:
            continue
        fr, p = welch(heart[near], fs=fp, nperseg=int(10 * fp), nfft=int(60 * fp))
        band = (fr >= filters.HEART_HZ[0]) & (fr <= filters.HEART_HZ[1])
        pairs.append((60 / np.median(rr), 60 * fr[band][np.argmax(p[band])]))
    if pairs:
        e, q = np.array(pairs).T
        ecg_hr.extend(e)
        ppg_hr.extend(q)
        session.extend([label] * len(e))
        print("  %s: %d, %d" % (os.path.basename(f), len(pairs), np.sum(np.abs(q / e - 1) <= 0.05)))

ecg_hr, ppg_hr = np.array(ecg_hr), np.array(ppg_hr)
mean = (ecg_hr + ppg_hr) / 2
diff = ppg_hr - ecg_hr
bias, sd = np.mean(diff), np.std(diff, ddof=1)
within = np.mean(np.abs(diff / ecg_hr) <= 0.05)
print("all: %d pieces; bias %.1f beats/min, limits of agreement %.1f to %.1f; PPG within 5 %% of ECG in %.0f %%"
      % (len(diff), bias, bias - 1.96 * sd, bias + 1.96 * sd, 100 * within))
good = np.abs(diff / ecg_hr) <= 0.05
print("pieces within 5 %%: bias %.1f, limits %.1f to %.1f beats/min" % (
    np.mean(diff[good]), np.mean(diff[good]) - 1.96 * np.std(diff[good], ddof=1),
    np.mean(diff[good]) + 1.96 * np.std(diff[good], ddof=1)))

style.thesis()
fig = plt.figure(figsize=(6.3, 5.4), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15])
ax = top.subplots()
piece = np.arange(1, len(ecg_hr) + 1)
ax.plot(piece, ecg_hr, "o-", color=style.BLUE, markersize=2.5, linewidth=0.8, label="ECG")
ax.plot(piece, ppg_hr, "o", color=style.ORANGE, markersize=2.5, markeredgewidth=0, label="PPG")
starts = [0] + [i for i in range(1, len(session)) if session[i] != session[i - 1]]
for k, s in enumerate(starts):
    end = starts[k + 1] if k + 1 < len(starts) else len(session)
    if s:
        ax.axvline(s + 0.5, color=style.AXIS, linewidth=0.5, zorder=0)
    ax.text((s + end) / 2 + 0.5, 1.02, session[s], transform=ax.get_xaxis_transform(), ha="center",
            fontsize=7, color=style.INK_2)
ax.set_xlim(0.5, len(piece) + 0.5)
ax.set_xlabel("30 s piece, in time order")
ax.set_ylabel("Heart rate (beats/min)")
ax.legend(loc="lower left", ncol=2)
style.letter(ax, "A")

axes = bottom.subplots(1, 2)
ax = axes[0]
ax.plot(ecg_hr, ppg_hr, "o", color=style.INK_2, markersize=3, alpha=0.5, markeredgewidth=0)
low, high = 200, 650
ax.plot([low, high], [low, high], color=style.AXIS, linewidth=0.8, zorder=0)
ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_aspect("equal")
ax.set_xlabel("ECG heart rate (beats/min)")
ax.set_ylabel("PPG heart rate (beats/min)")
style.letter(ax, "B")

ax = axes[1]
ax.plot(mean, diff, "o", color=style.INK_2, markersize=3, alpha=0.5, markeredgewidth=0)
for y, text in [(bias, "bias %.1f" % bias), (bias + 1.96 * sd, "+1.96 SD %.0f" % (bias + 1.96 * sd)),
                (bias - 1.96 * sd, "−1.96 SD %.0f" % (bias - 1.96 * sd))]:
    ax.axhline(y, color=style.INK_2, linewidth=0.8, linestyle="-" if text.startswith("bias") else "--")
    ax.text(1.0, y, " " + text, transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=style.INK_2)
ax.set_title("%.0f %% of %d pieces within 5 %% of the ECG" % (100 * within, len(diff)), loc="left",
             fontsize=7, color=style.INK_2)
ax.set_xlabel("Mean of ECG and PPG (beats/min)")
ax.set_ylabel("PPG − ECG (beats/min)")
style.letter(ax, "C")

style.save(fig, __file__, "hr_agreement")
plt.show()
