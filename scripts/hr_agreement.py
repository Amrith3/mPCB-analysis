"""Heart rate from the PPG against heart rate from the ECG (the reference), in every recording in
data/ with a usable PPG: a Bland-Altman plot (Bland and Altman 1986, Lancet 327:307-310).

Recordings where the ECG's beat repeats clearly in fewer than half the 30 s pieces are left out
(the ECG is the reference). One value pair per 30 s piece in which the beat repeats clearly:
ECG: 60 / median R-R. PPG: the strongest frequency in the heart band (analysis.ppg_heart_rate).
Bias = mean of PPG - ECG; limits of agreement = bias ± 1.96 SD.
It also prints how often the PPG heart rate is within 5 % of the ECG's at each PPG SNR (analysis.snr_db),
and how many of the wrong pieces are at twice the breathing rate.
"""
import contextlib
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the main folder
from mpcb import analysis, filters, plotting  # noqa: E402
from mpcb.recording import Recording, all_files  # noqa: E402
from mpcb.sessions import started  # noqa: E402

ecg_hr, ppg_hr, ppg_snr, breath_rate, session = [], [], [], [], []
print("recording: pieces with a clear beat, PPG within 5 % of ECG")
for f in sorted(all_files(), key=os.path.basename):
    when = started(os.path.basename(f))
    day = "%d %s" % (when.day, when.strftime("%b"))
    mouse = os.path.basename(os.path.dirname(f))
    label = day + (", " + mouse[6:-1] + " g" if mouse.startswith("mouse_") and mouse.endswith("g") else "")
    pieces = []
    with contextlib.redirect_stdout(io.StringIO()):
        rec = Recording(f)
        r = analysis.r_waves(rec.ecg, rec.fs_ecg, pieces)
    if not rec.has_ppg or sum(c for _, _, c in pieces) < len(pieces) / 2:
        continue                                         # no PPG, or the ECG reference is unclear in most of it
    r_times = rec.t_ecg[r]
    r_t = analysis.r_times(rec.ecg, rec.t_ecg, rec.fs_ecg, r)
    fp, tp = rec.fs_ppg, rec.t_ppg
    light = filters.ppg_lowpass(rec.ir_nA, fp)
    heart = filters.heart_band(light, fp)
    breath = filters.breathing_band(light, fp)
    pairs = []
    for first, end, clear in pieces:
        t0, t1 = rec.t_ecg[first], rec.t_ecg[end - 1]
        near = (tp >= t0) & (tp < t1)
        rr = np.diff(r_times[(r_times >= t0) & (r_times < t1)])
        rr = rr[(rr >= 0.08) & (rr <= 0.30)]
        if not clear or len(rr) < 10 or t1 - t0 < 20:
            continue
        snr = analysis.snr_db(tp, heart, r_t[(r_t >= t0) & (r_t < t1)], np.median(rr))
        pairs.append((60 / np.median(rr), analysis.ppg_heart_rate(heart[near], fp, 60), snr,
                      analysis.breathing_rate(breath[near], fp)))
    if pairs:
        e, q, n, b = np.array(pairs).T
        ecg_hr.extend(e)
        ppg_hr.extend(q)
        ppg_snr.extend(n)
        breath_rate.extend(b)
        session.extend([label] * len(e))
        print("  %s: %d, %d" % (os.path.basename(f), len(pairs), np.sum(np.abs(q / e - 1) <= 0.05)))

ecg_hr, ppg_hr, ppg_snr, breath_rate = np.array(ecg_hr), np.array(ppg_hr), np.array(ppg_snr), np.array(breath_rate)
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

right = np.abs(ppg_hr / ecg_hr - 1) <= 0.05
for lo, hi in [(-40, -5), (-5, 0), (0, 5), (5, 10), (10, 40)]:
    sel = (ppg_snr >= lo) & (ppg_snr < hi)
    if sel.any():
        print("PPG SNR %d to %d dB: %d pieces, PPG within 5 %% of ECG in %d" % (lo, hi, sel.sum(), right[sel].sum()))

twice_breathing = ~right & (np.abs(ppg_hr / (2 * breath_rate) - 1) <= 0.05)
print("PPG wrong: %d pieces, %d of them at twice the breathing rate" % ((~right).sum(), twice_breathing.sum()))

plotting.figure_style()
fig = plt.figure(figsize=(6.3, 5.6), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15])
ax = top.subplots()
ax.set_title("Heart rate of each 30 s piece", loc="left", pad=14)
piece = np.arange(1, len(ecg_hr) + 1)
ax.plot(piece, ecg_hr, "o-", color=plotting.BLUE, markersize=2.5, linewidth=0.8, label="ECG")
ax.plot(piece, ppg_hr, "o", color=plotting.ORANGE, markersize=2.5, markeredgewidth=0, label="PPG")
starts = [0] + [i for i in range(1, len(session)) if session[i] != session[i - 1]]
for k, s in enumerate(starts):
    end = starts[k + 1] if k + 1 < len(starts) else len(session)
    if s:
        ax.axvline(s + 0.5, color=plotting.AXIS, linewidth=0.5, zorder=0)
    ax.text((s + end) / 2 + 0.5, 1.02, session[s], transform=ax.get_xaxis_transform(), ha="center",
            fontsize=7, color=plotting.INK_2)
ax.set_xlim(0.5, len(piece) + 0.5)
ax.set_xlabel("30 s piece, in time order")
ax.set_ylabel("Heart rate (beats/min)")
ax.legend(loc="lower center", ncol=2)
plotting.letter(ax, "A")

axes = bottom.subplots(1, 2)
ax = axes[0]
ax.set_title("PPG against ECG", loc="left")
ax.plot(ecg_hr, ppg_hr, "o", color=plotting.INK_2, markersize=3, alpha=0.5, markeredgewidth=0)
low, high = 200, 650
ax.plot([low, high], [low, high], color=plotting.AXIS, linewidth=0.8, zorder=0)
ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_aspect("equal")
ax.set_xlabel("ECG heart rate (beats/min)")
ax.set_ylabel("PPG heart rate (beats/min)")
plotting.letter(ax, "B")

ax = axes[1]
ax.plot(mean, diff, "o", color=plotting.INK_2, markersize=3, alpha=0.5, markeredgewidth=0)
for y, text in [(bias, "bias %.1f" % bias), (bias + 1.96 * sd, "+1.96 SD %.0f" % (bias + 1.96 * sd)),
                (bias - 1.96 * sd, "−1.96 SD %.0f" % (bias - 1.96 * sd))]:
    ax.axhline(y, color=plotting.INK_2, linewidth=0.8, linestyle="-" if text.startswith("bias") else "--")
    ax.text(1.0, y, " " + text, transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=plotting.INK_2)
ax.set_title("Bland-Altman: %.0f %% of %d pieces within 5 %%" % (100 * within, len(diff)), loc="left")
ax.set_xlabel("Mean of ECG and PPG (beats/min)")
ax.set_ylabel("PPG − ECG (beats/min)")
plotting.letter(ax, "C")
bottom.align_titles()

plotting.save(fig, "hr_agreement")
plt.show()
