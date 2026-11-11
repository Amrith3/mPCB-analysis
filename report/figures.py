"""The graphs of the results report, one graph per file, each with its own title.

ECG (22 Oct 14:54:30, Mouse 1): ecg_trace, ecg_beats_enlarged, ecg_spectrum, and the spectrum after each
filter step: ecg_spectrum_after_highpass, ecg_spectrum_after_50hz_notch, ecg_spectrum_after_150hz_notch.
Averaged beat per dose (26 Oct, Mouse 1): beat_1pct, beat_2pct, beat_4pct. For display the beat is
interpolated to 8192 samples per second and smoothed with a Gaussian (sigma 0.5 ms); the intervals are
measured on the beat before smoothing.
Isoflurane: breathing_over_time (single breaths and their median over 30 s), heart_rate_over_time (26 Oct),
breathing_per_dose, heart_rate_per_dose (3 runs).
PPG (22 Oct 14:54:30): ppg_filtered, ppg_spectrum, ppg_on_r_waves.
PPG pulse per dose (26 Oct): pulse_1pct, pulse_2pct, pulse_4pct, pulse_0pct.
PPG against ECG heart rate (14 recordings): hr_ppg_vs_ecg, bland_altman.
Skin temperature (26 Oct): skin_temperature.
The font is DejaVu Sans, which comes with matplotlib and is free to use.
"""
import contextlib
import io
import os
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.ndimage import gaussian_filter1d, uniform_filter1d  # noqa: E402
from scipy.signal import butter, find_peaks, sosfiltfilt, welch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from mpcb import analysis, filters, plotting, sessions, stats  # noqa: E402
from mpcb.plotting import AXIS, BLUE, HAIRLINE, INK_2, ORANGE  # noqa: E402
from mpcb.recording import FILE, Recording  # noqa: E402

SESSION = "26 Oct, 30 g"
RUNS = [("22 Oct, 30 g", "Mouse 1, 22 Oct", BLUE), ("26 Oct, 30 g", "Mouse 1, 26 Oct", ORANGE),
        ("22 Oct, 25 g", "Mouse 2, 22 Oct", "#2f9e6b")]
RED = "tab:red"


def style():
    """The repository's figure style with DejaVu Sans, bold titles on the left."""
    plotting.figure_style()
    plt.rcParams.update({"font.sans-serif": ["DejaVu Sans"], "axes.titleweight": "bold", "axes.titlelocation": "left"})


def graph(title, size=(6.3, 2.6)):
    """A new figure with one axis and its title."""
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    ax.set_title(title)
    return fig, ax


def save(fig, name):
    """Save the figure as PDF and PNG in outputs/report/, then close it."""
    plotting.save(fig, name, "report")
    plt.close(fig)


def dose_bands(ax, v):
    """Shade every other dose and write the dose at the top."""
    for i, (level, start, end) in enumerate(sessions.epochs(v)):
        if i % 2 == 0:
            ax.axvspan(start / 60, end / 60, color=HAIRLINE, alpha=0.7, linewidth=0, zorder=0)
        ax.text((start + end) / 120, 1.0, "%g %%" % level, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=INK_2)
    ax.set_xlim(0, v["end"] / 60)
    ax.set_xlabel("Time (min)")


style()
with contextlib.redirect_stdout(io.StringIO()):
    rec = Recording(FILE)
    d = sessions.dose_averages(SESSION, ppg_ms=(-50, 200))
v = d["v"]
fs, t = rec.fs_ecg, rec.t_ecg
ecg_mv = filters.notches(filters.ecg_highpass(rec.ecg, fs), fs) / 1000
r = analysis.r_waves(rec.ecg, fs)


def evenness(start):
    """How even the R waves are in the 5 s from start: smallest R height / median R height."""
    amp = ecg_mv[r[(t[r] >= start) & (t[r] < start + 5)]]
    return amp.min() / np.median(amp)


def ms(x):
    """Milliseconds as a number of ECG samples."""
    return int(round(x * fs / 1000))


# ECG -------------------------------------------------------------------------------------------
t0 = max(range(20, 640, 5), key=evenness)                     # the 5 s with the most even R waves
rs = r[(t[r] >= t0) & (t[r] < t0 + 5)]
sel = (t >= t0) & (t < t0 + 5)
fig, ax = graph("ECG with the R waves found (22 Oct, Mouse 1, 2 % isoflurane)")
ax.plot(t[sel] - t0, ecg_mv[sel], color=BLUE, linewidth=0.7)
ax.plot(t[rs] - t0, ecg_mv[rs] + 0.08, "v", color=INK_2, markersize=2.5)
ax.set_xlim(0, 5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("ECG (mV)")
save(fig, "ecg_trace")

k = rs[19]
inside = (t >= t[rs[18]] - 0.06) & (t < t[rs[20]] + 0.06)
fig, ax = graph("Three ECG beats enlarged", (4.0, 2.8))
ax.plot(1000 * (t[inside] - t[k]), ecg_mv[inside], color=BLUE, linewidth=0.9)
p_i = k - ms(90) + np.argmax(ecg_mv[k - ms(90):k - ms(25)])
s_i = k + np.argmin(ecg_mv[k:k + ms(8)])
j_i = k + ms(6) + np.argmax(ecg_mv[k + ms(6):k + ms(16)])
for i, label, dx, dy in [(k, "R", -8, 0.03), (p_i, "P", 0, 0.04), (s_i, "S", 8, -0.04), (j_i, "J", 9, 0.0)]:
    ax.text(1000 * (t[i] - t[k]) + dx, ecg_mv[i] + dy, label, ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Time from the middle R wave (ms)")
ax.set_ylabel("ECG (mV)")
save(fig, "ecg_beats_enlarged")

seg = (t >= t0 - 30) & (t < t0 + 30)
f, pw = welch(rec.ecg[seg] - np.mean(rec.ecg[seg]), fs=fs, nperseg=int(8 * fs))
db = 10 * np.log10(pw)
fig, ax = graph("Spectrum of the raw ECG, before filtering", (6.3, 2.8))
ax.semilogx(f[1:], db[1:], color=BLUE, linewidth=0.6)
plt.sca(ax)
plotting.frequency_axis(fs)
ax.set_xlim(1, fs / 2)
fhr = 1 / np.median(np.diff(t[rs]))
k1 = np.argmin(np.abs(f - fhr))
ax.text(f[k1], db[k1] + 4, "heart rate %.1f Hz" % f[k1], ha="center", va="bottom", fontsize=7, color=ORANGE)
for m in range(2, 9):
    kh = np.argmin(np.abs(f - m * fhr))
    ax.plot(f[kh], db[kh] + 3, "v", color=ORANGE, markersize=2.5)
for fm in (50, 150):
    ax.axvline(fm, color=RED, linewidth=0.8, linestyle=":")
    ax.text(fm * 1.04, db.max(), "mains %d Hz" % fm, fontsize=7, color=RED, va="top")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB)")
save(fig, "ecg_spectrum")
print("ECG: %.0f s to %.0f s, %.0f beats/min, spectrum peak %.2f Hz" % (t0, t0 + 5, 60 * fhr, f[k1]))

# spectrum after each filter step, same 60 s -----------------------------------------------------
hp = filters.ecg_highpass(rec.ecg, fs)
n50 = filters.notch(hp, fs, 50)
n150 = filters.notch(n50, fs, 150)
for title, before, after, band, name in [
        ("after the 3 Hz high-pass", rec.ecg[seg] - np.mean(rec.ecg[seg]), hp[seg], (0.2, 2), "ecg_spectrum_after_highpass"),
        ("after the 50 Hz notch", hp[seg], n50[seg], (49.5, 50.5), "ecg_spectrum_after_50hz_notch"),
        ("after the 150 Hz notch", n50[seg], n150[seg], (149.5, 150.5), "ecg_spectrum_after_150hz_notch")]:
    fb, pb = welch(before, fs=fs, nperseg=int(8 * fs))
    fa, pa = welch(after, fs=fs, nperseg=int(8 * fs))
    inb = (fb >= band[0]) & (fb <= band[1])
    drop = 10 * np.log10(np.mean(pb[inb]) / np.mean(pa[inb]))
    fig, ax = graph("Spectrum of the ECG " + title, (6.3, 2.6))
    ax.semilogx(fb[1:], 10 * np.log10(pb[1:]), color=AXIS, linewidth=0.8, label="before this filter")
    ax.semilogx(fa[1:], 10 * np.log10(pa[1:]), color=BLUE, linewidth=0.6, label="after this filter")
    ax.axvspan(*band, color=RED, alpha=0.15, linewidth=0)
    plt.sca(ax)
    plotting.frequency_axis(fs)
    ax.set_xlim(0.3, fs / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.legend(loc="lower center", fontsize=7)
    save(fig, name)
    print("%s: %.0f dB less from %g to %g Hz" % (title, drop, *band))

# averaged beat per dose ------------------------------------------------------------------------
for level in (1, 2, 4):
    c = [c for c in d["doses"] if c["level"] == level][0]
    p = sessions.ecg_points(d["t_ecg"], c["beat"])
    pq, pd, qrs = p["QRS on"] - p["P on"], p["P off"] - p["P on"], p["QRS off"] - p["QRS on"]
    fig, ax = graph("Averaged ECG beat at %g %% isoflurane (26 Oct, Mouse 1)" % level, (4.0, 2.8))
    ax.axvspan(p["P on"], p["P off"], color=ORANGE, alpha=0.15, linewidth=0)
    ax.axvspan(p["QRS on"], p["QRS off"], color=BLUE, alpha=0.15, linewidth=0)
    t_fine = np.arange(d["t_ecg"][0], d["t_ecg"][-1], 1000 / 8192)
    smooth = gaussian_filter1d(CubicSpline(d["t_ecg"], c["beat"])(t_fine), 0.5 * 8192 / 1000)
    ax.plot(t_fine, (smooth - p["base"]) / 1000, color=BLUE, linewidth=1)
    ax.text((p["P on"] + p["P off"]) / 2, 0.5, "P\n%.1f ms" % pd, ha="center", fontsize=7, color=INK_2)
    ax.text(p["QRS off"] + 2, 0.5, "QRS\n%.1f ms" % qrs, ha="left", fontsize=7, color=INK_2)
    ax.annotate("", (p["P on"], -0.5), (p["QRS on"], -0.5), arrowprops=dict(arrowstyle="<->", color=INK_2, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text((p["P on"] + p["QRS on"]) / 2, -0.47, "PQ %.1f ms" % pq, ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xlim(-90, 60)
    ax.set_ylim(-0.6, 0.7)
    ax.set_xlabel("Time from R (ms)")
    ax.set_ylabel("ECG (mV)")
    save(fig, "beat_%gpct" % level)
    print("beat %g %%: P %.1f, PQ %.1f, QRS %.1f ms, %d beats" % (level, pd, pq, qrs, c["beats"]))

# breathing and heart rate against the dose -----------------------------------------------------
with contextlib.redirect_stdout(io.StringIO()):
    tb, rb = sessions.Session(SESSION).breaths()
median_30s = np.array([np.median(rb[np.abs(tb - x) < 15]) if np.sum(np.abs(tb - x) < 15) >= 3 else np.nan for x in v["t"]])
fig, ax = graph("Breathing rate during the 26 Oct session (Mouse 1)")
ax.plot(tb / 60, rb, ".", color=AXIS, markersize=1.6, markeredgewidth=0, label="single breaths")
ax.plot(v["t"] / 60, median_30s, color=BLUE, label="median of the breaths in 30 s")
dose_bands(ax, v)
ax.set_ylabel("Breathing rate (per min)")
ax.set_ylim(0, 260)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, fontsize=7, markerscale=4)
save(fig, "breathing_over_time")
fig, ax = graph("Heart rate during the 26 Oct session (Mouse 1)")
tm, y = sessions.with_gaps(v, "hr")
ax.plot(tm, y, color=BLUE)
dose_bands(ax, v)
ax.set_ylabel("Heart rate (per min)")
ax.set_ylim(None, np.nanmax(y) * 1.12)
save(fig, "heart_rate_over_time")

with contextlib.redirect_stdout(io.StringIO()):                # key -> dose -> each run's mean of the last 2 min
    per_dose = {key: stats.per_dose(key, [run for run, _, _ in RUNS]) for key in ("hr", "breathing")}
for key, label, name in [("breathing", "Breathing rate", "breathing_per_dose"), ("hr", "Heart rate", "heart_rate_per_dose")]:
    fig, ax = graph("%s at each dose, three runs" % label, (4.0, 3.0))
    for i, (_, run_label, colour) in enumerate(RUNS):
        ax.plot([1, 2, 4], [per_dose[key][lv][i] for lv in (1, 2, 4)], "o-", color=colour, markersize=3, label=run_label)
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("Isoflurane (%)")
    ax.set_ylabel("%s (per min)" % label)
    ax.legend(fontsize=7)
    save(fig, name)
    print("%s per dose: %s" % (label, np.round([per_dose[key][lv] for lv in (1, 2, 4)], 1).T.tolist()))

# PPG -------------------------------------------------------------------------------------------
ir, fp, tp = rec.ir_nA, rec.fs_ppg, rec.t_ppg
bp = sosfiltfilt(butter(2, [0.1, 15], btype="band", fs=fp, output="sos"), ir)
br = filters.breathing_band(filters.ppg_lowpass(ir, fp), fp)
pk, _ = find_peaks(br, distance=int(fp))
pk = pk[(tp[pk] > 20) & (tp[pk] < tp[-1] - 20)]
tb = tp[pk[np.argmax([np.max(bp[np.abs(tp - tp[i]) < 0.3]) for i in pk])]]       # the breath with the largest spike
near = np.abs(tp - tb) < 0.3
tb = tp[near][np.argmax(bp[near])]
win = (tp >= tb - 2.5) & (tp < tb + 3.5)
fig, ax = graph("PPG filtered from 0.1 to 15 Hz around a breath (22 Oct, Mouse 1)")
ax.plot(tp[win] - (tb - 2.5), bp[win], color=BLUE, linewidth=0.7)
ax.axvspan(2.3, 2.7, color=ORANGE, alpha=0.15, linewidth=0)
ax.text(2.5, bp[win].max(), "breath", ha="center", va="bottom", fontsize=7, color=INK_2)
calm = win & (tp > tb + 1.3) & (tp < tb + 2.1)
top = bp[calm].max() + 0.15 * np.ptp(bp[win])
ax.annotate("", (3.8, top), (4.6, top), arrowprops=dict(arrowstyle="|-|", color=INK_2, lw=0.7, mutation_scale=2))
ax.text(4.2, top, "heartbeats", ha="center", va="bottom", fontsize=7, color=INK_2)
ax.set_xlim(0, 6)
ax.set_ylim(None, bp[win].max() + 0.3 * np.ptp(bp[win]))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Light change (nA)")
save(fig, "ppg_filtered")

seg = (tp >= 100) & (tp < 160)
f = np.fft.rfftfreq(seg.sum(), 1 / fp)
amp = 2 * np.abs(np.fft.rfft(ir[seg] - np.mean(ir[seg]))) / seg.sum()
show = (f >= 0.1) & (f <= 30)
low = show & (f < 1.5)
fb, bmax = f[low][np.argmax(amp[low])], amp[low].max()
heart = (f >= 5) & (f <= 7.5)
fh, hmax = f[heart][np.argmax(amp[heart])], amp[heart].max()
fig, ax = graph("Spectrum of the raw PPG, 60 s", (4.5, 3.0))
ax.loglog(f[show], amp[show], color=BLUE, linewidth=0.6)
ax.text(fb, bmax * 1.3, "breathing\n%.2f Hz" % fb, ha="center", va="bottom", fontsize=7, color=INK_2)
ax.text(fh, hmax * 1.3, "heart\n%.2f Hz" % fh, ha="center", va="bottom", fontsize=7, color=INK_2)
for m in (2, 3):
    nk = show & (np.abs(f - m * fh) < 0.4)
    ax.text(f[nk][np.argmax(amp[nk])], amp[nk].max() * 1.3, "%d×" % m, ha="center", va="bottom", fontsize=7, color=INK_2)
ax.set_ylim(None, hmax * 6)
ax.set_xlim(0.1, 30)
ax.set_xticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
ax.minorticks_off()
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude (nA)")
save(fig, "ppg_spectrum")

lp = filters.ppg_lowpass(ir, fp)
lp = lp - uniform_filter1d(lp, int(fp))
lags = np.arange(-150, 151, 2) / 1000
rt = t[r][(t[r] > tp[0] + 0.2) & (t[r] < tp[-1] - 0.2)]
pulse = np.median([np.interp(x + lags, tp, lp) for x in rt], axis=0)
pulse -= pulse.min()
fig, ax = graph("PPG averaged on the ECG R waves (%d beats)" % len(rt), (4.0, 2.8))
ax.plot(lags * 1000, pulse, color=BLUE, linewidth=1)
ax.axvline(0, color=AXIS, linewidth=0.6, linestyle=":")
ax.set_xlabel("Time from R (ms)")
ax.set_ylabel("Light change (nA)")
save(fig, "ppg_on_r_waves")
print("PPG: breathing %.2f Hz, heart %.2f Hz, pulse %.1f nA (%.2f %% of %.0f nA) over %d R waves" % (
    fb, fh, np.ptp(pulse), 100 * np.ptp(pulse) / np.mean(ir), np.mean(ir), len(rt)))

# PPG pulse per dose ----------------------------------------------------------------------------
after = d["t_ppg"] > 0
sizes = [np.max(c["pulse"][after]) - np.min(c["pulse"]) for c in d["doses"]]
for c, size in zip(d["doses"], sizes):
    y = c["pulse"] - np.min(c["pulse"])
    i_max = int(np.flatnonzero(after)[np.argmax(y[after])])
    fig, ax = graph("PPG pulse at %g %% isoflurane (26 Oct, Mouse 1)" % c["level"], (4.0, 2.8))
    ax.plot(d["t_ppg"], y, color=BLUE, linewidth=1)
    ax.axvline(0, color=AXIS, linewidth=0.6, linestyle=":")
    ax.plot([d["t_ppg"][i_max]] * 2, [0, y[i_max]], color=ORANGE, linewidth=1.2)
    ax.annotate("%.3f %% (%.0f %% of 1 %%)" % (size, 100 * size / sizes[0]), (d["t_ppg"][i_max], y[i_max]), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom", fontsize=7, color=ORANGE)
    ax.set_xlim(-50, 200)
    ax.set_ylim(0, max(sizes) * 1.25)
    ax.set_xlabel("Time from R (ms)")
    ax.set_ylabel("PPG pulse (% of the light)")
    save(fig, "pulse_%gpct" % c["level"])
    print("pulse %g %%: %.3f %% of the light, %.0f %% of 1 %%, %d beats" % (c["level"], size, 100 * size / sizes[0], c["beats"]))

# PPG against ECG heart rate --------------------------------------------------------------------
plt.rcdefaults()                                                       # hr_agreement.py sets its own style
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(REPO, "scripts", "hr_agreement.py"))   # also rewrites its outputs
plt.close("all")
style()
e, q, good = g["ecg_hr"], g["ppg_hr"], g["ppg_snr"] >= -5
r2 = np.corrcoef(e[good], q[good])[0, 1] ** 2
diff, mean = q[good] - e[good], (q[good] + e[good]) / 2
bias, sd = diff.mean(), diff.std(ddof=1)
fig, ax = graph("PPG against ECG heart rate, %d pieces of 30 s" % len(e), (4.0, 4.0))
ax.plot([200, 650], [200, 650], color=AXIS, linewidth=0.8, zorder=0)
ax.plot(e[~good], q[~good], "o", markerfacecolor="none", markeredgecolor=AXIS, markersize=3.5, markeredgewidth=0.7,
        label="weak PPG pulse (%d)" % (~good).sum())
ax.plot(e[good], q[good], "o", color=BLUE, alpha=0.6, markersize=3.5, markeredgewidth=0, label="clear PPG pulse (%d)" % good.sum())
ax.set_xlim(200, 650)
ax.set_ylim(200, 650)
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=7)
ax.set_xlabel("ECG heart rate (per min)")
ax.set_ylabel("PPG heart rate (per min)")
save(fig, "hr_ppg_vs_ecg")

fig, ax = graph("Bland-Altman, %d pieces with a clear PPG pulse" % good.sum(), (5.0, 3.2))
ax.axhspan(bias - 1.96 * sd, bias + 1.96 * sd, color=HAIRLINE, alpha=0.7, linewidth=0, zorder=0)
ax.axhline(bias, color=INK_2, linewidth=0.8)
ax.plot(mean, diff, "o", color=BLUE, alpha=0.6, markersize=3.5, markeredgewidth=0)
ax.text(0.02, 0.98, "mean %.1f, limits of agreement %.1f to %.1f (grey)" % (bias, bias - 1.96 * sd, bias + 1.96 * sd),
        transform=ax.transAxes, va="top", fontsize=7, color=INK_2)
lim = np.abs(diff).max() + 5
ax.set_ylim(-lim, lim)
ax.set_xlabel("Mean of ECG and PPG (per min)")
ax.set_ylabel("PPG - ECG (per min)")
save(fig, "bland_altman")
print("agreement: %d of %d pieces with PPG SNR >= -5 dB, r2 %.3f, bias %.2f, limits %.1f to %.1f" % (
    good.sum(), len(e), r2, bias, bias - 1.96 * sd, bias + 1.96 * sd))

# skin temperature ------------------------------------------------------------------------------
o = np.argsort(d["t_temp"])
tt, temp = d["t_temp"][o] / 60, d["temp"][o]
smooth = uniform_filter1d(temp, 10)
fig, ax = graph("Skin temperature during the 26 Oct session (Mouse 1)")
ax.plot(tt, temp, color=BLUE, linewidth=0.5, alpha=0.3)
ax.plot(tt, smooth, color=BLUE, linewidth=1.2)
dose_bands(ax, v)
ax.set_ylim(None, smooth.max() + 0.5)
ax.set_ylabel("Skin temperature (°C)")
save(fig, "skin_temperature")
print("temperature: %.1f to %.1f °C in %.0f min" % (smooth[0], smooth[-1], tt[-1]))
