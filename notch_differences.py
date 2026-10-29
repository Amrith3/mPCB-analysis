"""Each mains notch tried on its own: the mains and the ECG it removes.

The ECG removed is measured on the ECG with the mains fitted out (filters.mains_fitted_out),
which itself costs about 1-2 % of the QRS.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import filters
import panels
from recording import load

rec = load()
fs = rec["fs_ecg"]
t = rec["t_ecg"]
recorded = filters.ecg_highpass(rec["ecg"], fs)
clean = filters.ecg_highpass(filters.mains_fitted_out(rec["ecg"], fs), fs)
r = filters.r_waves(rec["ecg"], fs)
if len(r) < 10:
    raise SystemExit("Fewer than 10 R waves found: the averaged beats cannot be made.")

half = int(0.1 * fs)
t_ms = 1000 * np.arange(-half, half) / fs
qrs_part = (t_ms >= -15) & (t_ms <= 25)
tried = filters.notch_frequencies(fs, filters.MAINS_HZ)


def qrs(beat):
    return np.max(beat[qrs_part]) - np.min(beat[qrs_part])


def spectrum_db(x):
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(8 * fs)))
    return f, 10 * np.log10(p + 1e-20)


beat_before = panels.averaged(clean, r, half)
start = max(t[0], min(10, t[-1] - 0.5))
show = (t >= start) & (t < start + 0.5)
n = len(tried)
plt.rcParams["font.size"] = 8
plt.figure(figsize=(19, 3.5 * n), layout="constrained")
for i in range(n):
    f0 = tried[i]
    after = filters.notch(clean, fs, f0)
    removed = clean - after
    mains = recorded - filters.notch(recorded, fs, f0) - removed
    beat_after = panels.averaged(after, r, half)
    change = 100 * (qrs(beat_after) / qrs(beat_before) - 1)
    if f0 in filters.NOTCH_HZ:
        title = str(f0) + " Hz notch (used)"
    else:
        title = str(f0) + " Hz notch (not used)"
    print(title + ": removes", round(np.std(mains), 1), "µV RMS of mains and", round(np.std(removed), 1),
          "µV RMS of ECG; QRS", round(change, 1), "%")

    plt.subplot(n, 4, 4 * i + 1)
    plt.plot(t[show], after[show], label="ECG after the notch")
    plt.plot(t[show], removed[show], color="tab:red", label="ECG the notch removed")
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("µV")
    plt.legend(fontsize=7)

    plt.subplot(n, 4, 4 * i + 2)
    plt.plot(t[show], removed[show], color="tab:red")
    plt.title("the removed ECG, enlarged (" + str(round(np.std(removed), 1)) + " µV RMS)")
    plt.xlabel("time (s)")
    plt.ylabel("µV")

    plt.subplot(n, 4, 4 * i + 3)
    plt.plot(t_ms, beat_before - np.median(beat_before), color="0.6", label="before")
    plt.plot(t_ms, beat_after - np.median(beat_after), label="after")
    plt.plot(t_ms, beat_before - beat_after, color="tab:red", label="removed")
    plt.title("averaged beat, QRS " + str(round(change, 1)) + " %")
    plt.xlabel("time from R (ms)")
    plt.ylabel("µV")
    plt.legend(fontsize=7)

    plt.subplot(n, 4, 4 * i + 4)
    f, p_before = spectrum_db(recorded)
    f, p_after = spectrum_db(filters.notch(recorded, fs, f0))
    near = (f >= f0 - 10) & (f <= f0 + 10)
    plt.plot(f[near], p_before[near], color="0.6", label="before")
    plt.plot(f[near], p_after[near], label="after")
    plt.title("near " + str(f0) + " Hz: " + str(round(np.std(mains), 1)) + " µV RMS of mains")
    plt.xlim(f0 - 10, f0 + 10)
    plt.locator_params(axis="x", nbins=5)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("dB")
    plt.legend(fontsize=7)

used = filters.notches(clean, fs)
beat_used = panels.averaged(used, r, half)
print("the notches used together", filters.notch_frequencies(fs), "Hz: QRS", round(100 * (qrs(beat_used) / qrs(beat_before) - 1), 1), "%")

plt.savefig("notch_differences.png", dpi=150)
plt.show()
