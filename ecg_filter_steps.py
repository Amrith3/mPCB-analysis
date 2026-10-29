"""ECG filtering levels: raw, + 3 Hz high-pass, + 50 Hz notch, + 150 Hz notch."""
import matplotlib.pyplot as plt

import filters
import panels
from recording import load

rec = load()
fs = rec["fs_ecg"]
t = rec["t_ecg"]
levels = filters.ecg_levels(rec["ecg"], fs)

start = max(t[0], min(10, t[-1] - 2))
show = (t >= start) & (t < start + 2)
n = len(levels)
plt.rcParams["font.size"] = 8
plt.figure(figsize=(13, 3.3 * n), layout="constrained")
for i in range(n):
    name, x, answer = levels[i]

    plt.subplot(n, 3, 3 * i + 1)
    plt.plot(t[show], x[show])
    plt.title(name)
    plt.xlabel("time (s)")
    plt.ylabel("µV")

    plt.subplot(n, 3, 3 * i + 2)
    if answer is None:
        panels.step1_amplitude(x, fs, "µV")
        for f0 in filters.notch_frequencies(fs, filters.MAINS_HZ):
            plt.axvline(f0, color="tab:red", linewidth=0.8, linestyle=":")
        plt.title("1. amplitude (µV); red: mains")
    else:
        panels.step2_gain(answer, fs)
        plt.subplot(n, 3, 3 * i + 3)
        panels.step3_before_after(levels[i - 1][1], x, fs)

plt.savefig("ecg_filter_steps.png", dpi=150)
plt.show()
