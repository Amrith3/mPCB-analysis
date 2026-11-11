"""ECG filtering levels: raw, + 3 Hz high-pass, + 50 Hz notch, + 150 Hz notch."""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the main folder
from mpcb import filters, plotting  # noqa: E402
from mpcb.recording import Recording, output  # noqa: E402

rec = Recording()
fs = rec.fs_ecg
t = rec.t_ecg
levels = filters.ecg_levels(rec.ecg, fs)

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
        plotting.step1_amplitude(x, fs, "µV")
        plt.title("1. FFT of the raw ECG: amplitude (µV); red: mains")
        plt.subplot(n, 3, 3 * i + 3)
        plotting.power_spectrum(x, fs)
        plt.title("power spectrum of the raw ECG (dB); red: mains")
        for ax in plt.gcf().axes[-2:]:
            for f0 in filters.notch_frequencies(fs, filters.MAINS_HZ):
                ax.axvline(f0, color="tab:red", linewidth=0.8, linestyle=":")
    else:
        plotting.step2_gain(answer, fs)
        plt.subplot(n, 3, 3 * i + 3)
        plotting.step3_before_after(levels[i - 1][1], x, fs)

plt.savefig(output("ecg_filter_steps.png"), dpi=150)
plt.show()
