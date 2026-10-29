"""PPG (IR) filtering stages and vitals per 30 s window: heart rate, breathing rate, perfusion index."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, welch

import filters
import panels
from recording import load

WINDOW_S = 30

rec = load()
if "ir_nA" not in rec:
    raise SystemExit("No usable PPG in this recording.")
fs = rec["fs_ppg"]
t = rec["t_ppg"]
if t[-1] - t[0] < WINDOW_S:
    raise SystemExit("The PPG is shorter than one window of " + str(WINDOW_S) + " s.")
raw = rec["ir_nA"]
ppg = filters.ppg_lowpass(raw, fs)
heart = filters.heart_band(ppg, fs)
breathing = filters.breathing_band(ppg, fs)

r = filters.r_waves(rec["ecg"], rec["fs_ecg"])
if len(r) < 10:
    raise SystemExit("Fewer than 10 R waves in the ECG: the pulse cannot be lined up on heartbeats.")
r_times = rec["t_ecg"][r]
pulses = np.searchsorted(t, r_times)
pulse_half = int(0.15 * fs)

middle = []
ecg_rate = []
ppg_rate = []
breathing_per_minute = []
perfusion = []
breaths = []
for start in np.arange(t[0], t[-1] - WINDOW_S, WINDOW_S):
    inside = (t >= start) & (t < start + WINDOW_S)
    beats_inside = (r_times >= start) & (r_times < start + WINDOW_S)
    middle.append(start + WINDOW_S / 2)

    if np.sum(beats_inside) > 2:
        ecg_rate.append(60 / np.median(np.diff(r_times[beats_inside])))
    else:
        ecg_rate.append(np.nan)

    f, p = welch(heart[inside], fs=fs, nperseg=int(10 * fs), nfft=int(WINDOW_S * fs))
    band = (f >= filters.HEART_HZ[0]) & (f <= filters.HEART_HZ[1])
    f_band = f[band]
    p_band = p[band]
    ppg_rate.append(60 * f_band[np.argmax(p_band)])

    rate = filters.breathing_rate(breathing[inside], fs)
    breathing_per_minute.append(rate)
    if not np.isnan(rate):
        peaks, _ = find_peaks(breathing[inside], distance=int(0.6 * fs * 60 / rate))
        first = np.flatnonzero(inside)[0]
        breaths.extend(first + peaks)

    pulse = panels.averaged(heart, pulses[beats_inside], pulse_half)
    if pulse is None:
        perfusion.append(np.nan)
    else:
        perfusion.append(100 * np.ptp(pulse) / np.mean(ppg[inside]))

ecg_rate = np.array(ecg_rate)
ppg_rate = np.array(ppg_rate)
agree = np.abs(ppg_rate / ecg_rate - 1) <= 0.05
pulse = panels.averaged(heart, pulses, pulse_half)
print("IR light level:", round(np.mean(raw)), "nA")
print("heart rate (median of the windows): ECG", round(np.nanmedian(ecg_rate)), "per minute, PPG",
      round(np.median(ppg_rate)), "per minute; the PPG is within 5 % of the ECG in", np.sum(agree), "of",
      len(agree), "windows")
breathing_per_minute = np.array(breathing_per_minute)
clear = ~np.isnan(breathing_per_minute)
if np.any(clear):
    typical = np.median(breathing_per_minute[clear])
    print("breathing rate (median of the windows with clear breathing):", round(typical), "per minute; clear in",
          np.sum(clear), "of", len(clear), "windows;", len(breaths), "breaths")
else:
    typical = 60
    print("breathing rate: no clear breathing in any window")
print("perfusion index (size of the averaged pulse, % of the light level):", round(100 * np.ptp(pulse) / np.mean(ppg), 3), "%")

breath_half = int(0.6 * fs * 60 / typical)
c = filters.click(fs)
# name, signal, signal before this filter, impulse response, centres, half width
rows = [["raw IR light", raw, None, None, pulses, pulse_half],
        ["IR after the " + str(filters.PPG_LP_HZ) + " Hz low-pass", ppg, raw, filters.ppg_lowpass(c, fs), pulses, pulse_half],
        ["IR heart band " + str(filters.HEART_HZ) + " Hz", heart, ppg, filters.heart_band(c, fs), pulses, pulse_half],
        ["IR breathing band " + str(filters.BREATH_HZ) + " Hz", breathing, ppg, filters.breathing_band(c, fs), breaths, breath_half]]
averaged_name = ["averaged pulse (on the R waves)", "averaged pulse (on the R waves)",
                 "averaged pulse (on the R waves)", "averaged breath"]

show = (t >= 10) & (t < 15)
plt.rcParams["font.size"] = 8
plt.figure(figsize=(16, 14), layout="constrained")
for i in range(4):
    name, x, before, answer, centres, half = rows[i]

    plt.subplot(5, 4, 4 * i + 1)
    plt.plot(t[show], x[show])
    plt.title(name)
    plt.xlabel("time (s)")
    plt.ylabel("nA")

    plt.subplot(5, 4, 4 * i + 2)
    if answer is None:
        panels.step1_amplitude(x, fs, "nA")
    else:
        panels.step2_gain(answer, fs)
        plt.subplot(5, 4, 4 * i + 3)
        panels.step3_before_after(before, x, fs)

    plt.subplot(5, 4, 4 * i + 4)
    average = panels.averaged(x, centres, half)
    if average is not None:
        plt.plot(1000 * np.arange(-half, half) / fs, average - np.median(average))
    plt.title(averaged_name[i])
    plt.xlabel("time from peak (ms)")

plt.subplot(5, 4, (17, 18))
plt.plot(middle, ecg_rate, "k-", label="ECG")
plt.plot(middle, ppg_rate, "o", label="PPG heart band")
plt.title("heart rate")
plt.xlabel("time (s)")
plt.ylabel("per minute")
plt.legend(fontsize=7)
plt.subplot(5, 4, 19)
plt.plot(middle, breathing_per_minute, "o-")
plt.title("breathing rate")
plt.xlabel("time (s)")
plt.ylabel("per minute")
plt.subplot(5, 4, 20)
plt.plot(middle, perfusion, "o-")
plt.title("perfusion index (averaged pulse)")
plt.xlabel("time (s)")
plt.ylabel("% of the light level")

plt.savefig("ppg_vitals.png", dpi=150)
plt.show()
