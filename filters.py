"""Filters, ECG filtering levels, filter gain and R-wave detection.

All filters are zero-phase (forwards and backwards), so -6 dB at a cut-off.
The chip already applies a 0.5 Hz high-pass and a 153.6 Hz low-pass (102.4 Hz before 24 Sep 2026).
"""
import numpy as np
from scipy.signal import butter, correlate, filtfilt, find_peaks, iirnotch, sosfiltfilt

ECG_HP_HZ = 3
MAINS_HZ = [50, 100, 150, 200, 250]   # mains and harmonics below 256 Hz
NOTCH_HZ = [50, 150]                  # see notch_differences.py
NOTCH_Q = 30
PPG_LP_HZ = 32
HEART_HZ = [4, 11]                    # 240-660 per minute
BREATH_HZ = [0.15, 3.5]               # 9-210 per minute


def notch_frequencies(fs, wanted=NOTCH_HZ):
    """The frequencies in `wanted` below fs / 2."""
    used = []
    for f0 in wanted:
        if f0 < fs / 2:
            used.append(f0)
    return used


def notch(x, fs, f0):
    b, a = iirnotch(f0, NOTCH_Q, fs=fs)
    return filtfilt(b, a, x)


def notches(x, fs, wanted=NOTCH_HZ):
    for f0 in notch_frequencies(fs, wanted):
        x = notch(x, fs, f0)
    return x


def mains_fitted_out(x, fs):
    """Subtracts a least-squares fit of the mains frequencies, one second at a time."""
    y = x.astype(float)
    n = int(fs)
    t = np.arange(n) / fs
    waves = []
    for f0 in notch_frequencies(fs, MAINS_HZ):
        waves.append(np.cos(2 * np.pi * f0 * t))
        waves.append(np.sin(2 * np.pi * f0 * t))
    waves = np.array(waves).T
    for start in range(0, len(x) - n + 1, n):
        piece = x[start:start + n] - np.mean(x[start:start + n])
        sizes = np.linalg.lstsq(waves, piece, rcond=None)[0]
        y[start:start + n] = x[start:start + n] - waves @ sizes
    return y


def ecg_highpass(x, fs):
    sos = butter(4, ECG_HP_HZ, "highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def ppg_lowpass(x, fs):
    if PPG_LP_HZ >= fs / 2:
        return x
    sos = butter(4, PPG_LP_HZ, "lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def heart_band(x, fs):
    sos = butter(2, HEART_HZ, "bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def breathing_band(x, fs):
    sos = butter(2, BREATH_HZ, "bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def click(fs):
    """An impulse in 256 s of zeros."""
    x = np.zeros(int(256 * fs))
    x[len(x) // 2] = 1
    return x


def gain_db(answer, fs):
    """Gain in dB from a filter's impulse response."""
    f = np.fft.rfftfreq(len(answer), 1 / fs)
    return f, 20 * np.log10(np.abs(np.fft.rfft(answer)) + 1e-12)


def ecg_levels(raw, fs):
    """[name, signal, impulse response] for raw, + high-pass, + each notch."""
    c = click(fs)
    levels = [["raw ECG (as recorded)", raw, None]]
    x = ecg_highpass(raw, fs)
    levels.append(["+ " + str(ECG_HP_HZ) + " Hz high-pass", x, ecg_highpass(c, fs)])
    for f0 in notch_frequencies(fs):
        x = notch(x, fs, f0)
        levels.append(["+ " + str(f0) + " Hz notch", x, notch(c, fs, f0)])
    return levels


def r_waves(ecg, fs):
    """R-wave sample numbers, found in the 20-90 Hz band per 30 s piece."""
    x = notches(ecg, fs, MAINS_HZ)
    sos = butter(2, [20, min(90, fs / 2 - 10)], "bandpass", fs=fs, output="sos")
    band = sosfiltfilt(sos, x)

    r = []
    clear = 0
    pieces = max(1, round(len(band) / (30 * fs)))
    for piece in np.array_split(np.arange(len(band)), pieces):
        beats, repeat = beats_in(band[piece], fs)
        r.extend(piece[0] + beats)
        clear += repeat
    r = np.array(r, dtype=int)
    print("R waves: the beat repeats clearly in", clear, "of", pieces, "pieces of 30 s"
          " (few or none: the heart rate may be noise)")

    # move each mark to the ECG maximum from 20 ms before to 5 ms after
    before = int(0.020 * fs)
    after = int(0.005 * fs)
    r = r[(r > before) & (r < len(ecg) - after - 1)]
    for i in range(len(r)):
        start = r[i] - before
        r[i] = start + np.argmax(x[start:r[i] + after + 1])
    return r


def beats_in(band, fs):
    """R waves in one piece of the band, and whether the beat interval was clear."""
    energy = band ** 2
    energy = np.minimum(energy, np.percentile(energy, 99))
    energy = sosfiltfilt(butter(2, 15, "lowpass", fs=fs, output="sos"), energy)
    energy = energy - np.mean(energy)
    alike = correlate(energy, energy, mode="full", method="fft")[len(energy) - 1:]
    alike = alike / alike[0]
    shortest = int(0.08 * fs)   # 750 per minute
    longest = int(0.30 * fs)    # 200 per minute
    part = alike[shortest:longest]
    tops, _ = find_peaks(part)
    repeat = len(tops) > 0 and np.max(part[tops]) >= 0.5
    if repeat:
        # shortest peak close to the best one, so two beats are not taken for one
        good = tops[part[tops] >= 0.8 * np.max(part[tops])]
        beat = shortest + good[0]
        wait = int(0.6 * beat)   # skips the T wave
    else:
        wait = shortest

    peaks, _ = find_peaks(band, distance=wait)
    if len(peaks) < 10:
        return np.array([], dtype=int), repeat
    heights = band[peaks]
    tall = heights[heights > np.percentile(heights, 80)]
    r, _ = find_peaks(band, height=0.3 * np.median(tall), distance=wait)
    return r, repeat
