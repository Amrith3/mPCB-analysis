"""Preprocessing: the filters applied to the ECG and the PPG, and the tools to show what a filter does.

Every filter here runs forwards and then backwards over the signal (scipy filtfilt or sosfiltfilt).
This makes it zero-phase: no wave is shifted in time, which matters when beats are averaged or
timed. Because the signal passes the filter twice, the gain at a cut-off frequency is -6 dB
(half the amplitude) instead of the usual -3 dB.

The ECG chip filters before anything is stored: a 0.5 Hz high-pass and a 153.6 Hz low-pass
(102.4 Hz before 24 Sep 2026). The analysis adds no low-pass to the ECG.
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

ECG_HP_HZ = 3                         # ECG high-pass cut-off
MAINS_HZ = [50, 100, 150, 200, 250]   # mains and its harmonics below 256 Hz
NOTCH_HZ = [50, 150]                  # the mains notches used (see notch_differences.py)
NOTCH_Q = 30                          # notch quality: width = frequency / 30
PPG_LP_HZ = 32                        # PPG low-pass cut-off
HEART_HZ = [4, 11]                    # PPG heart band, 240-660 per minute
BREATH_HZ = [0.15, 3.5]               # PPG breathing band, 9-210 per minute


def notch_frequencies(fs, wanted=NOTCH_HZ):
    """The frequencies in `wanted` that exist at sampling rate fs, i.e. are below fs / 2."""
    used = []
    for f0 in wanted:
        if f0 < fs / 2:
            used.append(f0)
    return used


def notch(x, fs, f0):
    """Remove one frequency f0 (Hz) from x.

    How: an IIR notch filter (scipy iirnotch) with quality factor NOTCH_Q, so the band it removes
    is f0 / 30 wide (1.7 Hz at 50 Hz, 5 Hz at 150 Hz). Run forwards and backwards.
    """
    b, a = iirnotch(f0, NOTCH_Q, fs=fs)
    return filtfilt(b, a, x)


def notches(x, fs, wanted=NOTCH_HZ):
    """Remove each frequency of `wanted` (default: the mains at 50 and 150 Hz) with notch()."""
    for f0 in notch_frequencies(fs, wanted):
        x = notch(x, fs, f0)
    return x


def mains_fitted_out(x, fs):
    """Remove the mains (50 Hz and its harmonics) without a notch filter.

    How: in each second of the signal, the sine and cosine waves of every mains frequency are
    fitted to the signal by least squares, and that fit is subtracted. Unlike a notch, this does
    not ring after the sharp QRS, so it is used where small waves are measured (the averaged beat).
    The last part second is fitted together with the second before it.
    """
    y = x.astype(float)
    n = int(fs)
    t = np.arange(n) / fs
    waves = []
    for f0 in notch_frequencies(fs, MAINS_HZ):
        waves.append(np.cos(2 * np.pi * f0 * t))
        waves.append(np.sin(2 * np.pi * f0 * t))
    waves = np.array(waves).T
    starts = list(range(0, len(x) - n + 1, n))
    if len(x) % n and len(x) > n:
        starts.append(len(x) - n)
    for start in starts:
        piece = x[start:start + n] - np.mean(x[start:start + n])
        sizes = np.linalg.lstsq(waves, piece, rcond=None)[0]
        y[start:start + n] = x[start:start + n] - waves @ sizes
    return y


def ecg_highpass(x, fs):
    """Remove baseline drift and most of the breathing from the ECG.

    How: 4th-order Butterworth high-pass at ECG_HP_HZ (3 Hz), forwards and backwards.
    """
    sos = butter(4, ECG_HP_HZ, "highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def ppg_lowpass(x, fs):
    """Remove noise above the pulse from the PPG.

    How: 4th-order Butterworth low-pass at PPG_LP_HZ (32 Hz), forwards and backwards. Skipped when
    32 Hz is not below half the sampling rate (PPG at 50 samples per second).
    """
    if PPG_LP_HZ >= fs / 2:
        return x
    sos = butter(4, PPG_LP_HZ, "lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def heart_band(x, fs):
    """Keep only the heart-rate range of the PPG (HEART_HZ, 240-660 beats per minute).

    How: 2nd-order Butterworth band-pass, forwards and backwards. In mice this band can also hold
    harmonics of the breathing.
    """
    sos = butter(2, HEART_HZ, "bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def breathing_band(x, fs):
    """Keep only the breathing range of the PPG (BREATH_HZ, 9-210 breaths per minute).

    How: 2nd-order Butterworth band-pass, forwards and backwards.
    """
    sos = butter(2, BREATH_HZ, "bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def click(fs):
    """A single 1 in the middle of 256 s of zeros. Filtering it gives the filter's impulse response."""
    x = np.zeros(int(256 * fs))
    x[len(x) // 2] = 1
    return x


def gain_db(answer, fs):
    """Frequencies and gain (dB) of a filter, from its impulse response (a filtered click).

    How: a click holds every frequency equally, so the Fourier transform of the filtered click is
    the filter's gain at each frequency.
    """
    f = np.fft.rfftfreq(len(answer), 1 / fs)
    return f, 20 * np.log10(np.abs(np.fft.rfft(answer)) + 1e-12)


def ecg_levels(raw, fs):
    """The ECG filtering steps, each adding one filter to the step before.

    Returns [name, signal, impulse response] for: raw, + 3 Hz high-pass, + each notch in NOTCH_HZ.
    The raw step has no impulse response (None).
    """
    c = click(fs)
    levels = [["raw ECG (as recorded)", raw, None]]
    x = ecg_highpass(raw, fs)
    levels.append(["+ " + str(ECG_HP_HZ) + " Hz high-pass", x, ecg_highpass(c, fs)])
    for f0 in notch_frequencies(fs):
        x = notch(x, fs, f0)
        levels.append(["+ " + str(f0) + " Hz notch", x, notch(c, fs, f0)])
    return levels
