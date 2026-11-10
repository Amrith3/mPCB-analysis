"""Analysis: finding beats and breaths in the filtered signals and measuring from them.

R waves and their exact times, heart rate, breathing rate (per window and breath by breath),
averaged beats and pulses, signal-to-noise ratio and heart-rate variability.
"""
import contextlib
import io

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt

from .filters import BREATH_HZ, MAINS_HZ, ecg_highpass, notches


def r_waves(ecg, fs, pieces_out=None):
    """Sample numbers of the R waves in an ECG (µV, sampled at fs).

    How:
    1. All mains frequencies are notched out and the ECG is band-passed to 20-90 Hz, where most of
       the sharp R wave lies.
    2. The ECG is cut into 30 s pieces, and the R waves of each piece are found by beats_in(),
       because the heart rate can change by half within one recording.
    3. Each mark is moved to the R peak itself: within 20 ms before to 5 ms after the mark, the
       earliest peak at least 0.6 x the tallest one. This keeps a J wave that is taller than R
       (common under isoflurane) from being taken for it.

    It prints in how many 30 s pieces the beat repeats clearly; with few or none the heart rate is
    probably noise. If a list is given as pieces_out, (first sample, last sample + 1, beat repeat
    clear) is added to it for each piece.
    """
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
        if pieces_out is not None:
            pieces_out.append((piece[0], piece[-1] + 1, repeat))
    r = np.array(r, dtype=int)
    print("R waves: the beat repeats clearly in", clear, "of", pieces, "pieces of 30 s"
          " (few or none: the heart rate may be noise)")

    level = ecg_highpass(x, fs)              # the 0.6 is of the height above the baseline
    before = int(0.020 * fs)
    after = int(0.005 * fs)
    r = r[(r > before) & (r < len(ecg) - after - 1)]
    for i in range(len(r)):
        start = r[i] - before
        seg = level[start:r[i] + after + 1]
        tops, _ = find_peaks(seg)
        tops = tops[seg[tops] >= 0.6 * np.max(seg)]
        r[i] = start + (tops[0] if len(tops) else np.argmax(seg))
    return r


def beats_in(band, fs):
    """R waves in one piece of the 20-90 Hz band, and whether the beat interval was clear.

    How: the band's energy (squared, top 1 % cut off, smoothed below 15 Hz) is compared with
    itself shifted by every delay from 80 to 300 ms (autocorrelation; 750 to 200 beats per minute).
    The first delay at which it looks at least 0.8 times as alike as the best one is one beat; it is
    clear if the best likeness is at least half that at delay 0. After each R wave the search then
    waits 0.6 x that beat (80 ms if not clear), so a T wave is not counted as a second beat. R waves
    are the peaks taller than 0.3 x the typical tall peak. Fewer than 10 peaks: none returned.
    """
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
        good = tops[part[tops] >= 0.8 * np.max(part[tops])]   # not two beats taken for one
        beat = shortest + good[0]
        wait = int(0.6 * beat)
    else:
        wait = shortest

    peaks, _ = find_peaks(band, distance=wait)
    if len(peaks) < 10:
        return np.array([], dtype=int), repeat
    heights = band[peaks]
    tall = heights[heights > np.percentile(heights, 80)]
    r, _ = find_peaks(band, height=0.3 * np.median(tall), distance=wait)
    return r, repeat


def r_times(ecg, t, fs, r, shift_ms=12):
    """R-wave times in seconds, between samples (nan where a beat cannot be matched).

    How: each beat's QRS (20 ms before to 30 ms after its mark) is lined up with the averaged beat
    of its 30 s by shifting it (at most shift_ms) to where the two match best. The time is then
    the averaged beat's R peak, read between samples with a parabola through the three highest
    points. A mark on the J wave instead of R still gives the right time, and the times are finer
    than the 2 ms between samples, which beat-to-beat variability needs.
    """
    x = ecg_highpass(notches(ecg, fs, MAINS_HZ), fs)
    m = round(shift_ms * fs / 1000)
    lo, hi = round(-0.020 * fs), round(0.030 * fs)
    n = hi - lo
    times = np.full(len(r), np.nan)
    piece = r // int(30 * fs)
    room = (r + lo - m - 1 >= 0) & (r + hi + m + 1 <= len(x))
    for p in np.unique(piece[room]):
        k = np.flatnonzero(room & (piece == p))
        if len(k) < 10:
            continue
        wide = x[r[k, None] + np.arange(lo - m - 1, hi + m + 1)]   # each beat with room to shift
        rows = np.arange(len(k))[:, None]
        shift = np.full(len(k), m + 1)                              # column m + 1 = not shifted
        for _ in range(3):
            average = np.median(wide[rows, shift[:, None] + np.arange(n)], axis=0)
            match = np.array([wide[:, j:j + n] @ average for j in range(2 * m + 3)]).T
            shift = 1 + np.argmax(match[:, 1:-1], axis=1)
        between = top_between(*[match[rows[:, 0], shift + d] for d in (-1, 0, 1)])
        search = average[:-lo + round(0.005 * fs) + 1]              # -20 to 5 ms, as r_waves
        tops, _ = find_peaks(search)
        tops = tops[search[tops] >= 0.6 * np.max(search)]
        q = tops[0] if len(tops) else int(np.argmax(search))
        q = q + top_between(*average[q - 1:q + 2]) if 0 < q < n - 1 else q
        times[k] = t[r[k]] + (shift - m - 1 + between + lo + q) / fs
    return times


def top_between(a, b, c):
    """Where the top of a parabola through three equally spaced points lies, from -0.5 to 0.5
    samples around the middle one (0 if the points do not form a peak)."""
    curve = a - 2 * b + c
    return np.clip(np.where(curve < 0, 0.5 * (a - c) / np.where(curve < 0, curve, -1), 0), -0.5, 0.5)


def window_heart_rate(rr):
    """Heart rate (beats per minute) from the R-R intervals (s) of one window, or nan.

    How: intervals outside 80-300 ms (200-750 per minute) are dropped; with fewer than 5 left the
    window gets no value. Intervals more than 30 % from their median (a missed or an extra beat)
    are dropped too, and the rate is 60 / mean interval.
    """
    beats = rr[(rr >= 0.08) & (rr <= 0.30)]
    if len(beats) < 5:
        return np.nan
    beats = beats[np.abs(beats / np.median(beats) - 1) < 0.3]
    return 60 / np.mean(beats)


def breathing_rate(x, fs):
    """Breaths per minute in a piece of the PPG breathing band, or nan if breathing is not clear.

    How: the signal is compared with itself shifted by every delay (autocorrelation, averaged over
    the part that overlaps). It looks most alike when shifted by exactly one breath. Delays outside
    the breathing band are ignored. One breath is the shortest delay at which it looks at least 0.8
    times as alike as the best one (shifted by two breaths it looks alike too), read between samples
    with a parabola. No value if the best likeness is below half that at delay 0.
    """
    x = x - np.mean(x)
    n = len(x)
    alike = np.correlate(x, x, mode="full")[n - 1:]
    alike = alike / (n - np.arange(n))
    alike = alike / alike[0]
    shortest = int(fs / BREATH_HZ[1])
    longest = int(fs / BREATH_HZ[0])
    part = alike[shortest:longest]
    tops, _ = find_peaks(part)
    if len(tops) == 0 or np.max(part[tops]) < 0.5:
        return np.nan
    good = tops[part[tops] >= 0.8 * np.max(part[tops])]
    k = shortest + good[0]
    a, b, c = alike[k - 1], alike[k], alike[k + 1]
    curve = a - 2 * b + c
    delay = k + 0.5 * (a - c) / curve if curve != 0 else k
    return 60 * fs / delay


def breaths(band, t, fs):
    """Single breaths in the PPG breathing band: their times (s) and rates (breaths per minute).

    How: each breath is a peak of the breathing band. A peak counts if it stands out by at least
    0.4 x the typical size of the band (90th minus 10th percentile over 30 s) and lies at least
    0.25 x the local size above the median; peaks are at least 0.2 s apart (300 per minute). A breath
    can show as a peak or as a dip in the light, so both are tried and the one with the more regular
    intervals is used. The rate of a breath is 60 / the time since the one before.

    Unlike breathing_rate(), this gives a value even when breathing is irregular (single deep
    breaths with pauses at deep anaesthesia). A missed breath shows as one rate at about half.
    """
    best = None
    for sign in (1, -1):
        y = sign * band
        step = int(fs)
        half = int(30 * fs) // 2
        size = np.array([np.subtract(*np.percentile(y[max(0, i - half):i + half], [90, 10]))
                         for i in range(0, len(y), step)])
        size = np.repeat(size, step)[:len(y)]
        peaks, _ = find_peaks(y, distance=int(0.2 * fs), prominence=0.4 * np.median(size))
        peaks = peaks[y[peaks] - np.median(y) > 0.25 * size[peaks]]
        spread = np.std(np.diff(peaks))
        if best is None or spread < best[0]:
            best = (spread, peaks)
    peaks = best[1]
    return t[peaks][1:], 60 / np.diff(t[peaks])


def averaged(x, centres, half):
    """Median of x over the stretches of half samples before to half after each centre (sample
    number), or None if fewer than 3 stretches fit inside x. Used for the averaged beat and pulse:
    what happens at the same moment of every beat stays, noise averages away."""
    pieces = []
    for k in centres:
        if k - half >= 0 and k + half <= len(x):
            pieces.append(x[k - half:k + half])
    if len(pieces) < 3:
        return None
    return np.median(pieces, axis=0)


def hrv(rr, fs):
    """Heart-rate variability: SDNN and RMSSD (ms) of R-R intervals (s, in order).

    How: only normal intervals count, those within 5 robust SDs of the median (robust SD = 1.4826 x
    the median distance from the median), but at least 2 samples and at most 20 % from it; premature
    beats, pauses and missed or extra R waves are not normal. SDNN is the standard deviation of the
    normal intervals, RMSSD the root mean square of the differences between successive normal ones.
    nan with fewer than 20 normal intervals.
    """
    middle = np.median(rr)
    spread = 1.4826 * np.median(np.abs(rr - middle))
    normal = np.abs(rr - middle) <= np.clip(5 * spread, 2 / fs, 0.2 * middle)
    if np.sum(normal) < 20:
        return np.nan, np.nan
    both = normal[1:] & normal[:-1]
    return 1000 * np.std(rr[normal], ddof=1), 1000 * np.sqrt(np.mean(np.diff(rr)[both] ** 2))


def snr_db(t, x, times, period):
    """Signal-to-noise ratio (dB) of a signal x (sampled at times t) that repeats with every beat.

    How: one beat interval (period, s) around each beat time is read from x between samples (cubic
    spline), so every beat is lined up on its own time. Their average is the signal; what each beat
    differs from the average is the noise. The average still holds 1/n of the noise of its n beats,
    which is taken off its power. SNR = 10 log10(signal power / noise power). nan with fewer than
    10 beats or no signal left.
    """
    step = np.median(np.diff(t))
    offsets = np.arange(-period / 2, period / 2, step)
    times = np.sort(times[(times + offsets[0] >= t[0]) & (times + offsets[-1] <= t[-1])])
    n = len(times)
    if n < 10:
        return np.nan
    part = slice(max(np.searchsorted(t, times[0] + offsets[0]) - 3, 0), np.searchsorted(t, times[-1] + offsets[-1]) + 4)
    beats = CubicSpline(t[part], x[part])(times[:, None] + offsets)
    average = np.mean(beats, axis=0)
    noise = np.mean((beats - average) ** 2) * n / (n - 1)
    signal = np.var(average) - noise / n
    return 10 * np.log10(signal / noise) if signal > 0 else np.nan


def hrv_floor(ecg, t, fs, r_t):
    """SDNN and RMSSD (ms) that a perfectly regular heart would read with this ECG's noise.

    How: the averaged beat is taken out at every R time and put back at regular times, and the R
    waves are found again. HRV values close to this floor are mostly measurement noise.
    nan with fewer than 20 beats.
    """
    r_t = np.sort(r_t[~np.isnan(r_t)])
    if len(r_t) < 20:
        return np.nan, np.nan
    period = np.median(np.diff(r_t))
    offsets = np.arange(-0.45 * period, 0.55 * period, 1 / fs)
    r_t = r_t[(r_t + offsets[0] > t[0]) & (r_t + offsets[-1] < t[-1])]
    average = np.median(CubicSpline(t, ecg)(r_t[:, None] + offsets), axis=0)
    average = average - np.linspace(average[0], average[-1], len(average))
    edge = np.clip(np.minimum(offsets - offsets[0], offsets[-1] - offsets) / 0.005, 0, 1)
    shape = CubicSpline(offsets, average * 0.5 * (1 - np.cos(np.pi * edge)))

    def beats(times):
        out = np.zeros(len(t))
        for c in times:
            k = slice(np.searchsorted(t, c + offsets[0]), np.searchsorted(t, c + offsets[-1]))
            out[k] += shape(t[k] - c)
        return out

    made = ecg - beats(r_t) + beats(np.arange(r_t[0], r_t[-1], period))
    with contextlib.redirect_stdout(io.StringIO()):
        found = r_times(made, t, fs, r_waves(made, fs))
    return hrv(np.diff(found[~np.isnan(found)]), fs)
