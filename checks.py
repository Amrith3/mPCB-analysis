"""The methods tried on made-up signals whose answer is known, the last one built from a real
recording's beat and noise. Each line prints what was found against the known value; an
AssertionError means a method no longer finds it."""
import contextlib
import io

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt

from mpcb import analysis, filters, sessions
from mpcb.recording import FILE, Recording

rng = np.random.default_rng(1)
fs = 512

# SNR: a sine of known power plus white noise of power 1, 200 beats, median of 50 tries
period = 60
samples = np.arange(200 * period)
for size, known in [(2, 3.01), (0.6, -7.45), (0.2, -16.99)]:
    beat = size * np.sin(2 * np.pi * np.arange(period) / period)
    found = np.median([analysis.snr_db(samples, np.tile(beat, 200) + rng.normal(0, 1, 200 * period),
                                      np.arange(period // 2, 200 * period, period, dtype=float), period)
                       for _ in range(50)])
    print("SNR: known %.2f dB, found %.2f dB" % (known, found))
    assert abs(found - known) < 0.2

# SNR of sharp beats (R 3 ms wide) at times between samples: each beat has to be read at its own time
t = np.arange(120 * fs) / fs
true = 0.2 + np.cumsum(0.150 + rng.normal(0, 0.0005, 790))
true = true[true < t[-1] - 0.2]
lowpass = butter(4, 150, "lowpass", fs=fs, output="sos")


def bumps(centres, width):
    """Sum of Gaussian bumps of the given width (s) at the given centres (s), sampled at t."""
    return np.exp(-0.5 * ((t[:, None] - centres) / width) ** 2).sum(axis=1)


clean = sosfiltfilt(lowpass, 450 * bumps(true, 0.0015) - 450 * bumps(true + 0.0045, 0.0015) + 350 * bumps(true + 0.009, 0.0022))
noise = sosfiltfilt(lowpass, rng.normal(0, 30, len(t)))
offsets = np.arange(-0.075, 0.075, 1 / fs)
known = 10 * np.log10(np.var(np.mean(CubicSpline(t, clean)(true[:, None] + offsets), axis=0)) / np.var(noise))
found = analysis.snr_db(t, clean + noise, true, 0.150)
print("SNR of sharp beats between samples: known %.2f dB, found %.2f dB" % (known, found))
assert abs(found - known) < 0.3

# HRV: a premature beat and the pause after it are left out
rr = 0.12 + rng.normal(0, 0.001, 300)
sdnn, rmssd = 1000 * np.std(rr, ddof=1), 1000 * np.sqrt(np.mean(np.diff(rr) ** 2))
odd = rr.copy()
odd[100], odd[101] = 0.08, 0.16
found = analysis.hrv(odd, fs)
print("HRV with a premature beat: SDNN %.3f (%.3f without it), RMSSD %.3f (%.3f) ms" % (found[0], sdnn, found[1], rmssd))
assert abs(found[0] / sdnn - 1) < 0.02 and abs(found[1] / rmssd - 1) < 0.02

# R waves and R times: beats with R, S and a J wave as tall as R; 40 % of the marks put on J
t = np.arange(60 * fs) / fs
true = 0.2 + np.cumsum(0.150 + rng.normal(0, 0.0005, 390))
ecg = 400 * bumps(true, 0.0025) - 350 * bumps(true + 0.004, 0.002) + 420 * bumps(true + 0.010, 0.003)
ecg = ecg + rng.normal(0, 15, len(t))
with contextlib.redirect_stdout(io.StringIO()):
    r = analysis.r_waves(ecg, fs)
print("R waves: %d found of %d" % (len(r), len(true)))
assert abs(len(r) - len(true)) <= 2
marks = np.round((true + 0.010 * (rng.random(len(true)) < 0.4)) * fs).astype(int)
found_t = analysis.r_times(ecg, t, fs, marks)
matched = ~np.isnan(found_t)
error = 1000 * (found_t[matched] - true[matched])
found = analysis.hrv(np.diff(found_t[matched]), fs)[1]
known = 1000 * np.sqrt(np.mean(np.diff(np.diff(true[matched])) ** 2))
print("R times: beat-to-beat error %.3f ms SD (marks off by 10 ms on 40 %%); RMSSD known %.3f, found %.3f ms"
      % (np.std(error), known, found))
assert np.std(error) < 0.1 and abs(found / known - 1) < 0.1

# HRV floor: a perfectly regular heart reads about 0 without noise, more with the noise of a recording
clean = 400 * bumps(true, 0.0025) - 350 * bumps(true + 0.004, 0.002) + 420 * bumps(true + 0.010, 0.003)
quiet_floor = analysis.hrv_floor(clean, t, fs, true)
noisy_floor = analysis.hrv_floor(clean + rng.normal(0, 40, len(t)), t, fs, true)
print("HRV floor: RMSSD %.3f ms without noise, %.3f ms with 40 µV of noise" % (quiet_floor[1], noisy_floor[1]))
assert quiet_floor[1] < 0.1 and noisy_floor[1] > quiet_floor[1]

# mains fit: 50 and 150 Hz removed to the last sample, a part second included
t3 = np.arange(int(3.4 * fs)) / fs
mains = 300 * np.sin(2 * np.pi * 50 * t3) + 40 * np.sin(2 * np.pi * 150 * t3 + 1)
left = filters.mains_fitted_out(mains, fs)[-int(0.4 * fs):]
print("mains fit: %.3f µV RMS left in the last 0.4 s (of %.0f)" % (np.std(left), np.std(mains)))
assert np.std(left) < 1

# breathing rate: 90 per minute, with a second hump in every breath
fp = 200
tp = np.arange(30 * fp) / fp
breath = np.sin(2 * np.pi * 1.5 * tp) + 0.6 * np.sin(2 * np.pi * 3 * tp + 0.5) + rng.normal(0, 0.2, len(tp))
found = analysis.breathing_rate(filters.breathing_band(breath, fp), fp)
print("breathing rate: known 90, found %.1f per minute" % found)
assert abs(found - 90) < 1

# wave boundaries: hann-shaped P (-55 to -40 ms), R (-5 to 5), S (5 to 11), J (11 to 21);
# with k = 0.15 a boundary lies 0.024 x the wave's length inside the wave's start or end
t_ms = 1000 * np.arange(-51, 30) / fs


def hann(start, end, height):
    """A Hann-shaped wave of the given height from start to end (ms), zero elsewhere."""
    inside = (t_ms > start) & (t_ms < end)
    return np.where(inside, height * 0.5 * (1 - np.cos(2 * np.pi * (t_ms - start) / (end - start))), 0)


beat = hann(-55, -40, 60) + hann(-5, 5, 500) + hann(5, 11, -300) + hann(11, 21, 350)
p = sessions.ecg_points(t_ms, beat)
for name, known in [("P on", -55 + 0.024 * 15), ("P off", -40 - 0.024 * 15), ("QRS on", -5 + 0.024 * 10),
                    ("QRS off", 12)]:
    print("%s: known %.2f ms, found %.2f ms" % (name, known, p[name]))
    assert abs(p[name] - known) < 0.7

# dose from the notes
for note, known in [("iso at 1", 1), ("iso increased from 1 to 2", 2), ("iso changed to 0 from 4", 0),
                    ("reducing to 1.5", 1.5), ("heat 2 min", None), ("gain to 80", None)]:
    assert sessions.dose_in(note) == known, note
print("dose notes: all read right")

# real noise: 2 min of 22 Oct 14:54:30; its averaged beat taken out at every R time and put back at
# known times (its own rhythm played backwards, or perfectly regular), with the noise made larger
with contextlib.redirect_stdout(io.StringIO()):
    rec = Recording(FILE)
    part = (rec.t_ecg >= 300) & (rec.t_ecg < 420)
    t, ecg = rec.t_ecg[part], rec.ecg[part]
    real = analysis.r_times(ecg, t, fs, analysis.r_waves(ecg, fs))
real, period, beats_at = analysis.beat_model(ecg, t, fs, real[~np.isnan(real)])


def found_at(x):
    """R-wave times (s) that r_waves and r_times find in x, without the ones they cannot match."""
    with contextlib.redirect_stdout(io.StringIO()):
        found = analysis.r_times(x, t, fs, analysis.r_waves(x, fs))
    return found[~np.isnan(found)]


noise = ecg - beats_at(real)
known = real[0] + 0.05 + np.concatenate([[0], np.cumsum(np.diff(real)[::-1])])
known = known[known < t[-1] - 0.1]
regular = np.arange(real[0] + 0.05, t[-1] - 0.1, period)
for scale in (1, 2, 3):
    made = scale * noise + beats_at(known)
    snr = analysis.snr_db(t, filters.notches(filters.ecg_highpass(made, fs), fs), known, period)
    found = found_at(made)
    j = np.clip(np.searchsorted(found, known), 1, len(found) - 1)
    hit = np.minimum(np.abs(found[j - 1] - known), np.abs(found[j] - known)) < 0.02
    rate = (60 / np.median(np.diff(known)), 60 / np.median(np.diff(found)))
    floor = analysis.hrv(np.diff(found_at(scale * noise + beats_at(regular))), fs)
    print("real noise x%d: SNR %.1f dB, beats found %.1f %%, extra %.1f %%, heart rate %.0f (known %.0f), "
          "a regular heart reads RMSSD %.2f, SDNN %.2f ms"
          % (scale, snr, 100 * hit.mean(), 100 * (len(found) - hit.sum()) / len(known), rate[1], rate[0], floor[1], floor[0]))
    assert abs(rate[1] / rate[0] - 1) < 0.01 and hit.mean() > (0.99 if scale == 1 else 0.9)
print("all checks passed")
