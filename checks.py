"""The methods tried on made-up signals whose answer is known. Each line prints what was found
against the known value; an AssertionError means a method no longer finds it."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "isoflurane"))
import filters  # noqa: E402
import sessions  # noqa: E402

rng = np.random.default_rng(1)
fs = 512

# SNR: a sine of known power plus white noise of power 1, 200 beats, median of 50 tries
period = 60
for size, known in [(2, 3.01), (0.6, -7.45), (0.2, -16.99)]:
    beat = size * np.sin(2 * np.pi * np.arange(period) / period)
    found = np.median([filters.snr_db(np.tile(beat, 200) + rng.normal(0, 1, 200 * period),
                                      np.arange(period // 2, 200 * period, period), period) for _ in range(50)])
    print("SNR: known %.2f dB, found %.2f dB" % (known, found))
    assert abs(found - known) < 0.2

# HRV: a premature beat and the pause after it are left out
rr = 0.12 + rng.normal(0, 0.001, 300)
sdnn, rmssd = 1000 * np.std(rr, ddof=1), 1000 * np.sqrt(np.mean(np.diff(rr) ** 2))
odd = rr.copy()
odd[100], odd[101] = 0.08, 0.16
found = filters.hrv(odd, fs)
print("HRV with a premature beat: SDNN %.3f (%.3f without it), RMSSD %.3f (%.3f) ms" % (found[0], sdnn, found[1], rmssd))
assert abs(found[0] / sdnn - 1) < 0.02 and abs(found[1] / rmssd - 1) < 0.02

# R waves and R times: beats with R, S and a J wave as tall as R; 40 % of the marks put on J
t = np.arange(60 * fs) / fs
true = 0.2 + np.cumsum(0.150 + rng.normal(0, 0.0005, 390))


def bumps(centres, width):
    return np.exp(-0.5 * ((t[:, None] - centres) / width) ** 2).sum(axis=1)


ecg = 400 * bumps(true, 0.0025) - 350 * bumps(true + 0.004, 0.002) + 420 * bumps(true + 0.010, 0.003)
ecg = ecg + rng.normal(0, 15, len(t))
with open(os.devnull, "w") as quiet:
    stdout, sys.stdout = sys.stdout, quiet
    r = filters.r_waves(ecg, fs)
    sys.stdout = stdout
print("R waves: %d found of %d" % (len(r), len(true)))
assert abs(len(r) - len(true)) <= 2
marks = np.round((true + 0.010 * (rng.random(len(true)) < 0.4)) * fs).astype(int)
found_t = filters.r_times(ecg, t, fs, marks)
matched = ~np.isnan(found_t)
error = 1000 * (found_t[matched] - true[matched])
found = filters.hrv(np.diff(found_t[matched]), fs)[1]
known = 1000 * np.sqrt(np.mean(np.diff(np.diff(true[matched])) ** 2))
print("R times: beat-to-beat error %.3f ms SD (marks off by 10 ms on 40 %%); RMSSD known %.3f, found %.3f ms"
      % (np.std(error), known, found))
assert np.std(error) < 0.1 and abs(found / known - 1) < 0.1

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
found = filters.breathing_rate(filters.breathing_band(breath, fp), fp)
print("breathing rate: known 90, found %.1f per minute" % found)
assert abs(found - 90) < 1

# wave boundaries: hann-shaped P (-55 to -40 ms), R (-5 to 5), S (5 to 11), J (11 to 21);
# with k = 0.15 a boundary lies 0.024 x the wave's length inside the wave's start or end
t_ms = 1000 * np.arange(-51, 30) / fs


def hann(start, end, height):
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
print("all checks passed")
