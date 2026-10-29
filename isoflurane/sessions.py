"""Isoflurane sessions: all recordings of one session on one clock, the dose from the notes,
the vitals every 10 s, and the averaged beat at each dose."""
import contextlib
import glob
import io
import os
import re
import sys
from datetime import datetime

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # filters, panels, recording
import filters  # noqa: E402
import panels  # noqa: E402
from recording import DATA, load, unzip  # noqa: E402

# name: (zip in data/, folder inside it)
SESSIONS = {
    "8 Oct": ("mouse_isoflurane_20261008", ""),
    "20 Oct": ("mouse_isoflurane_20261020", ""),
    "22 Oct, 30 g": ("mouse_isoflurane_20261022", "mouse_30g"),
    "22 Oct, 25 g": ("mouse_isoflurane_20261022", "mouse_25g"),
    "26 Oct, 30 g": ("mouse_isoflurane_20261026", "mouse_30g"),
}
STEP_S = 10
PPG_WINDOW_S = 30


def files(name):
    """The session's CSV files in time order; unzips data/<zip>.zip the first time."""
    folder, sub = SESSIONS[name]
    unzip(folder)
    return sorted(glob.glob(os.path.join(DATA, folder, sub, "*.csv")))


def started(file):
    return datetime.strptime(re.search(r"\d{8}_\d{6}", file).group(), "%Y%m%d_%H%M%S")


def dose_in(note):
    """The isoflurane % a note sets; nan if it is about isoflurane but gives no level; None if
    it is not about isoflurane."""
    level = re.search(r"(?:to|at)\s*(\d+(?:\.\d+)?)", note)
    if level:
        return float(level.group(1))
    if re.search(r"iso|anes", note):
        return np.nan
    return None


def vitals(name):
    """Every 10 s of the session: time (s from the first recording's start), dose (%), heart rate
    (ECG), breathing rate and perfusion index (PPG, 30 s around it), temperature (°C)."""
    first = started(files(name)[0])
    v = {k: [] for k in ["t", "hr", "noise", "mains", "breathing", "ppg_hr", "pi", "temp"]}
    changes = []
    for f in files(name):
        with contextlib.redirect_stdout(io.StringIO()):
            rec = load(f)
            r = filters.r_waves(rec["ecg"], rec["fs_ecg"])
        offset = (started(f) - first).total_seconds()
        for time, note in rec["notes"]:
            level = dose_in(note)
            if level is not None:
                changes.append((offset + time, level))

        fs = rec["fs_ecg"]
        t = rec["t_ecg"]
        hp = filters.ecg_highpass(rec["ecg"], fs)
        notched = filters.notches(hp, fs, filters.MAINS_HZ)
        r_times = t[r]
        rr = np.diff(r_times)
        rr_t = r_times[1:]
        has_ppg = "ir_nA" in rec
        if has_ppg:
            fp = rec["fs_ppg"]
            tp = rec["t_ppg"]
            ppg = filters.ppg_lowpass(rec["ir_nA"], fp)
            heart = filters.heart_band(ppg, fp)
            breath = filters.breathing_band(ppg, fp)
            pulses = np.searchsorted(tp, r_times)

        for s in np.arange(0, t[-1] - STEP_S, STEP_S):
            inside = (t >= s) & (t < s + STEP_S)
            v["t"].append(offset + s + STEP_S / 2)
            v["noise"].append(np.std(notched[inside]))
            v["mains"].append(np.std(hp[inside] - notched[inside]))
            beats = rr[(rr_t >= s) & (rr_t < s + STEP_S)]
            beats = beats[(beats >= 0.08) & (beats <= 0.30)]             # 200-750 per minute, as r_waves searches
            if len(beats) >= 5:
                beats = beats[np.abs(beats / np.median(beats) - 1) < 0.3]   # drop missed and extra beats
                v["hr"].append(60 / np.mean(beats))
            else:
                v["hr"].append(np.nan)
            warm = (rec["t_temp"] >= s) & (rec["t_temp"] < s + STEP_S)
            v["temp"].append(np.median(rec["temp_c"][warm]) if warm.any() else np.nan)

            middle = s + STEP_S / 2
            if not has_ppg or middle - PPG_WINDOW_S / 2 < 0 or middle + PPG_WINDOW_S / 2 > tp[-1]:
                v["breathing"].append(np.nan)
                v["ppg_hr"].append(np.nan)
                v["pi"].append(np.nan)
                continue
            near = (tp >= middle - PPG_WINDOW_S / 2) & (tp < middle + PPG_WINDOW_S / 2)
            v["breathing"].append(filters.breathing_rate(breath[near], fp))
            fr, p = welch(heart[near], fs=fp, nperseg=int(10 * fp), nfft=int(PPG_WINDOW_S * fp))
            band = (fr >= filters.HEART_HZ[0]) & (fr <= filters.HEART_HZ[1])
            v["ppg_hr"].append(60 * fr[band][np.argmax(p[band])])
            in_window = (r_times >= middle - PPG_WINDOW_S / 2) & (r_times < middle + PPG_WINDOW_S / 2)
            pulse = panels.averaged(heart, pulses[in_window], int(0.15 * fp))
            v["pi"].append(np.nan if pulse is None else 100 * np.ptp(pulse) / np.mean(ppg[near]))

    v = {k: np.array(x) for k, x in v.items()}
    # ECG noise or mains far above the session's usual (movement, heating pad) -> no heart rate there
    v["clean"] = (v["noise"] <= 3 * np.median(v["noise"])) & (v["mains"] <= 5 * np.median(v["mains"]))
    v["hr"][~v["clean"]] = np.nan
    # breathing more than 30 % from the median of the minute around it: two breaths taken for one
    b = v["breathing"]
    around = np.full(len(b), np.nan)
    for i in range(len(b)):
        near = b[max(0, i - 6):i + 7]
        near = near[~np.isnan(near)]
        if len(near) >= 3:
            around[i] = np.median(near)
    b[np.abs(b / around - 1) > 0.3] = np.nan
    # the PPG pulse counts only where its rate matches the ECG's
    v["pi"][~(np.abs(v["ppg_hr"] / v["hr"] - 1) <= 0.05)] = np.nan
    v["changes"] = sorted(changes)
    v["end"] = v["t"][-1] + STEP_S / 2
    v["dose"] = np.array([dose_at(v["changes"], x) for x in v["t"]])
    return v


def dose_at(changes, time):
    level = np.nan
    for when, value in changes:
        if when <= time:
            level = value
    return level


def epochs(v):
    """(dose, start, end) for each stretch at one dose, from the dose notes."""
    out = []
    for i, (when, level) in enumerate(v["changes"]):
        end = v["changes"][i + 1][0] if i + 1 < len(v["changes"]) else v["end"]
        if np.isnan(level) or end <= when:
            continue
        if out and out[-1][0] == level and out[-1][2] == when:   # a note repeating the same level
            out[-1] = (level, out[-1][1], end)
        else:
            out.append((level, when, end))
    return out


def last_values(v, key, start, end, last_s=120):
    """The values of v[key] in the last 2 min recorded at one dose (missing ones left out)."""
    at_dose = np.flatnonzero((v["t"] >= start) & (v["t"] < end))
    x = v[key][at_dose[-int(last_s / STEP_S):]]
    return x[~np.isnan(x)]


def steady(v, key, start, end):
    """Median and quartiles of v[key] in the last 2 min recorded at one dose, or None with
    fewer than 3 values."""
    x = last_values(v, key, start, end)
    if len(x) < 3:
        return None
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def with_gaps(v, key, longest_s=60):
    """Minutes and the values of v[key], joined where they are at most 1 min apart and broken
    (a nan) where more is missing."""
    known = ~np.isnan(v[key])
    t = v["t"][known]
    y = v[key][known]
    cut = np.flatnonzero(np.diff(t) > longest_s) + 1
    return np.insert(t, cut, np.nan) / 60, np.insert(y, cut, np.nan)


def dose_averages(name, ecg_ms=(-100, 60), ppg_ms=(-50, 150), shift_ms=12):
    """For each dose, the averaged ECG beat and PPG pulse of the clean beats in the last 2 min
    recorded at it, and the session's breathing band and skin temperature.

    ECG: mains fitted out (a notch rings after the QRS), 3 Hz high-pass, median of the beats. Each
    beat is shifted (at most shift_ms) so its QRS matches the first dose's average best, because
    isoflurane makes the J wave taller than the R wave and the R-wave mark can land on either.
    The odd and the even beats are also averaged on their own, to check a measurement is repeatable.
    PPG pulse: the heart band in % of the light level, read every 2 ms with the same shift.
    Breathing: the PPG breathing band in % of the light level.
    """
    v = vitals(name)
    stretches = epochs(v)
    last = [np.flatnonzero((v["t"] >= start) & (v["t"] < end))[-12:] for _, start, end in stretches]
    wins = [v["t"][idx][v["clean"][idx]] for idx in last]
    first = started(files(name)[0])
    t_ppg = np.arange(ppg_ms[0], ppg_ms[1], 2)
    pieces = [[] for _ in stretches]                         # (ECG piece with room to shift, R time, file)
    rr = [[] for _ in stretches]
    ppg = []                                                 # per file: heart band and light level, read at any time
    out = {"v": v, "t_ppg": t_ppg, "t_breath": [], "breath": [], "t_temp": [], "temp": [], "doses": []}
    for f in files(name):
        with contextlib.redirect_stdout(io.StringIO()):
            rec = load(f)
            r = filters.r_waves(rec["ecg"], rec["fs_ecg"])
        if "ir_nA" not in rec:
            raise SystemExit(os.path.basename(f) + " has no usable PPG, so the pulse and breathing cannot be averaged.")
        offset = (started(f) - first).total_seconds()
        fs = rec["fs_ecg"]
        ecg = filters.ecg_highpass(filters.mains_fitted_out(rec["ecg"], fs), fs)
        m = int(shift_ms * fs / 1000)
        a, b = int(ecg_ms[0] * fs / 1000) - m, int(ecg_ms[1] * fs / 1000) + m
        light = filters.ppg_lowpass(rec["ir_nA"], rec["fs_ppg"])
        ppg.append((CubicSpline(rec["t_ppg"], filters.heart_band(light, rec["fs_ppg"])),
                    CubicSpline(rec["t_ppg"], light), rec["t_ppg"][0], rec["t_ppg"][-1]))
        out["t_breath"].append(offset + rec["t_ppg"])
        out["breath"].append(100 * filters.breathing_band(light, rec["fs_ppg"]) / np.mean(light))
        out["t_temp"].append(offset + rec["t_temp"])
        out["temp"].append(rec["temp_c"])
        for j in range(len(r) - 1):
            k = r[j]
            when = offset + rec["t_ecg"][k]
            for i, w in enumerate(wins):
                if len(w) and np.min(np.abs(w - when)) < STEP_S / 2 and k + a >= 0 and k + b <= len(ecg):
                    pieces[i].append((ecg[k + a:k + b], rec["t_ecg"][k], len(ppg) - 1))
                    rr[i].append(rec["t_ecg"][r[j + 1]] - rec["t_ecg"][k])
    for key in ["t_breath", "breath", "t_temp", "temp"]:
        out[key] = np.concatenate(out[key])

    t_ecg = 1000 * np.arange(a + m, b - m) / fs
    qrs = (t_ecg >= -20) & (t_ecg <= 30)                     # the part the shift is matched on
    template = np.median([p for p, _, _ in pieces[0]], axis=0)[m:-m][qrs]
    out["t_ecg"] = t_ecg
    for i, (level, start, end) in enumerate(stretches):
        beats, pulses = [], []
        for piece, r_time, n in pieces[i]:
            match = [np.dot(piece[m + s:len(piece) - m + s][qrs], template) for s in range(-m, m + 1)]
            s = int(np.argmax(match)) - m
            heart, level_nA, t0, t1 = ppg[n]
            at = r_time + s / fs + t_ppg / 1000
            if at[0] < t0 or at[-1] > t1:
                continue
            beats.append(piece[m + s:len(piece) - m + s])
            pulses.append(100 * heart(at) / np.mean(level_nA(at)))
        if len(beats) >= 10:
            out["doses"].append({"level": level, "start": start, "end": end, "last": last[i],
                                 "beats": len(beats), "rr_ms": 1000 * np.median(rr[i]),
                                 "beat": np.median(beats, axis=0), "pulse": np.median(pulses, axis=0),
                                 "halves": (np.median(beats[0::2], axis=0), np.median(beats[1::2], axis=0))})
    return out


def ecg_points(t_ms, beat, k=0.15):
    """The wave boundaries and sizes on an averaged beat, as Merentie et al. (2015) find them: from
    the slope, a boundary is where it falls below k x the steepest slope of that wave.

    QRS onset: the start of the R upstroke (the dip before R is the chip's filter ringing, not a Q
    wave). QRS end: halfway between the S and J peaks. QRSp end: the J wave back at the baseline.
    Baseline: the level at P onset. Times in ms from R, sizes in µV from the baseline.
    """
    t = np.arange(t_ms[0], t_ms[-1], 1000 / 8192)            # 8192 points per second, between samples
    y = CubicSpline(t_ms, beat)(t)
    d = np.gradient(y, t)

    def where(lo, hi):
        return np.flatnonzero((t >= lo) & (t <= hi))

    def walk(i, step, slope):
        """From sample i, step back (-1) or forward (+1) while the slope is above k x slope."""
        while 0 < i < len(t) - 1 and d[i] / slope > k:
            i += step
        return i

    around = where(-3, 3)
    r = around[np.argmax(y[around])]
    up = where(t[r] - 10, t[r])
    steep = up[np.argmax(d[up])]
    on = walk(steep, -1, d[steep])
    after = where(t[r], t[r] + 8)
    s = after[np.argmin(y[after])]
    later = where(t[s], t[s] + 12)
    j = later[np.argmax(y[later])]

    before = where(t[on] - 70, t[on] - 15)
    p = before[np.argmax(y[before])]
    rise = where(t[p] - 20, t[p])
    p_up = rise[np.argmax(d[rise])]
    p_on = walk(p_up, -1, d[p_up])
    fall = where(t[p], t[p] + 20)
    p_down = fall[np.argmin(d[fall])]
    p_off = walk(p_down, 1, d[p_down])
    base = y[p_on]
    back = where(t[j], t[j] + 40)
    back = back[y[back] <= base]
    return {"base": base, "P on": t[p_on], "P off": t[p_off], "QRS on": t[on],
            "QRS off": (t[s] + t[j]) / 2, "QRSp off": t[back[0]] if len(back) else np.nan,
            "P": (t[p], y[p] - base), "R": (t[r], y[r] - base), "S": (t[s], y[s] - base), "J": (t[j], y[j] - base)}
