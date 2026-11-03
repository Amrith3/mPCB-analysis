"""Signal quality of every recording in data/, as a table (quality.csv):

- clear beat: the 30 s pieces in which the ECG's beat repeats clearly (filters.r_waves)
- mains: the share of the ECG's power (3 Hz high-passed) within 1 Hz of the mains and its harmonics
- missing: samples lost in gaps between time stamps; mains shift: lost without a gap (data_loss.py)
- clipped: ECG samples at the end of the chip's range
- PPG: used, or why not (recording.py)
- SNR of the ECG (high-pass and notches) and of the PPG heart band, and the heart-rate variability
  (SDNN, RMSSD): median (IQR) of the 30 s pieces with a clear beat (filters.snr_db, filters.hrv)
"""
import contextlib
import csv
import glob
import io
import os
import re

import numpy as np
from scipy.signal import welch

import filters
from recording import DATA, load, unzip

for z in glob.glob(os.path.join(DATA, "*.zip")):
    unzip(os.path.basename(z)[:-4])


def missing(t, fs):
    steps = np.diff(t) * fs
    return int(np.sum(np.round(steps[steps > 1.5]) - 1))


def median_iqr(x, fmt):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "-"
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return (fmt + " (" + fmt + "–" + fmt + ")") % (m, q1, q3)


rows = []
for f in sorted(glob.glob(os.path.join(DATA, "*", "**", "*.csv"), recursive=True)):
    pieces = []
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        rec = load(f)
        r = filters.r_waves(rec["ecg"], rec["fs_ecg"], pieces)
    fs = rec["fs_ecg"]
    ecg = filters.ecg_highpass(rec["ecg"], fs)
    used = filters.notches(ecg, fs)
    r_t = filters.r_times(rec["ecg"], rec["t_ecg"], fs, r)
    if "ir_nA" in rec:
        fp = rec["fs_ppg"]
        heart = filters.heart_band(filters.ppg_lowpass(rec["ir_nA"], fp), fp)
    snr_ecg, snr_ppg, sdnn, rmssd = [], [], [], []
    for first, end, clear in pieces:
        inside = (r >= first) & (r < end) & ~np.isnan(r_t)
        if not clear or np.sum(inside) < 10:
            continue
        rr = np.diff(r_t[inside])
        snr_ecg.append(filters.snr_db(used, np.searchsorted(rec["t_ecg"], r_t[inside]), int(np.median(rr) * fs)))
        if "ir_nA" in rec:
            pulses = np.searchsorted(rec["t_ppg"], r_t[inside])
            snr_ppg.append(filters.snr_db(heart, pulses, int(np.median(rr) * fp)))
        hrv = filters.hrv(rr, fs)
        sdnn.append(hrv[0])
        rmssd.append(hrv[1])
    fr, p = welch(ecg, fs=fs, nperseg=min(len(ecg), int(8 * fs)))
    total = p[fr >= filters.ECG_HP_HZ].sum()
    mains = sum(p[np.abs(fr - f0) <= 1].sum() for f0 in filters.notch_frequencies(fs, filters.MAINS_HZ))
    fr, p = welch(rec["ecg"], fs=fs, nperseg=min(len(ecg), int(32 * fs)))
    near = (fr > 48) & (fr < 52)
    f_mains = fr[near][np.argmax(p[near])]
    reason = [line for line in said.getvalue().splitlines() if line.startswith("PPG not used")]
    name = re.search(r"\d{8}_\d{6}", os.path.basename(f)).group()
    rows.append({
        "Recording": "%s.%s %s:%s:%s" % (name[6:8], name[4:6], name[9:11], name[11:13], name[13:15]),
        "Folder": os.path.relpath(os.path.dirname(f), DATA),
        "Duration (s)": "%.0f" % (len(rec["ecg"]) / fs),
        "Clear beat (pieces)": "%d of %d" % (sum(c for _, _, c in pieces), len(pieces)),
        "Mains (% of ECG power)": "%.1f" % (100 * mains / total),
        "ECG missing (samples)": missing(rec["t_ecg"], fs),
        "PPG missing (samples)": missing(rec["t_ppg"], rec["fs_ppg"]) if "t_ppg" in rec else "-",
        "Mains at (Hz)": "%.2f" % f_mains,
        "Lost without a gap (%)": "%.1f" % (100 * (1 - 50 / f_mains)) if f_mains > 50.2 else "0",
        "ECG clipped (%)": "%.2f" % (100 * np.mean(np.abs(rec["ecg_raw"]) >= 0.99 * 131071)),
        "ECG SNR (dB)": median_iqr(snr_ecg, "%.1f"),
        "PPG SNR (dB)": median_iqr(snr_ppg, "%.1f"),
        "SDNN (ms)": median_iqr(sdnn, "%.2f"),
        "RMSSD (ms)": median_iqr(rmssd, "%.2f"),
        "PPG": "used (IR %.0f nA)" % np.mean(rec["ir_nA"]) if "ir_nA" in rec else reason[0][len("PPG not used: "):] if reason else "not used",
    })

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality.csv"), "w", newline="", encoding="utf-8-sig") as out:
    writer = csv.DictWriter(out, list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(" | ".join(rows[0].keys()))
for row in rows:
    print(" | ".join(str(v) for v in row.values()))
