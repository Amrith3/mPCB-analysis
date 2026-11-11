"""Signal quality of every recording in data/, as a table (quality.csv):

- clear beat: the 30 s pieces in which the ECG's beat repeats clearly (analysis.r_waves)
- mains: the share of the ECG's power (3 Hz high-passed) within 1 Hz of the mains and its harmonics
- missing: samples lost in gaps between time stamps; mains shift: lost without a gap
  (Recording.lost_samples, Recording.mains_hz)
- clipped: ECG samples at the end of the chip's range (Recording.clipped)
- PPG: used, or why not (Recording)
- SNR of the ECG (high-pass and notches) and of the PPG heart band, and the heart-rate variability
  (SDNN, RMSSD): median (IQR) of the 30 s pieces with a clear beat (Recording.beat_quality)
"""
import contextlib
import io
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import welch

from mpcb import filters
from mpcb.recording import DATA, Recording, all_files
from mpcb.stats import median_iqr

rows = []
for f in all_files():
    with contextlib.redirect_stdout(io.StringIO()):
        rec = Recording(f)
        pieces, snr_ecg, snr_ppg, sdnn, rmssd = rec.beat_quality()
    fs = rec.fs_ecg
    ecg = filters.ecg_highpass(rec.ecg, fs)
    fr, p = welch(ecg, fs=fs, nperseg=min(len(ecg), int(8 * fs)))
    total = p[fr >= filters.ECG_HP_HZ].sum()
    mains = sum(p[np.abs(fr - f0) <= 1].sum() for f0 in filters.notch_frequencies(fs, filters.MAINS_HZ))
    f_mains = rec.mains_hz()
    name = re.search(r"\d{8}_\d{6}", os.path.basename(f)).group()
    rows.append({
        "Recording": "%s.%s %s:%s:%s" % (name[6:8], name[4:6], name[9:11], name[11:13], name[13:15]),
        "Folder": os.path.relpath(os.path.dirname(f), DATA),
        "Duration (s)": "%.0f" % (len(rec.ecg) / fs),
        "Clear beat (pieces)": "%d of %d" % (sum(c for _, _, c in pieces), len(pieces)),
        "Mains (% of ECG power)": "%.1f" % (100 * mains / total),
        "ECG missing (samples)": rec.lost_samples(),
        "PPG missing (samples)": rec.lost_samples("ppg") if rec.has_ppg else "-",
        "Mains at (Hz)": "%.2f" % f_mains,
        "Lost without a gap (%)": "%.1f" % (100 * (1 - 50 / f_mains)) if f_mains > 50.2 else "0",
        "ECG clipped (%)": "%.2f" % rec.clipped(),
        "ECG SNR (dB)": median_iqr(snr_ecg, "%.1f"),
        "PPG SNR (dB)": median_iqr(snr_ppg, "%.1f"),
        "SDNN (ms)": median_iqr(sdnn, "%.2f"),
        "RMSSD (ms)": median_iqr(rmssd, "%.2f"),
        "PPG": "used (IR %.0f nA)" % np.mean(rec.ir_nA) if rec.has_ppg else rec.ppg_problem,
    })

pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality.csv"), index=False,
                          encoding="utf-8-sig")
print(" | ".join(rows[0].keys()))
for row in rows:
    print(" | ".join(str(v) for v in row.values()))
