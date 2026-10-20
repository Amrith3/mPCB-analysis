"""Missing samples (gaps, mains shift) and ECG clipping."""
import numpy as np
from scipy.signal import welch

from recording import load

rec = load()

for name in ["ecg", "ppg"]:
    if "t_" + name in rec:
        t = rec["t_" + name]
        fs = rec["fs_" + name]
        steps = np.diff(t) * fs
        gaps = steps[steps > 1.5]
        missing = np.sum(np.round(gaps) - 1)
        print(name.upper(), "samples missing in gaps:", int(missing), "of", len(t) + int(missing))

# lost samples without a gap speed the signal up, so the mains shows above 50 Hz
piece = min(len(rec["ecg"]), int(32 * rec["fs_ecg"]))
f, p = welch(rec["ecg"], fs=rec["fs_ecg"], nperseg=piece)
near_50 = (f > 48) & (f < 52)
f_near = f[near_50]
p_near = p[near_50]
f_mains = f_near[np.argmax(p_near)]
print("mains appears at", round(f_mains, 2), "Hz")
if f_mains > 50.2:
    print("samples lost without a gap: about", round(100 * (1 - 50 / f_mains), 1), "%")
else:
    print("no samples lost without a gap")

clipped = np.abs(rec["ecg_raw"]) >= 0.99 * 131071
print("ECG samples clipped:", round(100 * np.mean(clipped), 2), "%")
