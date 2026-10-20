"""Reads one GUI recording. Change FILE to analyse another one."""
import pandas as pd

FILE = r"D:\Users\schwa\Documents\PlatformIO\Projects\mPCB\recordings\mouse_isoflurane_20261022\mouse_30g\mice_anest_20261022_145430.csv"


def load(file=FILE):
    with open(file) as f:
        first_line = f.readline()

    settings = {}
    for part in first_line.split():
        if "=" in part:
            name, value = part.split("=")
            settings[name] = value

    if "ecg_rate" not in settings:
        raise SystemExit("This recording has no settings line, so it is not used.")

    data = pd.read_csv(file, comment="#", low_memory=False)

    rec = {}
    ecg = data[data["ecg_uv"].notna()]
    rec["fs_ecg"] = float(settings["ecg_rate"].rstrip("sp"))
    rec["t_ecg"] = ecg["time_s"].to_numpy()
    rec["ecg"] = ecg["ecg_uv"].to_numpy()
    rec["ecg_raw"] = ecg["ecg_raw"].to_numpy()
    seconds = len(rec["ecg"]) / rec["fs_ecg"]
    print("ECG:", len(rec["ecg"]), "samples at", rec["fs_ecg"], "per second")

    adc_range = settings.get("adc_range", "unknown")
    if not adc_range.isdigit():
        print("PPG not used: the settings line gives no PPG range")
        return rec
    ppg = data[data["ir"].notna()]
    ir_nA = ppg["ir"].to_numpy() * float(adc_range) / 262143
    if ir_nA.mean() < 150:
        print("PPG not used: IR is", round(ir_nA.mean()), "nA, the sensor was not on the skin")
        return rec
    if (ir_nA >= 0.99 * float(adc_range)).mean() > 0.5:
        print("PPG not used: IR is clipped at the top of its range (too much light)")
        return rec
    rec["ir_nA"] = ir_nA
    rec["red_nA"] = ppg["red"].to_numpy() * float(adc_range) / 262143

    # the PPG clock runs ~0.2 % fast; counting its samples per ECG second puts it on the ECG clock
    written = float(settings["ppg_sr"].rstrip("sp"))
    counted = len(ir_nA) / seconds
    fast = counted / written
    if abs(fast - 1) < 0.006:
        rec["fs_ppg"] = counted
    else:
        rec["fs_ppg"] = written
    if fast < 1.0015 or fast > 1.0035:
        print("warning: the PPG has", round(100 * (fast - 1), 2), "% more samples than its setting says"
              " (0.15 to 0.35 % in a complete recording), so samples were lost: PPG-to-ECG timing is not reliable")
    rec["t_ppg"] = ppg["time_s"].to_numpy() * written / rec["fs_ppg"]
    print("PPG:", len(ir_nA), "samples at", round(rec["fs_ppg"], 2), "per second")
    return rec
