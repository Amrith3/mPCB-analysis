"""Tables of each isoflurane dose of each run, as the literature reports them.

vitals.csv: mean ± SD of the 10 s values in the last 2 min recorded at the dose (Constantinides
2011, Tsukamoto 2015); the SD is over time within one animal. The perfusion index is in % of the
run's 1 % value, since its size depends on how firmly the sensor sits. Skin temperature: the mean.
ecg.csv: intervals on the averaged beat of the same 2 min (Merentie 2015), found by
sessions.ecg_points. A time is given only where the averages of the odd and of the even beats
agree within 2 ms; otherwise it is "-" (the wave is too small for the noise). The P wave is also
"-" when R-R is within 1.5 ms of a whole number of mains cycles (20 ms): the mains left in the ECG
then falls at the same point of every beat and does not average away. Left out: QT (the end of the
T wave is not reliable here) and QRSp (the J wave does not come back to the baseline).
snr_hrv.csv: signal-to-noise ratio of the ECG and of the PPG heart band, SDNN and RMSSD, each as
median (IQR) of the 10 s steps in the same 2 min (each from the 30 s around it; sessions.vitals),
and the floor of SDNN and RMSSD: what a perfectly regular heart would read with that ECG's noise
(sessions.hrv_floor). SDNN or RMSSD close to its floor is mostly measurement noise.
table.csv: all of it, with the number of beats, the perfusion index in % of the light level, the
coefficient of variation (CV = 100 x SD / mean; not for temperature, as °C has no true zero) and
the wave sizes.
Saved in this folder.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the main folder
from mpcb import sessions  # noqa: E402

RUNS = [("22 Oct, 30 g", "Mouse 1 (30 g), 22 Oct"),
        ("26 Oct, 30 g", "Mouse 1 (30 g), 26 Oct"),
        ("22 Oct, 25 g", "Mouse 2 (25 g), 22 Oct")]
VITALS = [("hr", "Heart rate (beats/min)", "%.0f"), ("breathing", "Breathing (breaths/min)", "%.1f"),
          ("pi", "Perfusion index (%)", "%.3f"), ("temp", "Skin temperature (°C)", "%.2f")]
# title, value from the points, the times that must repeat in both halves
ECG = [("P (µV)", lambda p: p["P"][1], ["P on", "P off"]),
       ("P duration (ms)", lambda p: p["P off"] - p["P on"], ["P on", "P off"]),
       ("PQ (ms)", lambda p: p["QRS on"] - p["P on"], ["P on", "P off", "QRS on"]),
       ("QRS (ms)", lambda p: p["QRS off"] - p["QRS on"], ["QRS on", "QRS off"]),
       ("R (µV)", lambda p: p["R"][1], []),
       ("S (µV)", lambda p: p["S"][1], []),
       ("J (µV)", lambda p: p["J"][1], [])]
SIGNAL = [("ecg_snr", "ECG SNR (dB)", "%.1f"), ("ppg_snr", "PPG SNR (dB)", "%.1f"),
          ("sdnn", "SDNN (ms)", "%.2f"), ("rmssd", "RMSSD (ms)", "%.2f")]


def mean_sd(x, fmt):
    """'mean ± SD' of the values x in the format fmt, or '-' with fewer than 3 or missing values."""
    if len(x) < 3 or not np.all(np.isfinite(x)):
        return "-"
    return (fmt + " ± " + fmt) % (np.mean(x), np.std(x, ddof=1))


def median_iqr(v, key, start, end, fmt):
    """'median (quartile 1 to quartile 3)' of v[key] in the last 2 min at a dose, or '-'."""
    m = sessions.steady(v, key, start, end)
    return "-" if m is None else (fmt + " (" + fmt + "–" + fmt + ")") % m


def write(rows, name):
    """Save the rows (a list of dicts) as a CSV file in this folder and print them as a table."""
    columns = list(rows[0].keys())
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [columns] + [[str(row[c]) for c in columns] for row in rows]
    widths = [max(len(line[i]) for line in lines) for i in range(len(columns))]
    print("\n" + name)
    for line in lines:
        print("  ".join(cell.ljust(w) for cell, w in zip(line, widths)))


rows, vitals, ecg, signal = [], [], [], []
for name, label in RUNS:
    d = sessions.dose_averages(name)
    levels = []
    for c in d["doses"]:
        levels.append(c["level"])
        dose = "%g" % c["level"] + (" (again)" if levels.count(c["level"]) > 1 else "")
        row = {"Run": label, "Isoflurane (%)": dose, "Beats": c["beats"]}
        x = {}
        for key, title, fmt in VITALS:
            x[key] = sessions.last_values(d["v"], key, c["start"], c["end"])
            row[title] = mean_sd(x[key], fmt)
            if key != "temp":
                row[title.split(" (")[0] + " CV (%)"] = (
                    "%.1f" % (100 * np.std(x[key], ddof=1) / np.mean(x[key])) if len(x[key]) >= 3 else "-")
        if len(levels) == 1:
            pi_first = np.mean(x["pi"])
        row["RR (ms)"] = "%.1f" % c["rr_ms"]
        points = sessions.ecg_points(d["t_ecg"], c["beat"])
        odd, even = [sessions.ecg_points(d["t_ecg"], half) for half in c["halves"]]
        locked = abs(c["rr_ms"] - 20 * round(c["rr_ms"] / 20)) <= 1.5   # mains in step with the beats
        for title, value, times in ECG:
            repeats = all(abs(odd[t] - even[t]) <= 2 for t in times) and not (locked and "P on" in times)
            p = value(points)
            row[title] = "%.1f" % p if repeats and "ms" in title else "%.0f" % p if repeats else "-"
        for key, title, fmt in SIGNAL:
            row[title] = median_iqr(d["v"], key, c["start"], c["end"], fmt)
        floor = sessions.hrv_floor(c)
        row["SDNN floor (ms)"] = "%.2f" % floor[0] if np.isfinite(floor[0]) else "-"
        row["RMSSD floor (ms)"] = "%.2f" % floor[1] if np.isfinite(floor[1]) else "-"
        rows.append(row)
        run = label if len(levels) == 1 else ""
        vitals.append({"Run": run, "Isoflurane (%)": dose,
                       "Heart rate (beats/min)": row["Heart rate (beats/min)"],
                       "Breathing (breaths/min)": row["Breathing (breaths/min)"],
                       "Perfusion index (% of 1 %)": mean_sd(100 * x["pi"] / pi_first, "%.1f"),
                       "Skin temperature (°C)": "%.1f" % np.mean(x["temp"])})
        ecg.append({"Run": run, "Isoflurane (%)": dose,
                    **{k: row[k] for k in ["RR (ms)", "PQ (ms)", "P duration (ms)", "QRS (ms)"]}})

        signal.append({"Run": run, "Isoflurane (%)": dose, **{title: row[title] for _, title, _ in SIGNAL},
                       **{k: row[k] for k in ["SDNN floor (ms)", "RMSSD floor (ms)"]}})

write(vitals, "vitals.csv")
write(ecg, "ecg.csv")
write(signal, "snr_hrv.csv")
write(rows, "table.csv")
