"""Summary table: each isoflurane dose of each run, as the literature reports it.

Vitals: mean ± SD and coefficient of variation (CV = 100 x SD / mean) of the 10 s values in the
last 2 min recorded at the dose (Constantinides 2011, Tsukamoto 2015).
ECG: intervals and wave sizes on the averaged beat of the same 2 min (Merentie 2015), found by
sessions.ecg_points. A time is given only where the averages of the odd and of the even beats
agree within 2 ms; otherwise it is "-" (the wave is too small for the noise). The P wave is also
"-" when R-R is within 1.5 ms of a whole number of mains cycles (20 ms): the mains left in the ECG
then falls at the same point of every beat and does not average away. Left out: QT (the end of the
T wave is not reliable here) and QRSp (the J wave does not come back to the baseline).
Saved as table.csv in this folder.
"""
import csv
import os

import numpy as np

import sessions

RUNS = [("26 Oct, 30 g", "Mouse 1 (30 g), 26 Oct"),
        ("22 Oct, 30 g", "Mouse 1 (30 g), 22 Oct"),
        ("22 Oct, 25 g", "Mouse 2 (25 g), 22 Oct")]
VITALS = [("hr", "Heart rate (beats/min)", "%.0f"), ("breathing", "Breathing (breaths/min)", "%.0f"),
          ("pi", "Perfusion index (%)", "%.3f"), ("temp", "Skin temperature (°C)", "%.2f")]
# title, value from the points, the times that must repeat in both halves
ECG = [("P (µV)", lambda p: p["P"][1], ["P on", "P off"]),
       ("P duration (ms)", lambda p: p["P off"] - p["P on"], ["P on", "P off"]),
       ("PQ (ms)", lambda p: p["QRS on"] - p["P on"], ["P on", "P off", "QRS on"]),
       ("QRS (ms)", lambda p: p["QRS off"] - p["QRS on"], ["QRS on", "QRS off"]),
       ("R (µV)", lambda p: p["R"][1], []),
       ("S (µV)", lambda p: p["S"][1], []),
       ("J (µV)", lambda p: p["J"][1], [])]

rows = []
for name, label in RUNS:
    d = sessions.dose_averages(name)
    levels = []
    for c in d["doses"]:
        levels.append(c["level"])
        row = {"Run": label, "Isoflurane (%)": "%g" % c["level"] + (" (again)" if levels.count(c["level"]) > 1 else ""),
               "Beats": c["beats"]}
        for key, title, fmt in VITALS:
            x = sessions.last_values(d["v"], key, c["start"], c["end"])
            if len(x) >= 3:
                mean, sd = np.mean(x), np.std(x, ddof=1)
                row[title] = (fmt + " ± " + fmt) % (mean, sd)
                row[title.split(" (")[0] + " CV (%)"] = "%.1f" % (100 * sd / mean)
            else:
                row[title] = "-"
                row[title.split(" (")[0] + " CV (%)"] = "-"
        row["RR (ms)"] = "%.1f" % c["rr_ms"]
        points = sessions.ecg_points(d["t_ecg"], c["beat"])
        odd, even = [sessions.ecg_points(d["t_ecg"], half) for half in c["halves"]]
        locked = abs(c["rr_ms"] - 20 * round(c["rr_ms"] / 20)) <= 1.5   # mains in step with the beats
        for title, value, times in ECG:
            repeats = all(abs(odd[t] - even[t]) <= 2 for t in times) and not (locked and "P on" in times)
            x = value(points)
            row[title] = "%.1f" % x if repeats and "ms" in title else "%.0f" % x if repeats else "-"
        rows.append(row)

columns = list(rows[0].keys())
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "table.csv"), "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, columns)
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print("; ".join("%s: %s" % (k, row[k]) for k in columns))
