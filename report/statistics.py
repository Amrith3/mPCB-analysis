"""The t-tests of the results report: our values per dose against published values (t_tests.csv).

Ours: mean ± SD over the three runs that went through 1, 2 and 4 % (Mouse 1 on 22 and 26 Oct,
Mouse 2 on 22 Oct), each run's value the mean of its last 2 min at that dose (heart rate,
breathing) or the interval of its averaged beat (isoflurane/ecg.csv; run isoflurane/table.py first).
Published: mean ± SD and n as given in the paper (SD = SEM x sqrt(n) where the paper gives the SEM).
Test: two-sided Student's t-test from these summary numbers (mpcb.stats.t_test). With three runs from
two mice the tests are only a first indication.
"""
import contextlib
import csv
import io
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mpcb import stats  # noqa: E402

# (our value, dose, study, condition, mean, SD, n)
PUBLISHED = [
    ("P", 2, "Merentie et al. (2015)", "young mice, isoflurane", 15.9, 2.1, 73),
    ("PQ", 2, "Merentie et al. (2015)", "young mice, isoflurane", 43.6, 3.2, 73),
    ("QRS", 2, "Merentie et al. (2015)", "young mice, isoflurane", 10.8, 1.2, 73),
    ("P", 2, "Shintaku et al. (2015)", "2 % isoflurane", 18.5, stats.sd_from_sem(1.0, 71), 71),
    ("PQ", 2, "Shintaku et al. (2015)", "2 % isoflurane", 35.7, stats.sd_from_sem(0.9, 71), 71),
    ("QRS", 2, "Shintaku et al. (2015)", "2 % isoflurane", 11.1, stats.sd_from_sem(0.1, 71), 71),
    ("hr", 1, "Constantinides et al. (2011)", "1 % isoflurane", 408, 23, 6),
    ("hr", 1, "Picard et al. (2014)", "light anaesthesia, 0.5 % isoflurane", 522.5, 16.72, 12),
    ("hr", 2, "Constantinides et al. (2011)", "2 % isoflurane", 482, 10, 5),
    ("hr", 2, "Shintaku et al. (2015)", "2 % isoflurane", 494, 57.3, 146),
    ("hr", 4, "Picard et al. (2014)", "deep anaesthesia, 2 % with a temporary 4 %", 362.92, 27.26, 12),
    ("breathing", 2, "Tsukamoto et al. (2015)", "after 35 min at 2 % isoflurane", 78, 19, 8),
]
UNITS = {"hr": "beats/min", "breathing": "breaths/min", "P": "ms", "PQ": "ms", "QRS": "ms"}


def intervals():
    """{(interval, dose): [value of each run]} from isoflurane/ecg.csv, first stretch at each dose."""
    path = os.path.join(os.path.dirname(HERE), "isoflurane", "ecg.csv")
    if not os.path.exists(path):
        raise SystemExit("Run isoflurane/table.py first: it writes isoflurane/ecg.csv.")
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["Isoflurane (%)"] not in ("1", "2", "4"):
            continue
        for key, column in [("P", "P duration (ms)"), ("PQ", "PQ (ms)"), ("QRS", "QRS (ms)")]:
            if row[column] not in ("-", ""):
                out.setdefault((key, int(row["Isoflurane (%)"])), []).append(float(row[column]))
    return out


ours = intervals()
with contextlib.redirect_stdout(io.StringIO()):
    for key in ("hr", "breathing"):
        for dose, values in stats.per_dose(key).items():
            ours[(key, dose)] = values

rows = []
for key, dose, study, condition, mean, sd, n in PUBLISHED:
    m, s, k = stats.mean_sd(ours[(key, dose)])
    t, p = stats.t_test(m, s, k, mean, sd, n)
    rows.append({"Value": "%s [%s]" % ("Heart rate" if key == "hr" else "Breathing" if key == "breathing" else key, UNITS[key]),
                 "Dose (%)": dose, "Ours (mean ± SD)": "%.1f ± %.1f" % (m, s), "Ours n": k,
                 "Published (mean ± SD)": "%.1f ± %.1f" % (mean, sd), "Published n": n,
                 "Study": study, "Condition": condition, "t": "%.2f" % t, "p": "%.3f" % p})
pd.DataFrame(rows).to_csv(os.path.join(HERE, "t_tests.csv"), index=False, encoding="utf-8-sig")
for row in rows:
    print("%-24s %g %%  ours %-13s (n=%d)  %-29s %-13s (n=%3d)  p = %s" % (
        row["Value"], row["Dose (%)"], row["Ours (mean ± SD)"], row["Ours n"], row["Study"],
        row["Published (mean ± SD)"], row["Published n"], row["p"]))
