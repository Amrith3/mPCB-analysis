"""Statistics: our values per dose as mean ± SD over runs, median (IQR) for the tables, and t-tests
against published values.

A paper usually gives only a mean, a spread and the number of animals, so the t-test is worked out
from those summary numbers. It is the two-sided Student's t-test with a pooled variance, the same
as the MedCalc comparison-of-means calculator (https://www.medcalc.org/calc/comparison_of_means.php).
"""
import math

import numpy as np
from scipy.stats import ttest_ind_from_stats

from .sessions import Session

RUNS = ["22 Oct, 30 g", "26 Oct, 30 g", "22 Oct, 25 g"]   # the runs that went through 1, 2 and 4 %


def mean_sd(values):
    """Mean, standard deviation (n - 1) and n of a list of values, nan left out."""
    x = np.asarray(values, float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"), len(x)


def median_iqr(values, fmt, least=1):
    """'median (quartile 1–quartile 3)' of the values without nan, each in the format fmt, or '-'
    with fewer than `least` values."""
    x = np.asarray(values, float)
    x = x[~np.isnan(x)]
    if len(x) < max(least, 1):
        return "-"
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return (fmt + " (" + fmt + "–" + fmt + ")") % (m, q1, q3)


def sd_from_sem(sem, n):
    """The standard deviation from a standard error of the mean: SD = SEM x sqrt(n)."""
    return sem * math.sqrt(n)


def t_test(mean1, sd1, n1, mean2, sd2, n2):
    """Two-sided Student's t-test of two means from their mean, SD and n. Returns (t, p)."""
    t, p = ttest_ind_from_stats(mean1, sd1, n1, mean2, sd2, n2, equal_var=True)
    return float(t), float(p)


def per_dose(key, runs=RUNS, doses=(1, 2, 4)):
    """{dose: [value of each run]} for a key of the session vitals ("hr", "breathing", ...), each
    the mean of the last 2 min at the first stretch at that dose (Session.dose_values), in the order
    of runs; nan where a run has no stretch at that dose."""
    out = {d: [] for d in doses}
    for name in runs:
        values = Session(name).dose_values(key, doses)
        for d in doses:
            out[d].append(values.get(d, np.nan))
    return out
