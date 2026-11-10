"""Statistics: our values per dose as mean ± SD over runs, and t-tests against published values.

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
    """Mean, standard deviation (n - 1) and n of a list of values."""
    x = np.asarray(values, float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"), len(x)


def sd_from_sem(sem, n):
    """The standard deviation from a standard error of the mean: SD = SEM x sqrt(n)."""
    return sem * math.sqrt(n)


def t_test(mean1, sd1, n1, mean2, sd2, n2):
    """Two-sided Student's t-test of two means from their mean, SD and n. Returns (t, p)."""
    t, p = ttest_ind_from_stats(mean1, sd1, n1, mean2, sd2, n2, equal_var=True)
    return float(t), float(p)


def per_dose(key, runs=RUNS, doses=(1, 2, 4)):
    """{dose: [value of each run]} for a key of the session vitals ("hr", "breathing", ...), each
    the mean of the last 2 min at the first stretch at that dose (Session.dose_values)."""
    out = {d: [] for d in doses}
    for name in runs:
        values = Session(name).dose_values(key, doses)
        for d in doses:
            if d in values:
                out[d].append(values[d])
    return out
