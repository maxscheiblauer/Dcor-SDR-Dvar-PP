"""
evaluation.py — out-of-sample scoring, ported verbatim from the legacy steps
(step9/step13) so every number stays comparable across the rebuild:

    mspe                    : out-of-sample mean squared prediction error
    campbell_thompson_r2    : 1 - MSPE / MSPE(historical-mean benchmark)
    nw_var                  : Newey-West HAC long-run variance (lag 18)
    dm_test                 : Diebold-Mariano test on squared-error loss
"""
from __future__ import annotations
from math import erf, sqrt
import numpy as np

NW_LAG = 18


def mspe(err):
    err = np.asarray(err, float)
    return float((err ** 2).mean())


def campbell_thompson_r2(mspe_model, mspe_bench):
    return float(1.0 - mspe_model / mspe_bench)


def nw_var(u, lag=NW_LAG):
    u = np.asarray(u, float); u = u - u.mean()
    s = (u * u).mean()
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1)) * (u[k:] * u[:-k]).mean()
    return float(s)


def dm_test(e1, e2, lag=NW_LAG):
    """DM statistic and two-sided p-value for H0: equal squared-error loss.
    Positive DM => model 1 has larger loss (model 2 better)."""
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    d = e1 ** 2 - e2 ** 2
    n = len(d); s2 = nw_var(d, lag)
    if n <= lag + 1 or s2 <= 0:
        return float("nan"), float("nan")
    dm = sqrt(n) * d.mean() / sqrt(s2)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm) / sqrt(2.0))))
    return float(dm), float(p)
