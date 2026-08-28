"""
regression.py — Layer-2 second-stage regression (PLAN_rebuild.md §4).

    linear : y ~ g1..gd                 (OLS)
    poly   : y ~ g1, g1^3, g2..gd       (cube on the leading direction only —
                                         Ludvigson-Ng's BIC-selected form)

Each optionally adds the Cochrane-Piazzesi factor CP as one more regressor.
Coefficients are fit on train, frozen, and applied out-of-sample.  Two extra
DIAGNOSTIC specs (poly_bounded, poly_mono) reproduce the legacy Step-11/12
probes and are used only in the DvarSDR synthesis, not the core grid.
"""
from __future__ import annotations
import numpy as np


def ols_predict(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    B = np.column_stack([np.ones(len(Xte)), Xte])
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return B @ beta


def _empirical_normal_map(g_tr):
    """Train-fit monotone empirical-CDF -> standard-normal reshaping (Step-12
    `poly_mono`): the closest a projection score can get to INPCA's Brenier
    reshaping without optimal transport.  Returns a callable g -> z."""
    from scipy.stats import norm
    xs = np.sort(g_tr)
    n = len(xs)
    ranks_target = norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    def _map(g):
        # linear interpolation of the train quantile map, clipped to train range
        return np.interp(g, xs, ranks_target, left=ranks_target[0],
                         right=ranks_target[-1])
    return _map


def design(Z, d, spec, cp=None, g1_bound=None, mono_map=None):
    """Design matrix for `spec` in {linear, poly, poly_bounded, poly_mono}.

    Z        : (n, d) projected scores (all rows; caller slices train/test)
    cp       : (n,) CP column or None
    g1_bound : (lo, hi) train support to clip g1 before cubing (poly_bounded)
    mono_map : callable from `_empirical_normal_map` (poly_mono)
    """
    g1 = Z[:, 0]
    cols = [g1]
    if spec == "linear":
        pass
    elif spec == "poly":
        cols.append(g1 ** 3)
    elif spec == "poly_bounded":
        lo, hi = g1_bound
        cols.append(np.clip(g1, lo, hi) ** 3)
    elif spec == "poly_mono":
        cols.append(mono_map(g1))
    else:
        raise ValueError(f"unknown spec {spec!r}")
    for j in range(1, d):
        cols.append(Z[:, j])
    X = np.column_stack(cols)
    if cp is not None:
        X = np.column_stack([X, cp])
    return X


def predict(Z, d, spec, train, test, ytr, cp=None):
    """Fit `spec` on train rows, predict test rows.  Handles the train-only
    bits of the diagnostic specs (g1 bounds, monotone map) internally."""
    kw = {}
    if spec == "poly_bounded":
        g1_tr = Z[train, 0]
        kw["g1_bound"] = (float(g1_tr.min()), float(g1_tr.max()))
    if spec == "poly_mono":
        kw["mono_map"] = _empirical_normal_map(Z[train, 0])
    M = design(Z, d, spec, cp=cp, **kw)
    return ols_predict(M[train], ytr, M[test])
