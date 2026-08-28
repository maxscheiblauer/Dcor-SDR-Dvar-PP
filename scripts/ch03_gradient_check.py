"""Verification of the analytical gradients (Section 3.2.4).

Two independent checks, reported as one table and one figure.

1. Numerical.  Relative error of the analytical gradient against a central
   difference, for both indices, across sample sizes and dimensions, and as a
   function of the step size h.  The h-sweep is the informative part: it shows
   the characteristic V shape, truncation error falling as h shrinks until
   cancellation in the difference quotient takes over near h = 1e-5.  A formula
   that were merely close, rather than exact, would not reproduce that shape.

2. Structural.  The gradient-driven optimiser against two derivative-free
   solvers (Nelder-Mead, COBYLA) on the same objective: attained objective,
   angle between the returned directions, and wall-clock cost.

Run:  python scripts/ch03_gradient_check.py
"""

# Thesis:   Chapter 3, tab:se-gradcheck
# Writes:   results/results_gradcheck.csv
# Original: PP_Dcor/gradient_verification.py on the thesis branch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent
# PP_Dcor first: both packages define a module named data_generator, and the
# supervised one is the one wanted here.


from dpp.supervised.data_generator import data_generator                        # noqa: E402
from dpp.supervised.dcor_optimizer import _dcor_and_grad, _make_B, optimize_dcor  # noqa: E402
from dpp.unsupervised.dvar_optimizer import dvar_sq_and_grad_w                    # noqa: E402

SEED = 20260808          # the reference seed; sections 2 and 3 illustrate at this one

#: Five data seeds for the accuracy table of section 1. A gradient identity should hold
#: at every draw, so one seed was weak evidence for the claim the table supports.
SEEDS = (42, 7, 123, 2024, 5, 2026, 17, 99, 777, 31337)
NS = (200, 500)
PS = (5, 20, 50)


def _dcor_fun_grad(X, Y):
    """Value and Euclidean gradient of dCor(Y, Xw) as a function of w."""
    B, s_yy = _make_B(Y)

    def fg(w):
        return _dcor_and_grad(w, X, B, s_yy)

    return fg


def _dvar_fun_grad(X):
    """Value and Euclidean gradient of dVar^2(Xw) as a function of w."""

    def fg(w):
        return dvar_sq_and_grad_w(X, w)

    return fg


def central_difference(f, w, h):
    g = np.empty_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2.0 * h)
    return g


def rel_error(analytic, numeric):
    denom = max(np.linalg.norm(analytic), np.linalg.norm(numeric), 1e-300)
    return float(np.linalg.norm(analytic - numeric) / denom)


def check(index, n, p, h=1e-5, seed=SEED):
    rng = np.random.default_rng(seed)
    X, Y, _ = data_generator(n, p, 1, "cubic", 0.5, seed)
    w = rng.standard_normal(p)
    w /= np.linalg.norm(w)
    fg = _dcor_fun_grad(X, Y) if index == "dCor" else _dvar_fun_grad(X)
    _, grad = fg(w)
    return rel_error(grad, central_difference(lambda v: fg(v)[0], w, h))


def angle_deg(a, b):
    c = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, c))))


if __name__ == "__main__":

    print("=" * 62)
    print("1. Analytical gradient vs central difference")
    print("=" * 62)
    print("   Reported per configuration: the relative error at the best step size,")
    print("   and at the fixed h = 1e-5.  The two differ because the difference")
    print("   quotient carries its own error, which the analytical gradient does")
    print("   not.")
    print(f"   Median over {len(SEEDS)} seeds {list(SEEDS)}.")
    print()
    print(f"{'index':6s} {'n':>6s} {'p':>5s} {'best h':>9s} "
          f"{'error at best h':>17s} {'error at 1e-5':>15s}")
    print("-" * 62)
    HS_GRID = np.logspace(-8, -3, 11)
    rows = []
    worst_best = 0.0
    for index in ("dCor", "dVar"):
        for n in NS:
            for p in PS:
                best_errs, best_hs, fixed_errs = [], [], []
                for seed in SEEDS:
                    errs = [check(index, n, p, h=h, seed=seed) for h in HS_GRID]
                    i = int(np.argmin(errs))
                    best_errs.append(errs[i])
                    best_hs.append(HS_GRID[i])
                    fixed_errs.append(check(index, n, p, seed=seed))
                    rows.append(dict(index=index, seed=seed, n=n, p=p,
                                     best_h=float(HS_GRID[i]),
                                     err_at_best_h=float(errs[i]),
                                     err_at_1e5=float(fixed_errs[-1])))
                med_best = float(np.median(best_errs))
                worst_best = max(worst_best, max(best_errs))
                print(f"{index:6s} {n:6d} {p:5d} {np.median(best_hs):9.1e} "
                      f"{med_best:17.2e} {np.median(fixed_errs):15.2e}")
    print(f"\nlargest best-step error over all "
          f"{2 * len(NS) * len(PS)} configurations and {len(SEEDS)} seeds: "
          f"{worst_best:.2e}")
    write_csv(RESULTS / "results_gradcheck.csv", rows, seeds=SEEDS,
                         script="ch03_gradient_check.py")
    print("wrote results_gradcheck.csv")

    print()
    print("=" * 56)
    print("2. Step-size sweep (n = 200, p = 5)")
    print("=" * 56)
    hs = np.logspace(-9, -1, 17)
    curves = {}
    for index in ("dCor", "dVar"):
        curves[index] = [check(index, 200, 5, h=h) for h in hs]
        i = int(np.argmin(curves[index]))
        print(f"{index}: minimum relative error {curves[index][i]:.2e} "
              f"at h = {hs[i]:.1e}")

    print()
    print("=" * 56)
    print("3. Gradient optimiser vs derivative-free solvers")
    print("=" * 56)
    print(f"{'solver':14s} {'dCor':>8s} {'angle to grad':>15s} {'seconds':>9s}")
    print("-" * 50)
    X, Y, _ = data_generator(200, 5, 1, "cubic", 0.5, SEED)
    ref = None
    for name, opt in (("gradient", "gradient_ascent"),
                      ("Nelder-Mead", "nelder_mead"),
                      ("COBYLA", "cobyla")):
        t0 = time.time()
        beta, val, _ = optimize_dcor(X, Y, init_method="random", optimizer=opt,
                                     n_restarts=3, seed=SEED, max_iter=100)
        dt = time.time() - t0
        if ref is None:
            ref = beta
        print(f"{name:14s} {val:8.4f} {angle_deg(beta, ref):14.2f}  {dt:9.2f}")

