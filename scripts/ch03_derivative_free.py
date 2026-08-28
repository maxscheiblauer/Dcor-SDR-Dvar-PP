"""Derivative-free comparison for Section 3.2.4 (supervisor request, p. 17).

The thesis previously reported a single configuration.  This script runs the
gradient optimiser against two derivative-free solvers, Nelder-Mead and COBYLA,
across sample sizes and predictor dimensions, for BOTH indices:

  supervised    dCor^2(Y, Xw)   maximised by `optimize_dcor`
  unsupervised  dVar(Xw)        maximised by `optimise_one_direction`

For every configuration the three solvers start from the same random directions
and get the same restart budget, so the only thing that differs is how the
search direction is obtained.  Reported per configuration:

  * the attained objective of each solver (dCor^2 on the supervised side,
    dVar on the unsupervised side -- the U-statistic / double-centred
    conventions of the thesis, unchanged);
  * the angle between each derivative-free solution and the gradient solution;
  * wall-clock seconds, and the ratio to the gradient solver.

Writes results_derivative_free.csv.

Run:  python scripts/ch03_derivative_free.py
"""

# Thesis:   Chapter 3, tab:se-dfree
# Writes:   results/results_derivative_free.csv
# Original: PP_Dcor/derivative_free_study.py on the thesis branch.
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
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent


from dpp.supervised.data_generator import data_generator                        # noqa: E402
from dpp.supervised.dcor_optimizer import optimize_dcor                         # noqa: E402

import importlib                                                 # noqa: E402
_dvar_dg = importlib.import_module("Dvar-PP.data_generator") if False else None
from dpp.unsupervised.dvar_optimizer import dvar, optimise_one_direction          # noqa: E402

#: Three data seeds, not the project-wide five. The derivative-free solvers are given
#: twenty times the gradient solver's iteration budget by design, which costs about
#: thirty minutes per seed; the rerun inventory of thesis/REVISION_PLAN_2026-08-11.md
#: allows three to five here for that reason. Do not cut DF_ITER instead — an earlier
#: run at equal budget produced a false finding (both solvers appearing to fail above
#: p = 5). The first seed is the original single seed, so its values stay identifiable.
SEEDS = (20260808, 7, 123)
NS = (200, 500)
PS = (5, 20, 50)
RESTARTS = 3
MAX_ITER = 100
# Derivative-free solvers get a far larger iteration budget than the gradient
# solver, so that a failure cannot be blamed on the budget.  Reported cost
# ratios are measured with this budget in place.
DF_ITER = 2000


def angle_deg(a, b):
    c = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, c))))


def _starts(rng, p, r):
    W = rng.standard_normal((r, p))
    return W / np.linalg.norm(W, axis=1, keepdims=True)


# --------------------------------------------------------------- unsupervised
def _dvar_obj(w, X):
    nw = np.linalg.norm(w)
    if nw < 1e-12:
        return 0.0
    return float(dvar(X @ (w / nw)))


def dvar_derivative_free(X, starts, method, max_iter=DF_ITER):
    """Maximise dVar(Xw) from each start with a derivative-free solver."""
    best_w, best_val = None, -np.inf
    for w0 in starts:
        if method == "nelder_mead":
            res = minimize(lambda w: -_dvar_obj(w, X), w0, method="Nelder-Mead",
                           options=dict(maxiter=max_iter * len(w0), xatol=1e-6,
                                        fatol=1e-8))
        else:
            res = minimize(lambda w: -_dvar_obj(w, X), w0, method="COBYLA",
                           constraints=[dict(
                               type="ineq",
                               fun=lambda w: 1.0 - abs(np.linalg.norm(w) - 1.0))],
                           options=dict(maxiter=max_iter * len(w0), rhobeg=0.2))
        w = res.x / np.linalg.norm(res.x)
        v = float(dvar(X @ w))
        if v > best_val:
            best_w, best_val = w, v
    return best_w, best_val


def load_dvar_generator():
    """The factor-model generator of the unsupervised side.

    This used to be loaded from its file by path, because both pillars had a module
    named ``data_generator`` sitting in sibling directories and a plain import would
    have picked up whichever came first on the path.  The two are separate
    subpackages here, so the clash is gone and the import is ordinary.
    """
    from dpp.unsupervised.data_generator import generate_data
    return generate_data


if __name__ == "__main__":
    generate_data = load_dvar_generator()
    rows = []

    print("=" * 78)
    print("Supervised index: dCor^2(Y, Xw), response `cubic`, noise 0.5")
    print("=" * 78)
    print(f"{'n':>5s} {'p':>4s} {'solver':13s} {'dCor^2':>9s} "
          f"{'angle to grad':>14s} {'sec':>8s} {'x grad':>7s}")
    print("-" * 78)
    for seed in SEEDS:
        for n in NS:
            for p in PS:
                X, Y, _ = data_generator(n, p, 1, "cubic", 0.5, seed)
                ref_w, ref_t = None, None
                for name, opt in (("gradient", "gradient_ascent"),
                                  ("Nelder-Mead", "nelder_mead"),
                                  ("COBYLA", "cobyla")):
                    budget = MAX_ITER if opt == "gradient_ascent" else DF_ITER
                    t0 = time.time()
                    w, val, _ = optimize_dcor(X, Y, init_method="random",
                                              optimizer=opt, n_restarts=RESTARTS,
                                              seed=seed, max_iter=budget)
                    dt = time.time() - t0
                    if ref_w is None:
                        ref_w, ref_t = w, dt
                    ang = angle_deg(w, ref_w)
                    rows.append(dict(index="dCor2", seed=seed, n=n, p=p, solver=name,
                                     objective=val, angle_to_gradient=ang,
                                     seconds=dt, cost_ratio=dt / ref_t))
                    print(f"{n:5d} {p:4d} {name:13s} {val:9.4f} {ang:13.2f}  "
                          f"{dt:8.2f} {dt / ref_t:7.1f}   seed {seed}")
    print()

    print("=" * 78)
    print("Unsupervised index: dVar(Xw), factor model, gaussian_mix, k=1")
    print("=" * 78)
    print(f"{'n':>5s} {'p':>4s} {'solver':13s} {'dVar':>9s} "
          f"{'angle to grad':>14s} {'sec':>8s} {'x grad':>7s}")
    print("-" * 78)
    for seed in SEEDS:
        for n in NS:
            for p in PS:
                d = generate_data(n=n, p=p, k=1, sigma_signal=2.0,
                                  dist="gaussian_mix", seed=seed)
                X = d["X"]
                rng = np.random.default_rng(seed)
                starts = _starts(rng, p, RESTARTS)

                t0 = time.time()
                out = optimise_one_direction(X, n_starts=RESTARTS,
                                             max_iter=MAX_ITER, seed=seed,
                                             init_dirs=starts.T, n_jobs=1)
                ref_t = time.time() - t0
                ref_w, ref_val = out["w"], out["obj"]
                rows.append(dict(index="dVar", seed=seed, n=n, p=p,
                                 solver="gradient", objective=ref_val,
                                 angle_to_gradient=0.0, seconds=ref_t,
                                 cost_ratio=1.0))
                print(f"{n:5d} {p:4d} {'gradient':13s} {ref_val:9.4f} "
                      f"{0.0:13.2f}  {ref_t:8.2f} {1.0:7.1f}   seed {seed}")

                for name, method in (("Nelder-Mead", "nelder_mead"),
                                     ("COBYLA", "cobyla")):
                    t0 = time.time()
                    w, val = dvar_derivative_free(X, starts, method)
                    dt = time.time() - t0
                    ang = angle_deg(w, ref_w)
                    rows.append(dict(index="dVar", seed=seed, n=n, p=p, solver=name,
                                     objective=val, angle_to_gradient=ang,
                                     seconds=dt, cost_ratio=dt / ref_t))
                    print(f"{n:5d} {p:4d} {name:13s} {val:9.4f} {ang:13.2f}  "
                          f"{dt:8.2f} {dt / ref_t:7.1f}   seed {seed}")

    df = pd.DataFrame(rows)
    write_csv(RESULTS / "results_derivative_free.csv", df, seeds=SEEDS,
                         script="ch03_derivative_free.py",
                         n_restarts=RESTARTS, gradient_max_iter=MAX_ITER,
                         derivative_free_max_iter=DF_ITER)
    print(f"\nwrote {_HERE / 'results_derivative_free.csv'}")

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"(medians over {len(SEEDS)} seeds {list(SEEDS)}; the CSV keeps every "
          f"replicate)")
    for index in ("dCor2", "dVar"):
        sub = df[df["index"] == index]
        piv = sub.pivot_table(index=["n", "p"], columns="solver",
                              values="objective", aggfunc="median")
        gap = (piv["gradient"] - piv[["Nelder-Mead", "COBYLA"]].max(axis=1))
        ang = sub[sub.solver != "gradient"]["angle_to_gradient"]
        cost = sub[sub.solver != "gradient"]["cost_ratio"]
        print(f"{index}: gradient objective minus best derivative-free, "
              f"worst case {gap.max():+.4f}, best case {gap.min():+.4f}")
        print(f"       angle to gradient solution: median {ang.median():.2f} deg, "
              f"max {ang.max():.2f} deg")
        print(f"       cost relative to gradient: median {cost.median():.1f}x, "
              f"max {cost.max():.1f}x")
