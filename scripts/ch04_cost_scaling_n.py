"""
cost_scaling_n.py — cost against the sample size, Section 4.5.1.

The counterpart of the `scaling` block of `sheng_yin_2016_study.py`, which holds
n fixed at 500 and varies the predictor dimension. Nothing else measures cost
against the sample size: `results_1d_baselines.csv` varies n but records angles
only. Read the ratio between the two solvers rather than the absolute seconds,
which belong to whichever machine produced the file.

The design is the p-scaling design with the two axes exchanged. Model A part (1)
of Sheng & Yin (2016) as restated by Wu & Chen (2021), the predictor dimension
held at p = 20, n in {100, 200, 500, 1000}, three replicates. Four estimators on
identical data, timed on one complete fit including whatever initialisation each
performs.

Why the two searches are expected to differ in n as well as in p: the gradient
of Chapter 3 is closed-form in beta but still needs the pairwise distance matrix
of the projected sample, which is O(n^2) in time and memory, so the gradient
search is O(n^2) per iteration. A finite-difference gradient pays that same
O(n^2) once per parameter, so the ratio between the two should be roughly flat
in n and set by p — which is the claim this run tests.

Three replicates rather than five because the SQP solver at (1000, 20) is the
binding cost; the runtime is documented as the reason, in the sense of Rule 5d.

Writes results_cost_scaling_n.csv (one row per size, replicate and method).
"""

# Thesis:   Chapter 4, §4.5.1
# Writes:   results/results_cost_scaling_n.csv
# Original: PP_Dcor/cost_scaling_n.py on the thesis branch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import sys
import time
from pathlib import Path

import numpy as np


from dpp.supervised.dcor_optimizer import dcor_u                    # noqa: E402
from dpp.supervised.evaluation import principal_angles              # noqa: E402
from dpp.supervised.pp_helpers import seq_pp                        # noqa: E402
from dpp.supervised.sdr_baselines import sir                        # noqa: E402
from dpp.supervised.sheng_yin import sheng_yin_sdr                  # noqa: E402
from designs.sheng_yin_2016 import make_data, ols      # noqa: E402

import warnings
warnings.filterwarnings("ignore")

#: Base of the replicate seeds, the same base the p-scaling block uses, offset
#: below so that no replicate here shares a stream with one there.
BASE_SEED = 20260813
STREAM_OFFSET = 100

#: The axis of this run. p is held at the largest value the chapter's other
#: experiments use.
SCALING_N = [100, 200, 500, 1000, 2000]
SCALING_P = 20
REPS_DEFAULT = 3

#: Budgets, unchanged from `sheng_yin_2016_study.py` so that the two curves are
#: measured on one configuration of both solvers.
N_PERTURB = 500
N_RESTARTS = 5
MAX_ITER = 150

D = 2          # model A has a two-dimensional central subspace


def _reps() -> int:
    """The replicate count.  A literal, so shortening it is a visible edit."""
    return REPS_DEFAULT


def _mean_angle(B_hat, B_true) -> float:
    p = len(B_true)
    return float(np.mean(principal_angles(
        np.asarray(B_hat, float).reshape(p, -1),
        np.asarray(B_true, float).reshape(p, -1))))


def _mean_dcor2(B_hat, X, Y) -> float:
    Z = X @ np.asarray(B_hat, float).reshape(X.shape[1], -1)
    return float(np.mean([dcor_u(Z[:, j], Y) for j in range(Z.shape[1])]))


def _score(rows, common, method, B_hat, X, Y, B_true, seconds):
    if B_hat is None:
        rows.append(dict(common, method=method, angle=float("nan"),
                         dcor2_u=float("nan"), seconds=round(seconds, 4)))
        return
    rows.append(dict(common, method=method,
                     angle=round(_mean_angle(B_hat, B_true), 4),
                     dcor2_u=round(_mean_dcor2(B_hat, X, Y), 6),
                     seconds=round(seconds, 4)))


def run():
    reps = _reps()
    rows = []
    t_start = time.time()
    for k, n in enumerate(SCALING_N):
        for rep in range(reps):
            rng = np.random.default_rng([BASE_SEED, STREAM_OFFSET + k, rep])
            X, Y, B_true = make_data("A", 1, n, SCALING_P, rng)
            common = dict(model="A", part=1, n=n, p=SCALING_P, d=D, rep=rep)

            t = time.time()
            W, _ = seq_pp(X, Y, D, deflation="X_deflation",
                          n_restarts=N_RESTARTS, max_iter=MAX_ITER, seed=rep)
            _score(rows, common, "dCor-SDR (this thesis)", W, X, Y, B_true,
                   time.time() - t)

            B_sy, info = sheng_yin_sdr(X, Y, d=D, n_perturb=N_PERTURB, seed=rep)
            _score(rows, common, "Sheng-Yin (SQP)", B_sy, X, Y, B_true,
                   info["seconds"])

            for method, fn in (("SIR", sir), ("OLS", ols)):
                t = time.time()
                try:
                    B = fn(X, Y, k=D)
                except Exception:                    # singular slice covariance
                    B = None
                _score(rows, common, method, B, X, Y, B_true, time.time() - t)

        print(f"  n={n:<5d} p={SCALING_P}  {reps} replicates done "
              f"[{time.time() - t_start:.0f}s]", flush=True)
    return rows, reps


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    # aggfunc is explicit: pivot_table defaults to the mean, which is the wrong
    # summary on three replicates (Rule 5d).
    for agg in ("median", "min", "max"):
        print(f"\n=== seconds per fit, {agg} over replicates ===")
        print(df.pivot_table(index="method", columns="n", values="seconds",
                             aggfunc=agg).round(3).to_string())
    print("\n=== mean principal angle (deg), median over replicates ===")
    print(df.pivot_table(index="method", columns="n", values="angle",
                         aggfunc="median").round(2).to_string())

    piv = df.pivot_table(index="method", columns="n", values="seconds",
                         aggfunc="median")
    print("\n=== ratio Sheng-Yin / dCor-SDR, median seconds ===")
    print((piv.loc["Sheng-Yin (SQP)"] / piv.loc["dCor-SDR (this thesis)"])
          .round(2).to_string())
    return df


if __name__ == "__main__":
    rows, reps = run()
    df = summarise(rows)
    out = RESULTS / "results_cost_scaling_n.csv"
    st = write_csv(
        out, df, seeds=(BASE_SEED,),
        script="ch04_cost_scaling_n.py",
        replicates=f"{reps} at each n",
        scaling_block=f"model A part 1, p={SCALING_P}, n in "
                      f"{{{', '.join(str(v) for v in SCALING_N)}}}",
        n_perturb=N_PERTURB, n_restarts=N_RESTARTS, max_iter=MAX_ITER,
        methods="dCor-SDR (this thesis), Sheng-Yin (SQP), SIR, OLS",
        design_source="wu2021mm pp.12-13, restating Sheng & Yin (2016)")
    print(f"\nWrote {out.name} ({len(df)} rows")
