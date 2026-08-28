"""
robustness.py — Step 7. S3 outlier scenario, varying outlier_frac.

Compare on n=200, p=50, k=2, sigma_signal=2.0, gaussian_mix:
    - dVar-PP plain (analytical gradient, fast)
    - dVar-PP robust (biloop transform; central-difference gradient, slow)
    - PCA
    - FastICA

Vary outlier_frac in {0.0, 0.05, 0.1, 0.2}, outlier_magnitude=10.

Tests the R-header claim: biloop helps under contamination, hurts on
clean data.
"""

# Thesis:   Chapter 5, fig:p2-robustness
# Writes:   results/results_robust.csv, results/results_robust_agg.csv
# Original: Dvar-PP/robustness.py on the thesis branch.
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
import pandas as pd

from dpp.unsupervised.data_generator import generate_data
from dpp.unsupervised.dvar_optimizer import pp_dvar
from dpp.unsupervised.evaluation import (mss_principal, pca_directions, fastica_directions,
                        orthogonality_defect)



CFG = dict(n=150, p=30, k=2, sigma_signal=2.5, dist="gaussian_mix")
FRACS = [0.0, 0.05, 0.1, 0.2]
N_STARTS = 4                   # robust path is slow; few starts are enough
MAX_ITER_PLAIN = 120
MAX_ITER_ROBUST = 40           # robust landscape flattens; long runs don't help
#: Five replicates per configuration. Was three until 2026-08-11; extended to match
#: the seed list used across the project, since a claim from few replicates is what
#: finding F2 of thesis/REVISION_PLAN_2026-08-11.md is about. The run resumes from
#: results_robust.csv, so the two added seeds cost only their own runtime.
SEEDS = [42, 7, 123, 2024, 5]


def run_one(frac, seed):
    dat = generate_data(seed=seed, outlier_frac=frac, outlier_magnitude=10.0,
                        **CFG)
    X, W_true = dat["X"], dat["W_true"]
    k = CFG["k"]

    # Plain
    t0 = time.time()
    fit_p = pp_dvar(X, k=k, whiten=False, strategy="sequential",
                    index_fun="plain",
                    n_starts=N_STARTS, max_iter=MAX_ITER_PLAIN,
                    lr=0.05, n_jobs=2, seed=seed)
    t_p = time.time() - t0

    # Robust (biloop)
    t0 = time.time()
    fit_r = pp_dvar(X, k=k, whiten=False, strategy="sequential",
                    index_fun="robust",
                    n_starts=N_STARTS, max_iter=MAX_ITER_ROBUST,
                    lr=0.05, n_jobs=2, seed=seed)
    t_r = time.time() - t0

    W_pca = pca_directions(X, k)
    W_ica = fastica_directions(X, k, seed=seed)

    return dict(
        frac=frac, seed=seed,
        MSS_plain=mss_principal(W_true, fit_p["W"]),
        MSS_robust=mss_principal(W_true, fit_r["W"]),
        MSS_PCA=mss_principal(W_true, W_pca),
        MSS_FastICA=mss_principal(W_true, W_ica),
        time_plain=t_p, time_robust=t_r,
    )


def main():
    csv = RESULTS / "results_robust.csv"
    # No resume path. It existed once, read the partial CSV back and skipped
    # the rows already present; a run that spanned a code change then wrote one
    # file whose rows came from two different code states. Recomputing from
    # scratch is cheap enough (about four minutes) to not be worth that risk.
    rows = []

    plan = [(f, s) for f in FRACS for s in SEEDS]
    for frac, seed in plan:
        r = run_one(frac, seed)
        rows.append(r)
        write_csv(csv, rows, seeds=SEEDS, n_starts=N_STARTS,
                             script="ch05_robustness.py")
        print(f"  frac={frac:.2f} seed={seed:3d}  "
              f"MSS plain={r['MSS_plain']:.3f} robust={r['MSS_robust']:.3f} "
              f"PCA={r['MSS_PCA']:.3f} ICA={r['MSS_FastICA']:.3f}  "
              f"t_p={r['time_plain']:.1f}s t_r={r['time_robust']:.1f}s",
              flush=True)

    df = pd.DataFrame(rows)
    # Median and range are reported beside mean and sd: on five replicates with one
    # possible outlier the sd is misleading, and how the thesis presents multi-seed
    # results is still open (decision G6 of the revision plan). Both are in the file
    # so that decision does not require another run.
    stats = ["mean", "std", "median", "min", "max"]
    cols = ["MSS_plain", "MSS_robust", "MSS_PCA", "MSS_FastICA"]
    agg = df.groupby("frac").agg({c: stats for c in cols}).round(4)
    print(f"\nAcross {df['seed'].nunique()} seeds per outlier fraction:")
    print(agg)
    flat = agg.copy()
    flat.columns = [f"{a}_{b}" for a, b in flat.columns]
    write_csv(RESULTS / "results_robust_agg.csv", flat.reset_index(),
                         seeds=sorted(df["seed"].unique().tolist()),
                         script="ch05_robustness.py")



if __name__ == "__main__":
    main()
