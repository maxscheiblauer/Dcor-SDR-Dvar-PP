"""
sheng_yin_comparison.py

Chapter 4 experiments re-run with Sheng & Yin's own dCov-SDR solver
(`sheng_yin.py`, ported from the MATLAB their group distributes) alongside this
project's Riemannian dCor optimiser, on identical data.

Two parts, mirroring the two Chapter 4 experiments:

  A. the single-index grid of `step3_experiments.py`
     (5 nonlinearities x 4 noise levels x 3 dimensions, n = 200);
  B. the multi-direction configurations of `step4_projection_pursuit.py`.

Every solution is scored under **both** conventions, because the two methods
optimise different functionals:

  * ``dcor2_u``  — the U-statistic dCor^2 of this thesis (`dcor_u`), which is
    what our optimiser maximises;
  * ``dcov_v``   — the V-statistic, unsquared distance covariance of
    `DistCorrVec.m`, which is what their `fmincon` maximises.

A method winning on its own criterion proves nothing; the comparison of
interest is the recovery angle, which neither method optimises.

Writes results_sheng_yin_ch4.csv (one row per configuration, seed and method).
"""

# Thesis:   Chapter 4, §4.7
# Writes:   results/results_sheng_yin_ch4.csv
# Original: PP_Dcor/sheng_yin_comparison.py on the thesis branch.
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
import itertools
from pathlib import Path

import numpy as np


from dpp.supervised.data_generator import data_generator                      # noqa: E402
from dpp.supervised.dcor_optimizer import optimize_dcor, dcor_u               # noqa: E402
from dpp.supervised.evaluation import _subspace_angle_1d, principal_angles    # noqa: E402
from dpp.supervised.pp_helpers import seq_pp                                  # noqa: E402
from dpp.supervised.joint_optimization import joint_optimize                  # noqa: E402
from dpp.supervised.sheng_yin import sheng_yin_sdr, dcov_v                    # noqa: E402

import warnings
warnings.filterwarnings("ignore")

SEEDS = (42, 7, 123, 2024, 5)

# Part A — the step3 grid, with that script's optimiser budget.
NONLINEARITIES = ['square', 'sine', 'abs', 'cubic', 'tanh']
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0]
DIMENSIONS = [2, 5, 20]
GRID_N = 200
N_RESTARTS = 3
MAX_ITER = 100

# Part B — the step4 configurations, with that script's budget.
MULTI = [
    (2, 5,  'product',      0.0, 400, 'k2_p5_product_noiseless'),
    (2, 5,  'product',      0.5, 400, 'k2_p5_product_noisy'),
    (2, 10, 'sum_squares',  0.0, 400, 'k2_p10_sumSq_noiseless'),
    (2, 10, 'sum_squares',  0.5, 400, 'k2_p10_sumSq_noisy'),
    (3, 10, 'sum_squares',  0.0, 500, 'k3_p10_sumSq_noiseless'),
    (2, 5,  'sine_product', 0.0, 400, 'k2_p5_sineProd_noiseless'),
    (2, 20, 'product',      0.0, 500, 'k2_p20_product_noiseless'),
]

#: Perturbation count of `dcsol.m`. Kept at their 500 so the initialisation is
#: theirs; the cost of it is part of what is being measured.
N_PERTURB = 500


def _score(W, X, Y):
    """Both criteria at a (p, k) solution: mean per-column dCor^2_u, and the
    V-statistic dCov of the whole k-dimensional projection (their objective)."""
    W = np.atleast_2d(W.T).T if W.ndim == 1 else W
    Z = X @ W
    dc = float(np.mean([dcor_u(Z[:, j], Y) for j in range(W.shape[1])]))
    return dc, dcov_v(Z, Y)


def run_single_index():
    rows = []
    configs = list(itertools.product(NONLINEARITIES, NOISE_LEVELS, DIMENSIONS))
    t0 = time.time()
    for seed in SEEDS:
        for nl, noise, p in configs:
            X, Y, W_true = data_generator(GRID_N, p, 1, nl, noise, seed)

            t = time.time()
            beta, _, _ = optimize_dcor(X, Y, init_method='random',
                                       optimizer='gradient_ascent',
                                       n_restarts=N_RESTARTS, seed=seed,
                                       max_iter=MAX_ITER)
            secs_ours = time.time() - t
            dc_o, dv_o = _score(beta[:, None], X, Y)
            rows.append(dict(part='single', seed=seed, config=f'{nl}_p{p}_s{noise}',
                             nonlinearity=nl, noise=noise, n=GRID_N, p=p, k=1,
                             method='dCor-PP (this thesis)',
                             angle=round(_subspace_angle_1d(beta, W_true), 4),
                             dcor2_u=round(dc_o, 6), dcov_v=round(dv_o, 6),
                             seconds=round(secs_ours, 3)))

            B, info = sheng_yin_sdr(X, Y, d=1, n_perturb=N_PERTURB, seed=seed)
            b = B[:, 0] / np.linalg.norm(B[:, 0])
            dc_s, dv_s = _score(b[:, None], X, Y)
            rows.append(dict(part='single', seed=seed, config=f'{nl}_p{p}_s{noise}',
                             nonlinearity=nl, noise=noise, n=GRID_N, p=p, k=1,
                             method='Sheng-Yin (dcsol)',
                             angle=round(_subspace_angle_1d(b, W_true), 4),
                             dcor2_u=round(dc_s, 6), dcov_v=round(dv_s, 6),
                             seconds=round(info['seconds'], 3)))
        print(f"  [A] seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
    return rows


def run_multi():
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        for (k, p, nl, noise, n, label) in MULTI:
            X, Y, W_true = data_generator(n, p, k, nl, noise, seed=seed)

            t = time.time()
            W_seq, _ = seq_pp(X, Y, k, deflation='X_deflation',
                              n_restarts=5, max_iter=150, seed=0)
            secs = time.time() - t
            dc, dv = _score(W_seq, X, Y)
            rows.append(dict(part='multi', seed=seed, config=label,
                             nonlinearity=nl, noise=noise, n=n, p=p, k=k,
                             method='dCor-PP sequential (X-deflation)',
                             angle=round(float(principal_angles(W_seq, W_true).mean()), 4),
                             dcor2_u=round(dc, 6), dcov_v=round(dv, 6),
                             seconds=round(secs, 3)))

            t = time.time()
            W_j, _, _ = joint_optimize(X, Y, k=k, lam=0.0, n_restarts=4,
                                       max_iter=100, seed=seed)
            secs = time.time() - t
            dc, dv = _score(W_j, X, Y)
            rows.append(dict(part='multi', seed=seed, config=label,
                             nonlinearity=nl, noise=noise, n=n, p=p, k=k,
                             method='dCor-PP joint (Stiefel, lam=0)',
                             angle=round(float(principal_angles(W_j, W_true).mean()), 4),
                             dcor2_u=round(dc, 6), dcov_v=round(dv, 6),
                             seconds=round(secs, 3)))

            B, info = sheng_yin_sdr(X, Y, d=k, n_perturb=N_PERTURB, seed=seed)
            dc, dv = _score(B, X, Y)
            rows.append(dict(part='multi', seed=seed, config=label,
                             nonlinearity=nl, noise=noise, n=n, p=p, k=k,
                             method='Sheng-Yin (dcsol)',
                             angle=round(float(principal_angles(B, W_true).mean()), 4),
                             dcor2_u=round(dc, 6), dcov_v=round(dv, 6),
                             seconds=round(info['seconds'], 3)))
            print(f"  [B] seed {seed:5d} {label:<28} ({time.time()-t0:.0f}s)",
                  flush=True)
    return rows


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    for part in ('single', 'multi'):
        d = df[df.part == part]
        if d.empty:
            continue
        print(f"\n=== {part}: median over {len(SEEDS)} seeds, [min, max] on the angle ===")
        piv = d.pivot_table(index='config', columns='method',
                            values='angle', aggfunc='median')
        lo = d.pivot_table(index='config', columns='method',
                           values='angle', aggfunc='min')
        hi = d.pivot_table(index='config', columns='method',
                           values='angle', aggfunc='max')
        for cfg in piv.index:
            parts = "  ".join(
                f"{m}: {piv.loc[cfg, m]:6.2f} [{lo.loc[cfg, m]:5.2f},{hi.loc[cfg, m]:6.2f}]"
                for m in piv.columns)
            print(f"  {cfg:<28} {parts}")
        print("  medians — angle / dCor^2_u / dCov_v / seconds")
        for m, g in d.groupby('method'):
            print(f"    {m:<34} {g.angle.median():6.2f}  {g.dcor2_u.median():.4f}  "
                  f"{g.dcov_v.median():.4f}  {g.seconds.median():6.2f}s")
    return df


if __name__ == '__main__':
    rows = run_single_index() + run_multi()
    df = summarise(rows)
    out = RESULTS / 'results_sheng_yin_ch4.csv'
    stamp = write_csv(out, df, seeds=SEEDS,
                                 script='ch04_shengyin_on_our_grid.py',
                                 n_perturb=N_PERTURB, n_restarts=N_RESTARTS,
                                 max_iter=MAX_ITER)
    print(f"\nWrote {out.name} ({len(df)} rows")
