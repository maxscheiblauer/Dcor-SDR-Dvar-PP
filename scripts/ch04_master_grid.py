"""unified_grid.py — the Chapter 4 master grid.

All six presentation-set models × 4 methods × 2 sizes × 5 seeds, under normal
predictors (see PARTS).  Single-direction and multi-direction results in one run, constant
sample size n = 500 throughout.

Single-direction grid (all 6 models):
    square, cubic       — k = 1
    A, C                — k = 1 (C) and k = 2 (A), but scored as single-dir
    B                   — k = 2, scored as single-dir
    sum_squares         — k = 2, scored as single-dir

Multi-direction grid (4 models with k >= 2):
    A, B, sum_squares, product — k = 2
    Uses X-deflation (sequential) for dCor-SDR.

Predictor distributions:
    part 1: N(0, I_p)                    — all models
    part 2: Uniform(-2, 2)               — all models (model B's distribution)
    part 3: Poisson(1)                   — all models (model A's distribution)

    For models A, B, C the Sheng & Yin paper specifies model-specific distributions
    in parts 2 and 3.  This grid uses a single distribution per part for all six
    models so that the predictor effect is comparable across rows; the model-specific
    distributions are reported separately in sheng_yin_2016_study.py.

Writes: results_unified_grid.csv
"""

# Thesis:   Chapter 4, the master grid
# Writes:   results/results_unified_grid.csv
# Original: PP_Dcor/unified_grid.py on the thesis branch.
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

import numpy as np
import pandas as pd


from dpp.supervised.dcor_optimizer import dcor_u
from dpp.supervised.evaluation import principal_angles
from dpp.supervised.joint_optimization import joint_optimize
from dpp.supervised.pp_helpers import seq_pp
from dpp.supervised.sdr_baselines import sir
from dpp.supervised.sheng_yin import sheng_yin_sdr


# ── configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 7, 123, 2024, 5]
SIZES = [(500, 6), (500, 20)]
N_RESTARTS = 5
#: The joint Stiefel search gets a larger restart budget than the sequential one,
#: fixed a priori by the dimension of the space each searches: the sphere S^{p-1}
#: has p - 1 dimensions, St(p,k) has pk - k(k+1)/2, and sequential extraction
#: already spends N_RESTARTS ascents per direction where the joint search spends
#: N_RESTARTS in total.  Measured 2026-08-17: at 20 restarts the joint fit costs
#: 6.8 s at p = 20 against 13.0 s for the Sheng-Yin solver, so the raised budget
#: stays inside their per-fit cost.
JOINT_RESTARTS = 20
N_PERTURB = 500
MAX_ITER = 150
NOISE_SD = 0.5           # additive noise for self-chosen models
#: Predictor laws to run.  The uniform and Poisson blocks were dropped from the
#: chapter on 2026-08-17: uniform is symmetric like the normal, so it reproduces
#: the normal picture without adding to it, and the claim that the objective
#: assumes no particular predictor law is made from its definition rather than
#: from a grid.  Set PARTS = (1, 2, 3) to restore them.
PARTS = (1,)

# ── predictor generation ─────────────────────────────────────────────────────

def _predictors(part, n, p, rng):
    """Uniform predictor distribution across all models.

    part 1: N(0, I_p)
    part 2: Uniform(-2, 2)  — the distribution Sheng & Yin use for model B
    part 3: Poisson(1)      — the distribution Sheng & Yin use for model A
    """
    if part == 1:
        return rng.standard_normal((n, p))
    if part == 2:
        return rng.uniform(-2.0, 2.0, size=(n, p))
    if part == 3:
        return rng.poisson(1.0, size=(n, p)).astype(float)
    raise ValueError(f"unknown part: {part}")


# ── response functions ───────────────────────────────────────────────────────

def _betas(p):
    """beta1, beta2, beta3 with the Sheng & Yin first-six convention."""
    b1 = np.zeros(p); b1[0] = 1.0
    b2 = np.zeros(p); b2[1] = 1.0
    b3 = np.zeros(p); b3[0] = 1.0; b3[1] = 0.5; b3[2] = 1.0
    return b1, b2, b3


# Registry: model -> (k, response_fn(X, rng) -> Y, B_true)
def _make_data(model, X, p, rng):
    """Return Y, B_true for the given model and predictor matrix X."""
    b1, b2, b3 = _betas(p)
    n = X.shape[0]

    if model == 'A':
        Y = (X @ b1) ** 2 + (X @ b2) + 0.1 * rng.standard_normal(n)
        return Y, np.column_stack([b1, b2])
    if model == 'B':
        e1 = rng.standard_normal(n)
        e2 = rng.standard_normal(n)
        Y = np.sign(2.0 * (X @ b1) + e1) * np.log(np.abs(2.0 * (X @ b2) + 4.0 + e2))
        return Y, np.column_stack([b1, b2])
    if model == 'C':
        Y = np.exp(X @ b3) * rng.standard_normal(n)
        return Y, b3.reshape(-1, 1)
    if model == 'square':
        Z1 = X[:, 0]
        Y = Z1 ** 2 + NOISE_SD * rng.standard_normal(n)
        return Y, np.eye(p, 1)
    if model == 'cubic':
        Z1 = X[:, 0]
        Y = Z1 ** 3 + NOISE_SD * rng.standard_normal(n)
        return Y, np.eye(p, 1)
    if model == 'sum_squares':
        k = 2
        Z = X[:, :k]
        Y = np.sum(Z ** 2, axis=1) + NOISE_SD * rng.standard_normal(n)
        return Y, np.eye(p, k)
    if model == 'product':
        k = 2
        Y = X[:, 0] * X[:, 1] + NOISE_SD * rng.standard_normal(n)
        return Y, np.eye(p, k)
    raise ValueError(f"unknown model: {model}")


MODEL_K = {
    'A': 2, 'B': 2, 'C': 1,
    'square': 1, 'cubic': 1,
    'sum_squares': 2, 'product': 2,
}

# Which models go into which grid
SINGLE_DIR_MODELS = ['A', 'B', 'C', 'square', 'cubic', 'sum_squares']
MULTI_DIR_MODELS  = ['A', 'B', 'sum_squares', 'product']


# ── scoring ──────────────────────────────────────────────────────────────────

def _as_matrix(B, p):
    return np.asarray(B, float).reshape(p, -1)


def mean_angle(B_hat, B_true):
    p = len(B_true)
    return float(np.mean(principal_angles(
        _as_matrix(B_hat, p), _as_matrix(B_true, p))))


def max_angle(B_hat, B_true):
    p = len(B_true)
    return float(np.max(principal_angles(
        _as_matrix(B_hat, p), _as_matrix(B_true, p))))


def mean_dcor2(B_hat, X, Y):
    Z = X @ _as_matrix(B_hat, X.shape[1])
    return float(np.mean([dcor_u(Z[:, j], Y) for j in range(Z.shape[1])]))


def ols(X, Y, k=1):
    b, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(Y, float), rcond=None)
    nrm = np.linalg.norm(b)
    return (b / nrm if nrm > 1e-12 else b).reshape(-1, 1)


def _score(rows, common, method, B_hat, X, Y, B_true, seconds):
    if B_hat is None:
        rows.append(dict(common, method=method, angle_mean=float('nan'),
                         angle_max=float('nan'), dcor2_u=float('nan'),
                         seconds=round(seconds, 3)))
        return
    rows.append(dict(common, method=method,
                     angle_mean=round(mean_angle(B_hat, B_true), 4),
                     angle_max=round(max_angle(B_hat, B_true), 4),
                     dcor2_u=round(mean_dcor2(B_hat, X, Y), 6),
                     seconds=round(seconds, 3)))


def _run_four_methods(rows, common, X, Y, B_true, k, seed):
    """Score all four estimators on one data set."""
    # dCor-SDR
    t = time.time()
    W, _ = seq_pp(X, Y, k, deflation='X_deflation',
                  n_restarts=N_RESTARTS, max_iter=MAX_ITER, seed=seed)
    _score(rows, common, 'dCor-SDR', W, X, Y, B_true, time.time() - t)

    # Sheng-Yin SQP
    B_sy, info = sheng_yin_sdr(X, Y, d=k, n_perturb=N_PERTURB, seed=seed)
    _score(rows, common, 'Sheng-Yin', B_sy, X, Y, B_true, info['seconds'])

    # SIR
    t = time.time()
    try:
        B_sir = sir(X, Y, k=k)
    except Exception:
        B_sir = None
    _score(rows, common, 'SIR', B_sir, X, Y, B_true, time.time() - t)

    # OLS
    t = time.time()
    B_ols = ols(X, Y, k=k)
    _score(rows, common, 'OLS', B_ols, X, Y, B_true, time.time() - t)


# ── main ─────────────────────────────────────────────────────────────────────

def run():
    rows = []
    t_start = time.time()
    config_id = 0

    # ── single-direction grid ────────────────────────────────────────────────
    for model in SINGLE_DIR_MODELS:
        k = MODEL_K[model]
        for part in (1, 2, 3):
            if part not in PARTS:
                # keep config_id in step with the full three-part design, so that
                # the data streams are identical to a run with every part enabled
                config_id += len(SIZES)
                continue
            for n, p in SIZES:
                for si, seed in enumerate(SEEDS):
                    rng = np.random.default_rng([seed, config_id])
                    X = _predictors(part, n, p, rng)
                    Y, B_true = _make_data(model, X, p, rng)
                    common = dict(grid='single', model=model, k=k,
                                  part=part, n=n, p=p, seed=seed)
                    _run_four_methods(rows, common, X, Y, B_true, k, seed)
                config_id += 1
                print(f"  [single] {model} part {part}  (n,p)=({n},{p})  "
                      f"5 seeds done  [{time.time() - t_start:.0f}s]", flush=True)

    # ── multi-direction grid ─────────────────────────────────────────────────
    for model in MULTI_DIR_MODELS:
        k = MODEL_K[model]
        assert k >= 2, f"{model} has k={k}, need k >= 2 for multi-direction"
        for part in (1, 2, 3):
            if part not in PARTS:
                # keep config_id in step with the full three-part design, so that
                # the data streams are identical to a run with every part enabled
                config_id += len(SIZES)
                continue
            for n, p in SIZES:
                for si, seed in enumerate(SEEDS):
                    rng = np.random.default_rng([seed, config_id])
                    X = _predictors(part, n, p, rng)
                    Y, B_true = _make_data(model, X, p, rng)
                    common = dict(grid='multi', model=model, k=k,
                                  part=part, n=n, p=p, seed=seed)
                    _run_four_methods(rows, common, X, Y, B_true, k, seed)
                    # Joint Stiefel optimizer (lambda=0)
                    t = time.time()
                    B_joint, _, _ = joint_optimize(
                        X, Y, k, lam=0.0, n_restarts=JOINT_RESTARTS,
                        max_iter=MAX_ITER, seed=seed)
                    _score(rows, common, 'dCor-SDR (joint)',
                           B_joint, X, Y, B_true, time.time() - t)
                config_id += 1
                print(f"  [multi]  {model} part {part}  (n,p)=({n},{p})  "
                      f"5 seeds done  [{time.time() - t_start:.0f}s]", flush=True)

    return rows


def summarise(rows):
    df = pd.DataFrame(rows)

    for grid_name in ('single', 'multi'):
        sub = df[df.grid == grid_name]
        if sub.empty:
            continue
        print(f"\n{'='*60}")
        print(f"  {grid_name}-direction grid: median angle (deg) over 5 seeds")
        print(f"{'='*60}")
        for (model, part, n, p), g in sub.groupby(
                ['model', 'part', 'n', 'p'], sort=True):
            entries = []
            for method in ['OLS', 'SIR', 'Sheng-Yin', 'dCor-SDR']:
                h = g[g.method == method]
                if h.empty:
                    entries.append(f"{method}: --")
                    continue
                med = h.angle_mean.median()
                mx = h.angle_mean.max()
                entries.append(f"{method}: {med:.1f} [{mx:.1f}]")
            print(f"  {model} part {part}  (n,p)=({n},{p})  " +
                  " | ".join(entries))


def main():
    print(f"seeds: {SEEDS}")
    print(f"sizes: {SIZES}")
    print(f"single-direction models: {SINGLE_DIR_MODELS}")
    print(f"multi-direction models:  {MULTI_DIR_MODELS}")
    print(f"predictor parts: {PARTS}  (1 = normal, 2 = uniform, 3 = poisson)")
    print(f"restarts: {N_RESTARTS} sequential, {JOINT_RESTARTS} joint")
    print()

    rows = run()
    df = pd.DataFrame(rows)

    out = RESULTS / 'results_unified_grid.csv'
    write_csv(
        out, df, seeds=SEEDS, script="ch04_master_grid.py",
        sizes=str(SIZES), noise_sd=NOISE_SD,
        n_restarts=N_RESTARTS, joint_restarts=JOINT_RESTARTS,
        parts=str(PARTS), n_perturb=N_PERTURB,
        max_iter=MAX_ITER,
        single_models=str(SINGLE_DIR_MODELS),
        multi_models=str(MULTI_DIR_MODELS),
    )
    print(f"\nwrote {out}  ({len(df)} rows)")

    summarise(rows)


if __name__ == '__main__':
    main()
