"""
fair_comparison.py

Two questions the Chapter 4 comparison against Sheng & Yin's solver leaves open,
both of them about fairness rather than about which method is better.

Question 1 — does their solver depend on its inverse-regression seed?
--------------------------------------------------------------------
`init_ablation.py` runs one-directionally: it grants whitening, an SIR seed and a
larger restart budget to *this project's* optimiser and finds that none of them
closes the accuracy gap at p = 20. That establishes nothing about their solver.
Their published pipeline always starts from the better of SIR and SAVE and then
scores 500 orthonormalised perturbations of that seed before the SQP solve, and
in every experiment in this repository it was run that way. So the sentence "the
difference is not in where the search starts" is supported for one solver only.

This script removes the seed from *their* side. `sheng_yin_sdr` already supports
it: `n_perturb=0` reproduces `dcsol2.m`, and `v0` bypasses the initialisation
entirely (`dcsol1.m`). No change to `sheng_yin.py` is needed or made — editing it
would invalidate `results_sheng_yin_2016.csv` and `results_sheng_yin_ch4.csv`.

Question 2 — who maximises dCor better, holding the objective fixed?
--------------------------------------------------------------------
The published comparison scores two solvers that maximise *different* things:
theirs the V-statistic (biased) distance covariance, unsquared and scaled by 10
(`dcreg.m`); this project's the U-statistic dCor^2. A solver cannot be blamed for
losing on a criterion it never optimised, and the chapter's "87% of
configurations" figure is a comparison of solutions, not of optimisers.

The `sqp-dcor2` arm below removes that confound: their *optimiser* — SLSQP with a
finite-difference gradient, run on whitened predictors under the constraint
V'V = I_d, at their tolerance and iteration budget — driving *this project's*
objective, the marginal sum

    S(V) = sum_j dCor^2_u(Z v_j, y),        Z = (X - x_bar) Sigma^{-1/2},

which at lambda = 0 is exactly eq. (4.20). Its partner arm `adam-dcor2` is
`joint_optimize` on the same Z, the same constraint (B'B = I_k, so
beta_i' Sigma beta_j = 0 in the original scale) and the same objective, from the
same kind of random start. Those two arms differ in the optimiser and in nothing
else, which is the comparison the closed-form gradient is supposed to win.

One deliberate deviation. The SQP arm sees the *unclamped* ratio
s_xy / sqrt(s_xx s_yy), whereas `dcor_u` clamps a non-positive s_xy to zero. The
clamp would hand SLSQP a flat region and stop it at its starting point; Adam does
not suffer from it because `_dcor_and_grad` supplies a surrogate ascent direction
there instead. The two functions agree wherever the clamp is inactive, which is
every point either optimiser reports as a solution.

Arms
----
On their objective (V-statistic dCov, their SLSQP solve):

    sy-full         SIR/SAVE seed + 500 perturbations   (their published default)
    sy-seed         SIR/SAVE seed, no perturbations     (`dcsol2.m`)
    sy-rand1        one random orthonormal start        (`dcsol1.m`)
    sy-rand5        best of five random starts, scored by their own objective
    sy-rand500      random seed + 500 perturbations of it

    sy-full vs sy-seed isolates the perturbation stage; sy-seed vs sy-rand1
    isolates the SIR seed; sy-rand500 vs sy-full isolates the seed with the
    perturbation budget held fixed.

On this project's objective (U-statistic dCor^2):

    pp-rand1        sequential X-deflation from a *single* random start — the
                    like-for-like partner of sy-rand1, one start on each side
    pp-default      what the thesis reports: sequential X-deflation, 5 random
                    restarts, Riemannian Adam in the original coordinates
    pp-probe500     `pp-default` with the restarts drawn from 500 value-probed
                    directions instead of at random — the budget-matched answer
                    to their 500 perturbations. Note the composition:
                    `optimize_dcor` keeps one random start and replaces the
                    remaining four with the best-scoring probes, so this arm has
                    the same number of Adam runs as `pp-default` and differs only
                    in where four of the five begin. That mirrors `dcsol.m`,
                    which also spends its 500 candidates on choosing a start
                    rather than on extra solves.
    adam-dcor2      joint Riemannian Adam on whitened Z, B'B = I_k
    sqp-dcor2       their SLSQP solve on whitened Z, V'V = I_d, same objective

Every arm is scored under *both* criteria (dCor^2_u summed over columns, and
their unsquared V-statistic dCov) plus the accuracy measures, so no arm is judged
only by what it maximises.

Data
----
  * `grid`: the single-index grid at p = 20, n = 200, k = 1 — five
    nonlinearities x three noise levels, which is where the published gap is
    measured (median 22.25 degrees against 17.51);
  * `sy_small`: the Sheng & Yin (2016) design at (n, p) = (100, 6), models A, B
    and C, predictor parts (1) standard normal and (3) discrete;
  * `sy_large`: the same design at (500, 20), part (1) only, where their solver
    costs 14.67 s per fit against 1.62 s.

The seed list and the replicate counts are literals below, and both are written
into the header of the CSV this produces.

Writes results_fair_comparison.csv.
"""

# Thesis:   Chapter 4, §4.5
# Writes:   results/results_fair_comparison.csv
# Original: PP_Dcor/fair_comparison.py on the thesis branch.
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
from scipy.optimize import minimize


from dpp.supervised.data_generator import data_generator                  # noqa: E402
from dpp.supervised.dcor_optimizer import (optimize_dcor, dcor_u,          # noqa: E402
                            _u_center, _make_B)
from dpp.supervised.evaluation import _subspace_angle_1d, principal_angles  # noqa: E402
from dpp.supervised.joint_optimization import joint_optimize              # noqa: E402
from dpp.supervised.pp_helpers import deflate_X, sigma_inv_root           # noqa: E402
from dpp.supervised.sheng_yin import (sheng_yin_sdr, dcov_v, _orth,        # noqa: E402
                       _orth_complement, _double_center, _dist)
from designs.sheng_yin_2016 import MODELS, make_data, delta_m  # noqa: E402
from dpp.supervised.sdr_baselines import _whiten                          # noqa: E402

import warnings
warnings.filterwarnings("ignore")

SEEDS = (42, 7, 123, 2024, 5)

# ── budgets ──────────────────────────────────────────────────────────────────
#: Their solver's own settings, unchanged from `sheng_yin.py` / `dcsol.m`.
N_PERTURB = 500
SQP_TOL = 1e-4
SQP_MAX_ITER = 1000

#: This project's own settings, unchanged from `step4_projection_pursuit.py`.
N_RESTARTS = 5
MAX_ITER = 150

#: Value-probe budget of the `pp-probe500` arm, matched to N_PERTURB.
N_PROBES = 500

# ── data ─────────────────────────────────────────────────────────────────────
GRID_NONLINEARITIES = ['square', 'sine', 'abs', 'cubic', 'tanh']
GRID_NOISE = [0.0, 0.5, 1.0]
GRID_N, GRID_P = 200, 20

SY_BASE_SEED = 20260813        # the base seed of sheng_yin_2016_study
SY_SMALL = (100, 6)
SY_LARGE = (500, 20)
REPS_DEFAULT = (20, 8)


def _reps() -> tuple[int, int]:
    """The replicate counts for the two design parts, as literals."""
    return REPS_DEFAULT


# ── this project's objective, in the form SLSQP needs ────────────────────────

def _dcor2_u_raw(z, B_y, s_yy):
    """``s_xy / sqrt(s_xx s_yy)`` for one projection, **without** the clamp.

    `dcor_u` returns 0 when s_xy <= 0. That is the right reporting convention —
    the U-statistic can go slightly negative under independence — but as an
    objective it is a plateau, and a derivative-free solver started inside it
    never leaves. This version is the same rational function with the clamp
    removed, so it is smooth everywhere the distances are distinct and agrees
    with `dcor_u` at every point where s_xy > 0.
    """
    n = len(z)
    A = _u_center(np.abs(z[:, None] - z[None, :]))
    denom = n * (n - 3)
    s_xx = np.einsum('ij,ij->', A, A) / denom
    s_xy = np.einsum('ij,ij->', A, B_y) / denom
    if s_xx <= 0 or s_yy <= 0:
        return 0.0
    return float(s_xy / np.sqrt(s_xx * s_yy))


def _marginal_sum(V, Z, B_y, s_yy):
    """``sum_j dCor^2_u(Z v_j, y)`` — the signal term of eq. (4.20)."""
    ZV = Z @ V
    return sum(_dcor2_u_raw(ZV[:, j], B_y, s_yy) for j in range(V.shape[1]))


def sqp_on_dcor2(X, y, d=1, n_starts=5, seed=0, tol=SQP_TOL,
                 max_iter=SQP_MAX_ITER):
    """Their optimiser on this project's objective.

    SLSQP with a finite-difference gradient, on the whitened predictors, under
    the orthonormality constraint ``V'V = I_d`` — the parameterisation, the
    constraint, the tolerance and the iteration budget of `dcsol.m`. The only
    change is which function is maximised.

    Returns the directions in the original predictor scale, unit-normed, plus an
    info dict. The columns satisfy ``beta_i' Sigma_hat beta_j = 0``, which is the
    Sigma-orthogonality of Section 4.6, not Euclidean orthogonality.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n, p = X.shape
    t0 = time.perf_counter()

    Z, inv_root, _ = _whiten(X)
    B_y, s_yy = _make_B(y)
    iu = np.triu_indices(d)
    rng = np.random.default_rng(seed)

    def neg_obj(v_flat):
        return -_marginal_sum(v_flat.reshape(p, d), Z, B_y, s_yy)

    def con(v_flat):
        V = v_flat.reshape(p, d)
        return (V.T @ V - np.eye(d))[iu]

    best, nfev, status = None, 0, -1
    for _ in range(n_starts):
        V0 = _orth(rng.standard_normal((p, d)))[:, :d]
        if V0.shape[1] < d:
            continue
        res = minimize(neg_obj, V0.ravel(), method='SLSQP',
                       constraints=[{'type': 'eq', 'fun': con}],
                       options={'ftol': tol, 'maxiter': max_iter})
        nfev += int(res.nfev)
        V = res.x.reshape(p, d)
        if np.linalg.matrix_rank(V) == d:
            V = _orth(V)[:, :d]
        else:
            V = V0
        val = -neg_obj(V.ravel())
        if best is None or val > best[0]:
            best, status = (val, V), int(res.status)

    val, V = best
    B = inv_root @ V
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    return B, {'objective': val, 'seconds': time.perf_counter() - t0,
               'nfev': nfev, 'status': status}


def adam_on_dcor2(X, y, d=1, n_restarts=N_RESTARTS, seed=0, max_iter=MAX_ITER):
    """This project's optimiser on the same objective and the same manifold.

    `joint_optimize` at ``lam = 0`` is Riemannian Adam on St(p, d) maximising the
    marginal sum; running it on the whitened predictors makes its constraint
    ``B'B = I_d`` the same Sigma-orthogonality that `sqp_on_dcor2` imposes, so the
    two arms differ only in how the manifold is searched.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    t0 = time.perf_counter()
    inv_root = sigma_inv_root(X)
    Z = X @ inv_root
    V, val, info = joint_optimize(Z, y, d, lam=0.0, n_restarts=n_restarts,
                                  max_iter=max_iter, seed=seed)
    B = inv_root @ V
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    return B, {'objective': float(val), 'seconds': time.perf_counter() - t0,
               'spread': info['spread'], 'nfev': -1}


def seq_pp_local(X, y, d, n_restarts=N_RESTARTS, max_iter=MAX_ITER, seed=0,
                 n_probes=0):
    """`pp_helpers.seq_pp` with ``deflation='X_deflation'``, plus probe support.

    `seq_pp` hard-codes ``init_method='random'`` and no probes, so the
    budget-matched arm needs its own loop. Everything else is identical.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    p = X.shape[1]
    t0 = time.perf_counter()
    W = np.zeros((p, d))
    X_work = X.copy()
    spreads = []
    for j in range(d):
        beta, _, info = optimize_dcor(X_work, y, init_method='random',
                                      optimizer='gradient_ascent',
                                      n_restarts=n_restarts, max_iter=max_iter,
                                      seed=seed + j, n_probes=n_probes)
        W[:, j] = beta
        spreads.append(info['val_spread'])
        if j < d - 1:
            X_work = deflate_X(X_work, W[:, :j + 1])
    return W, {'seconds': time.perf_counter() - t0,
               'spread': float(np.mean(spreads)), 'nfev': -1}


# ── the arms ─────────────────────────────────────────────────────────────────

def _random_v0(p, d, seed):
    """One random orthonormal start in whitened coordinates."""
    rng = np.random.default_rng([seed, 7717])
    for _ in range(20):
        V = _orth(rng.standard_normal((p, d)))
        if V.shape[1] >= d:
            return V[:, :d]
    raise RuntimeError("no full-rank random start")


def _sy_random_perturb(X, y, d, seed, n_perturb=N_PERTURB):
    """Their perturbation stage applied to a *random* seed instead of SIR/SAVE.

    Reproduces the loop of `sheng_yin_sdr` — 500 orthonormalised perturbations
    ``orth(v + 0.1 Vc G)`` scored by their own objective — then hands the winner
    to the same solve through ``v0``. Isolates the SIR seed with the
    perturbation budget held fixed.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n, p = X.shape
    t0 = time.perf_counter()
    Z, _, _ = _whiten(X)
    B_y = _double_center(_dist(y.reshape(n, 1)))

    def obj(V):                              # dcreg.m, sign flipped to maximise
        A = _double_center(_dist(Z @ V))
        return np.sqrt(max(np.einsum('ij,ij->', A, B_y) / A.size, 0.0))

    v_seed = _random_v0(p, d, seed)
    v_best, best_val = v_seed, obj(v_seed)
    if p > d:
        rng = np.random.default_rng([seed, 9931])
        Vc = _orth_complement(v_seed)
        for _ in range(n_perturb):
            cand = _orth(v_seed + 0.1 * (Vc @ rng.standard_normal((p - d, d))))
            if cand.shape[1] < d:
                continue
            val = obj(cand[:, :d])
            if val > best_val:
                best_val, v_best = val, cand[:, :d]
    B, info = sheng_yin_sdr(X, y, d=d, seed=seed, v0=v_best,
                            tol=SQP_TOL, max_iter=SQP_MAX_ITER)
    info['seconds'] += time.perf_counter() - t0 - info['seconds']
    return B, info


def _sy_rand_multi(X, y, d, seed, n_starts=N_RESTARTS):
    """Best of `n_starts` random starts, each solved and scored by their objective."""
    t0 = time.perf_counter()
    p = X.shape[1]
    best = None
    for r in range(n_starts):
        v0 = _random_v0(p, d, seed * 1000 + r)
        B, info = sheng_yin_sdr(X, y, d=d, seed=seed, v0=v0,
                                tol=SQP_TOL, max_iter=SQP_MAX_ITER)
        if best is None or info['objective'] > best[1]['objective']:
            best = (B, info)
    B, info = best
    info = dict(info, seconds=time.perf_counter() - t0)
    return B, info


def run_arm(arm, X, y, d, seed):
    """One arm on one data set. Returns (B in X-scale, seconds, nfev)."""
    if arm == 'sy-full':
        B, info = sheng_yin_sdr(X, y, d=d, n_perturb=N_PERTURB, seed=seed,
                                tol=SQP_TOL, max_iter=SQP_MAX_ITER)
    elif arm == 'sy-seed':
        B, info = sheng_yin_sdr(X, y, d=d, n_perturb=0, seed=seed,
                                tol=SQP_TOL, max_iter=SQP_MAX_ITER)
    elif arm == 'sy-rand1':
        B, info = sheng_yin_sdr(X, y, d=d, seed=seed,
                                v0=_random_v0(X.shape[1], d, seed),
                                tol=SQP_TOL, max_iter=SQP_MAX_ITER)
    elif arm == 'sy-rand5':
        B, info = _sy_rand_multi(X, y, d, seed)
    elif arm == 'sy-rand500':
        B, info = _sy_random_perturb(X, y, d, seed)
    elif arm == 'pp-default':
        B, info = seq_pp_local(X, y, d, seed=seed)
    elif arm == 'pp-rand1':
        B, info = seq_pp_local(X, y, d, n_restarts=1, seed=seed)
    elif arm == 'pp-probe500':
        B, info = seq_pp_local(X, y, d, seed=seed, n_probes=N_PROBES)
    elif arm == 'adam-dcor2':
        B, info = adam_on_dcor2(X, y, d=d, seed=seed)
    elif arm == 'sqp-dcor2':
        B, info = sqp_on_dcor2(X, y, d=d, n_starts=N_RESTARTS, seed=seed)
    else:
        raise ValueError(arm)
    B = np.asarray(B, float).reshape(X.shape[1], -1)
    return B, float(info['seconds']), int(info.get('nfev', -1))


ARMS = ['sy-full', 'sy-seed', 'sy-rand1', 'sy-rand5', 'sy-rand500',
        'pp-rand1', 'pp-default', 'pp-probe500', 'adam-dcor2', 'sqp-dcor2']


# ── scoring ──────────────────────────────────────────────────────────────────

def score(B, X, y, B_true, d):
    """Both criteria and both accuracy measures for one solution."""
    Bn = B / np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-300)
    Z = X @ Bn
    dcor2_sum = float(sum(dcor_u(Z[:, j], y) for j in range(d)))
    out = {
        'delta_m': round(delta_m(B, B_true), 6),
        'dcor2_u_sum': round(dcor2_sum, 6),
        'dcor2_u_mean': round(dcor2_sum / d, 6),
        'dcov_v': round(dcov_v(Z, y), 6),          # their criterion, multivariate
    }
    if d == 1:
        out['angle'] = round(float(_subspace_angle_1d(Bn[:, 0], B_true)), 4)
    else:
        out['angle'] = round(float(np.mean(principal_angles(Bn, B_true))), 4)
    return out


# ── the three data parts ─────────────────────────────────────────────────────

def run_grid():
    rows = []
    for seed in SEEDS:
        for nl in GRID_NONLINEARITIES:
            for noise in GRID_NOISE:
                X, Y, W_true = data_generator(GRID_N, GRID_P, 1, nl, noise, seed)
                for arm in ARMS:
                    B, secs, nfev = run_arm(arm, X, Y, 1, seed)
                    rows.append(dict(
                        part='grid', seed=seed, model='', nonlinearity=nl,
                        noise=noise, n=GRID_N, p=GRID_P, d=1, arm=arm,
                        seconds=round(secs, 3), nfev=nfev,
                        **score(B, X, Y, W_true, 1)))
        print(f"  [grid] seed {seed} done", flush=True)
    return rows


def run_sy(part_name, n, p, predictor_parts, n_rep):
    rows = []
    configs = [(m, pt) for m in MODELS for pt in predictor_parts]
    for cid, (model, pt) in enumerate(configs):
        d = MODELS[model]
        for rep in range(n_rep):
            rng = np.random.default_rng([SY_BASE_SEED, cid, rep])
            X, Y, B_true = make_data(model, pt, n, p, rng)
            for arm in ARMS:
                B, secs, nfev = run_arm(arm, X, Y, d, rep)
                rows.append(dict(
                    part=part_name, seed=rep, model=f'{model}({pt})',
                    nonlinearity='', noise=0.0, n=n, p=p, d=d, arm=arm,
                    seconds=round(secs, 3), nfev=nfev,
                    **score(B, X, Y, B_true, d)))
        print(f"  [{part_name}] model {model} part {pt} "
              f"({n_rep} reps) done", flush=True)
    return rows


# ── reporting ────────────────────────────────────────────────────────────────

def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    order = [a for a in ARMS if a in set(df.arm)]

    def pivot(frame, values, index):
        t = frame.pivot_table(index=index, columns='arm', values=values,
                              aggfunc='median')
        return t.reindex(columns=[c for c in order if c in t.columns]).round(4)

    grid = df[df.part == 'grid']
    if not grid.empty:
        print(f"\n=== single-index grid, p = {GRID_P}, median over "
              f"{grid.seed.nunique()} seeds ===")
        print("\n-- recovery angle (deg), by nonlinearity and noise --")
        print(pivot(grid, 'angle', ['nonlinearity', 'noise']).to_string())
        print("\n-- overall median --")
        print(pivot(grid, 'angle', 'p').to_string())
        print("\n-- dCor^2_u at the solution (this project's criterion) --")
        print(pivot(grid, 'dcor2_u_sum', 'p').to_string())
        print("\n-- dCov_v at the solution (their criterion) --")
        print(pivot(grid, 'dcov_v', 'p').to_string())

    for part in ('sy_small', 'sy_large'):
        sub = df[df.part == part]
        if sub.empty:
            continue
        n, p = sub.n.iloc[0], sub.p.iloc[0]
        print(f"\n=== Sheng & Yin design, (n, p) = ({n}, {p}), median over "
              f"{sub.seed.nunique()} replicates ===")
        print("\n-- Delta_m (smaller is better) --")
        print(pivot(sub, 'delta_m', 'model').to_string())
        print("\n-- dCor^2_u summed over columns --")
        print(pivot(sub, 'dcor2_u_sum', 'model').to_string())
        print("\n-- dCov_v (their criterion) --")
        print(pivot(sub, 'dcov_v', 'model').to_string())

    print("\n=== median seconds per fit, by arm and part ===")
    print(pivot(df, 'seconds', 'part').to_string())

    print("\n=== the two questions, in one line each ===")
    for part, label in (('grid', f'grid p={GRID_P}'),
                        ('sy_small', 'SY (100,6)'), ('sy_large', 'SY (500,20)')):
        sub = df[df.part == part]
        if sub.empty:
            continue
        med = sub.groupby('arm').delta_m.median()
        got = lambda a: f"{med[a]:.3f}" if a in med else "n/a"       # noqa: E731
        print(f"  {label:14s} Delta_m  seed-vs-random: "
              f"sy-full {got('sy-full')} | sy-seed {got('sy-seed')} | "
              f"sy-rand1 {got('sy-rand1')} | sy-rand5 {got('sy-rand5')} | "
              f"sy-rand500 {got('sy-rand500')}")
        print(f"  {' ':14s}          same-objective:  "
              f"adam-dcor2 {got('adam-dcor2')} | sqp-dcor2 {got('sqp-dcor2')}")
    return df


if __name__ == '__main__':
    reps_small, reps_large = _reps()
    t0 = time.time()
    rows = run_grid()
    rows += run_sy('sy_small', *SY_SMALL, (1, 3), reps_small)
    rows += run_sy('sy_large', *SY_LARGE, (1,), reps_large)
    print(f"\ntotal {time.time() - t0:.0f}s")
    df = summarise(rows)
    out = RESULTS / 'results_fair_comparison.csv'
    stamp = write_csv(
        out, df, seeds=SEEDS, script='ch04_fair_comparison.py',
        arms=", ".join(ARMS),
        replicates=f"{reps_small} at {SY_SMALL}, {reps_large} at {SY_LARGE}",
        n_perturb=N_PERTURB, n_probes=N_PROBES, n_restarts=N_RESTARTS,
        max_iter=MAX_ITER, sqp_max_iter=SQP_MAX_ITER, sqp_tol=SQP_TOL)
    print(f"\nWrote {out.name} ({len(df)} rows")
