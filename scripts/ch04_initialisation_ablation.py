"""
init_ablation.py

Where does the accuracy gap against Sheng & Yin's solver come from?

`results_sheng_yin_ch4.csv` puts the two solvers level at p = 2 and p = 5 and
apart at p = 20 (median recovery angle 22.25 degrees against 17.51), with this
project's optimiser six times faster. The two searches differ in three ways: the
objective (U-statistic dCor^2 against the V-statistic dCov), the search geometry
(the sphere in the original coordinates against the sphere in *whitened*
coordinates) and the starting point (random against the better of SIR and SAVE,
then 500 perturbations of it).

This script holds the objective and the optimiser fixed and varies the other
two, so that the gap can be attributed rather than guessed at:

    random           the current default: random starts, original coordinates
    sir              seeded from SIR/SAVE, original coordinates
    whitened         random starts, search on Z = X Sigma^{-1/2}
    whitened+sir     both

`sir_initialisation_study.py` already asked half of this question and found that
SIR seeding alone does not help on the isotropic grid. What is new here is the
whitening, and that the arms are run at the dimension where the gap is measured.

Two data sets, both at p = 20:

  * the single-index grid of `step3_experiments.py` at p = 20, five
    nonlinearities x four noise levels, n = 200, k = 1;
  * the three models of the Sheng & Yin (2016) design at (n, p) = (500, 20),
    standard-normal predictors, reusing `sheng_yin_2016_study.make_data`.

Writes results_init_ablation.csv.
"""

# Thesis:   Chapter 4, §4.7
# Writes:   results/results_init_ablation.csv
# Original: PP_Dcor/init_ablation.py on the thesis branch.
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
from pathlib import Path

import numpy as np


from dpp.supervised.data_generator import data_generator            # noqa: E402
from dpp.supervised.dcor_optimizer import optimize_dcor             # noqa: E402
from dpp.supervised.evaluation import _subspace_angle_1d, principal_angles   # noqa: E402
from dpp.supervised.pp_helpers import seq_pp, sigma_inv_root        # noqa: E402
from designs.sheng_yin_2016 import MODELS, make_data, delta_m   # noqa: E402

import warnings
warnings.filterwarnings("ignore")

SEEDS = (42, 7, 123, 2024, 5)

GRID_NONLINEARITIES = ['square', 'sine', 'abs', 'cubic', 'tanh']
GRID_NOISE = [0.0, 0.5, 1.0, 2.0]
GRID_N, GRID_P = 200, 20

SY_N, SY_P = 500, 20
SY_BASE_SEED = 20260813        # the base seed of sheng_yin_2016_study

N_RESTARTS = 5
MAX_ITER = 150

#: Restart budget of the "more starts" arm. `dcsol.m` scores 500 perturbations
#: of its seed before the solve; five restarts against that is not a like-for-
#: like search, so one arm buys the budget back.
N_RESTARTS_BIG = 25

#: (arm, whiten, init_method, informed_inits, n_restarts)
ARMS = [
    ('random',        False, 'random', (),         N_RESTARTS),
    ('sir',           False, 'sir',    ('save',),  N_RESTARTS),
    ('whitened',      True,  'random', (),         N_RESTARTS),
    ('more starts',   False, 'random', (),         N_RESTARTS_BIG),
    ('whitened+sir+starts', True, 'sir', ('save',), N_RESTARTS_BIG),
]

#: Predictor covariance for the grid part. Whitening cannot matter under
#: rho = 0, so an ablation run only there would answer nothing.
GRID_RHOS = [0.0, 0.5]


def _fit_1d(X, Y, whiten, init_method, informed, seed, n_restarts=N_RESTARTS):
    """One single-index fit under one arm; returns the direction in X-scale."""
    inv_root = sigma_inv_root(X) if whiten else None
    X_work = X @ inv_root if whiten else X
    beta, val, info = optimize_dcor(X_work, Y, init_method=init_method,
                                    optimizer='gradient_ascent',
                                    n_restarts=n_restarts, max_iter=MAX_ITER,
                                    seed=seed, informed_inits=informed)
    if whiten:
        beta = inv_root @ beta
        beta = beta / np.linalg.norm(beta)
    return beta, val, info['val_spread']


def run_grid():
    rows = []
    for seed in SEEDS:
        for nl in GRID_NONLINEARITIES:
          for rho in GRID_RHOS:
            for noise in GRID_NOISE:
                X, Y, W_true = data_generator(GRID_N, GRID_P, 1, nl, noise, seed,
                                              rho=rho)
                for arm, whiten, init_method, informed, restarts in ARMS:
                    t = time.time()
                    beta, val, spread = _fit_1d(X, Y, whiten, init_method,
                                                informed, seed, restarts)
                    rows.append(dict(
                        part='grid', seed=seed, config=f'{nl}_s{noise}_r{rho}',
                        model='', nonlinearity=nl, noise=noise, rho=rho,
                        n=GRID_N, p=GRID_P, d=1, arm=arm,
                        angle=round(_subspace_angle_1d(beta, W_true), 4),
                        delta_m=round(delta_m(beta.reshape(-1, 1), W_true), 6),
                        dcor2_u=round(float(val), 6),
                        restart_spread=round(float(spread), 6),
                        seconds=round(time.time() - t, 3)))
        print(f"  [grid] seed {seed} done", flush=True)
    return rows


def run_sheng_yin_models():
    """The same arms on the replication design, where the gap was measured."""
    rows = []
    for seed_idx, seed in enumerate(SEEDS):
        for model, d in MODELS.items():
            rng = np.random.default_rng([SY_BASE_SEED, 900 + seed_idx, 0])
            X, Y, B_true = make_data(model, 1, SY_N, SY_P, rng)
            for arm, whiten, init_method, informed, restarts in ARMS:
                t = time.time()
                if d == 1:
                    beta, val, spread = _fit_1d(X, Y, whiten, init_method,
                                                informed, seed, restarts)
                    W = beta.reshape(-1, 1)
                    ang = float(_subspace_angle_1d(beta, B_true))
                else:
                    # k > 1: the whitened arm is exactly Sigma-orthogonal
                    # deflation, which is what the whitening buys here. The
                    # seeded arms recompute the inverse-regression start on the
                    # deflated predictors at every direction.
                    W, info = seq_pp(X, Y, d,
                                     deflation='X_sigma' if whiten else 'X_deflation',
                                     n_restarts=restarts, max_iter=MAX_ITER,
                                     seed=seed, init_method=init_method,
                                     informed_inits=informed)
                    val = float(np.mean([i['dcor_vs_Y_orig'] for i in info]))
                    spread = float(np.mean([i['spread'] for i in info]))
                    ang = float(np.mean(principal_angles(W, B_true)))
                rows.append(dict(
                    part='sheng_yin_model', seed=seed, config=f'model_{model}',
                    model=model, nonlinearity='', noise=0.0,
                    n=SY_N, p=SY_P, d=d, arm=arm,
                    angle=round(ang, 4) if ang == ang else '',
                    delta_m=round(delta_m(W, B_true), 6),
                    dcor2_u=round(float(val), 6),
                    restart_spread=round(float(spread), 6),
                    seconds=round(time.time() - t, 3)))
        print(f"  [SY models] seed {seed} done", flush=True)
    return rows


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    grid = df[df.part == 'grid']
    if not grid.empty:
        print(f"\n=== single-index grid, p = {GRID_P}: recovery angle, "
              f"median over {len(SEEDS)} seeds ===")
        print(grid.pivot_table(index=['rho', 'nonlinearity', 'noise'],
                               columns='arm', values='angle', aggfunc='median')
              .round(2).to_string())
        print("\n  overall median angle by arm and predictor covariance")
        print(grid.pivot_table(index='rho', columns='arm', values='angle',
                               aggfunc='median').round(3).to_string())
        print("\n  overall median dCor^2 at the solution by arm")
        print(grid.groupby('arm').dcor2_u.median().round(4).to_string())
    sy = df[df.part == 'sheng_yin_model']
    if not sy.empty:
        print(f"\n=== Sheng & Yin models at (n, p) = ({SY_N}, {SY_P}): "
              f"Delta_m, median over {len(SEEDS)} seeds ===")
        print(sy.pivot_table(index='model', columns='arm', values='delta_m',
                             aggfunc='median').round(3).to_string())
        multi = sy[sy.d > 1]
        if not multi.empty:
            print(f"\n=== the same models at d > 1: mean principal angle "
                  f"(deg), median over {len(SEEDS)} seeds ===")
            print(multi.pivot_table(index='model', columns='arm',
                                    values='angle', aggfunc='median')
                  .round(2).to_string())
            print("\n  spread across seeds, [min, max] by model and arm")
            for (model, arm), g in multi.groupby(['model', 'arm']):
                a = g.angle.astype(float)
                print(f"    {model:>3s} {arm:>20s}  median {a.median():6.2f}  "
                      f"[{a.min():.2f}, {a.max():.2f}]")
    print("\n=== median seconds per fit by arm ===")
    print(df.groupby('arm').seconds.median().round(3).to_string())
    return df


if __name__ == '__main__':
    rows = run_grid() + run_sheng_yin_models()
    df = summarise(rows)
    out = RESULTS / 'results_init_ablation.csv'
    stamp = write_csv(out, df, seeds=SEEDS,
                                 script='ch04_initialisation_ablation.py',
                                 arms=", ".join(a[0] for a in ARMS),
                                 n_restarts=N_RESTARTS, max_iter=MAX_ITER)
    print(f"\nWrote {out.name} ({len(df)} rows")
