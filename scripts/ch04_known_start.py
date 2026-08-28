"""
known_start_check.py

Is a failed recovery the optimiser's fault or the criterion's?

The supervisor's mark on the `sine` case reads: "Does it help if you re-line at
known maximum? Try it. Then you can justify this claim saying that this is
indeed an optimisation problem." The check is the obvious one and it settles
more than the `sine` case.

For each single-index configuration this script records three things:

  * ``dcor2_true``   the objective at the *true* direction e_1;
  * ``dcor2_found``  the objective at the direction a random-start search finds;
  * ``dcor2_from_truth`` / ``angle_from_truth``  what happens when the search is
    started at e_1 — does it stay, and at what value.

They separate two cases that look identical from the outside:

  * ``dcor2_found < dcor2_true``: the maximiser was missed. The landscape beat
    the search, and a better optimiser or more restarts would help. This is an
    optimisation problem.
  * ``dcor2_found > dcor2_true``: the search did its job and the sample
    criterion simply does not have its maximum at the truth. No optimiser can
    fix that; it is the criterion, or the sample size, that is at fault.

The second case is what the comparison against Sheng & Yin's solver at p = 20
points to: our solutions attain the higher dCor^2 in 87% of those
configurations while recovering the subspace less accurately.

Writes results_known_start.csv.
"""

# Thesis:   Chapter 4, §4.4
# Writes:   results/results_known_start.csv
# Original: PP_Dcor/known_start_check.py on the thesis branch.
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
from dpp.supervised.dcor_optimizer import optimize_dcor, dcor_u     # noqa: E402
from dpp.supervised.evaluation import _subspace_angle_1d            # noqa: E402

import warnings
warnings.filterwarnings("ignore")

SEEDS = (42, 7, 123, 2024, 5)

NONLINEARITIES = ['square', 'sine', 'abs', 'cubic', 'tanh']
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0]
DIMENSIONS = [5, 20]
N = 200
N_RESTARTS = 3
MAX_ITER = 100


def run():
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        for nl in NONLINEARITIES:
            for noise in NOISE_LEVELS:
                for p in DIMENSIONS:
                    X, Y, W_true = data_generator(N, p, 1, nl, noise, seed)
                    beta_true = W_true[:, 0]
                    v_true = dcor_u(X @ beta_true, Y)

                    beta, v_found, _ = optimize_dcor(
                        X, Y, init_method='random', optimizer='gradient_ascent',
                        n_restarts=N_RESTARTS, max_iter=MAX_ITER, seed=seed)

                    # The same optimiser started exactly at the truth. `ols` is
                    # replaced by an explicit start vector: `informed_inits` and
                    # `init_method` only name canned initialisers, so the start
                    # is injected by running a single restart from beta_true.
                    beta_ks, v_ks = _from_start(X, Y, beta_true)

                    rows.append(dict(
                        seed=seed, nonlinearity=nl, noise=noise, n=N, p=p,
                        dcor2_true=round(float(v_true), 6),
                        dcor2_found=round(float(v_found), 6),
                        dcor2_from_truth=round(float(v_ks), 6),
                        angle_found=round(_subspace_angle_1d(beta, W_true), 4),
                        angle_from_truth=round(
                            _subspace_angle_1d(beta_ks, W_true), 4),
                        found_beats_truth=bool(v_found > v_true + 1e-9),
                    ))
        print(f"  seed {seed} done [{time.time() - t0:.0f}s]", flush=True)
    return rows


def _from_start(X, Y, beta0):
    """Run the gradient search from one given starting direction."""
    from dpp.supervised.dcor_optimizer import _make_B, _riemannian_adam
    B, s_yy = _make_B(Y)
    beta, val = _riemannian_adam(beta0 / np.linalg.norm(beta0), X, B, s_yy,
                                 max_iter=MAX_ITER)
    return beta, val


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== does the search beat the truth on its own criterion? ===")
    print(df.pivot_table(index=['p'], columns='nonlinearity',
                         values='found_beats_truth', aggfunc='mean')
          .round(2).to_string())
    print("\n=== median dCor^2: at the truth / found / started at the truth ===")
    print(df.pivot_table(index=['p', 'nonlinearity'],
                         values=['dcor2_true', 'dcor2_found',
                                 'dcor2_from_truth'], aggfunc='median')
          .round(4).to_string())
    print("\n=== median angle: random start against a start at the truth ===")
    print(df.pivot_table(index=['p', 'nonlinearity'], columns='noise',
                         values=['angle_found', 'angle_from_truth'],
                         aggfunc='median').round(2).to_string())
    return df


if __name__ == '__main__':
    rows = run()
    df = summarise(rows)
    out = RESULTS / 'results_known_start.csv'
    stamp = write_csv(out, df, seeds=SEEDS,
                                 script='ch04_known_start.py',
                                 n_restarts=N_RESTARTS, max_iter=MAX_ITER)
    print(f"\nWrote {out.name} ({len(df)} rows")
