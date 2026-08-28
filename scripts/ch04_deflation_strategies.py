"""
Step 4 — Multi-direction projection pursuit, k > 1.

Three deflation strategies investigated:
  (a) X-deflation: project X onto the orthogonal complement of the directions
      found so far. Enforces β_{i+1} ⊥ β_i in weight space (R^p).
  (b) Σ-orthogonal deflation: the same, in the Σ inner product, so that
      β_i' Σ β_j = 0 and the *projections* are uncorrelated. Implemented by
      whitening. Identical to (a) when the predictors are isotropic, which is
      why the configurations below are also run with correlated predictors.
  (c) Y-residual deflation: remove the projection of Y onto z_j = X β_j from Y.
      Enforces diversity in data space (R^n). From Friedman & Stuetzle (1981).

Key tension: nonlinear signal cannot be fully removed by linear deflation.
After removing β₁ in weight space, X β₁ still exists in the deflated X's column
span and can still explain Y. Y-residual deflation sidesteps this by working
in R^n directly — but it only removes the *linear* projection of Y onto z_j,
not the nonlinear residual.

All three are expected to have problems; the goal is to understand what those
problems are and document them honestly.
"""

# Thesis:   Chapter 4, tab:p1-deflation
# Writes:   results/results_deflation.csv
# Original: PP_Dcor/step4_projection_pursuit.py on the thesis branch.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import sys, time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dpp.supervised.data_generator import data_generator
from dpp.supervised.dcor_optimizer import optimize_dcor, dcor_u
from dpp.supervised.evaluation import evaluate, principal_angles
from dpp.supervised.pp_helpers import deflate_X, deflate_Y, seq_pp, sigma_inv_root

import sys
from pathlib import Path



# ── Deflation strategies ──────────────────────────────────────────────────────
# The three strategies and the sequential loop live in `pp_helpers.py`, so that
# this script, `sheng_yin_comparison.py` and any later caller run the identical
# code. `X_deflation` orthogonalises the weights, `X_sigma` orthogonalises them
# in the Sigma inner product so that the projections are uncorrelated, and
# `Y_residual` residualises the response linearly.


# ── Evaluation: subspace recovery ────────────────────────────────────────────

def eval_subspace(W_hat, W_true, X, Y, label='', verbose=False):
    """Principal angles between the found and the true subspace.

    ``verbose`` defaults to False since the seed loop calls this 70 times; the run
    summary is printed once per seed and configuration instead.
    """
    angles = principal_angles(W_hat, W_true)
    k = W_true.shape[1]

    # Inter-direction dCor^2 (redundancy)
    inter = []
    for i in range(k):
        for j_idx in range(i+1, k):
            z_i = X @ W_hat[:, i]
            z_j = X @ W_hat[:, j_idx]
            inter.append(dcor_u(z_i, z_j))

    # Per-direction dCor^2 with Y
    per_dir = [dcor_u(X @ W_hat[:, j], Y) for j in range(k)]

    # Orthonormality
    gram = W_hat.T @ W_hat
    orth_err = np.linalg.norm(gram - np.eye(k), 'fro')

    if verbose:
        print(f"  [{label}]")
        print(f"    Principal angles (deg): {np.round(angles, 2)}")
        print(f"    Mean principal angle:   {angles.mean():.2f}°")
        print(f"    Per-dir dCor^2(z_j, Y):  {np.round(per_dir, 4)}")
        if inter:
            print(f"    Redundancy (inter-dCor^2): {np.round(inter, 4)}  "
                  f"(0=ideal, >0.3=redundant)")
        print(f"    ||W^T W - I||_F:       {orth_err:.4f}", flush=True)

    return {'angles': angles, 'per_dir': per_dir, 'inter': inter,
            'orth_err': orth_err, 'mean_angle': angles.mean()}


# ── Experiments ───────────────────────────────────────────────────────────────

experiments = [
    # (k, p, nl, noise, n, label)
    (2, 5,  'product',      0.0, 400, 'k2_p5_product_noiseless'),
    (2, 5,  'product',      0.5, 400, 'k2_p5_product_noisy'),
    (2, 10, 'sum_squares',  0.0, 400, 'k2_p10_sumSq_noiseless'),
    (2, 10, 'sum_squares',  0.5, 400, 'k2_p10_sumSq_noisy'),
    (3, 10, 'sum_squares',  0.0, 500, 'k3_p10_sumSq_noiseless'),
    (2, 5,  'sine_product', 0.0, 400, 'k2_p5_sineProd_noiseless'),
    (2, 20, 'product',      0.0, 500, 'k2_p20_product_noiseless'),
]

#: Five data seeds; this script ran the single seed 42 until 2026-08-11. See
#: SEEDS in step3_experiments.py for why.
SEEDS = (42, 7, 123, 2024, 5)

#: The three strategies, in the order the thesis presents them.
DEFLATIONS = ['X_deflation', 'X_sigma', 'Y_residual']
_SHORT = {'X_deflation': 'X-defl', 'X_sigma': 'Sigma-defl', 'Y_residual': 'Y-resid'}

#: Predictor covariance. 0.0 is the isotropic X ~ N(0, I) of every earlier run;
#: 0.5 is the AR(1) Sigma_ij = 0.5^|i-j| of Sheng & Yin (2016). Weight
#: orthogonality and uncorrelated projections coincide under the first and part
#: company under the second, which is the whole point of running both.
RHOS = [0.0, 0.5]

per_seed = {}          # (label, rho, deflation) -> {seed: metrics}

print("=" * 65)
print("Step 4 — Multi-direction Projection Pursuit")
print(f"{len(experiments)} configurations x {len(RHOS)} predictor covariances x "
      f"{len(DEFLATIONS)} deflation schemes x {len(SEEDS)} seeds")
print("=" * 65, flush=True)

t_start = time.time()
for seed in SEEDS:
    for (k, p, nl, noise, n, label) in experiments:
        for rho in RHOS:
            X, Y, W_true = data_generator(n, p, k, nl, noise, seed=seed, rho=rho)
            for deflation in DEFLATIONS:
                t0 = time.time()
                W_hat, info = seq_pp(X, Y, k, deflation=deflation,
                                     n_restarts=5, max_iter=150, seed=0)
                elapsed = time.time() - t0
                metrics = eval_subspace(W_hat, W_true, X, Y, label=deflation)
                metrics['per_dir_info'] = info
                metrics['elapsed'] = elapsed
                per_seed.setdefault((label, rho, deflation), {})[seed] = metrics
            got = "  ".join(
                f"{_SHORT[d]} {per_seed[(label, rho, d)][seed]['mean_angle']:6.2f}"
                for d in DEFLATIONS)
            print(f"  seed {seed:5d}  rho={rho:.1f}  {label:<28} {got}  "
                  f"[{time.time()-t_start:.0f}s]", flush=True)


def _agg(label, rho, deflation, getter):
    """Median and range over seeds of one scalar."""
    vals = [getter(per_seed[(label, rho, deflation)][s]) for s in SEEDS]
    a = np.asarray(vals, dtype=float)
    return float(np.median(a)), float(a.min()), float(a.max())


# `all_results` keeps the shape the plotting code below expects, holding medians.
all_results = {}
for key, by_seed in per_seed.items():
    med = dict(by_seed[SEEDS[0]])          # structure, including per-direction info
    med['mean_angle'] = float(np.median([by_seed[s]['mean_angle'] for s in SEEDS]))
    med['orth_err'] = float(np.median([by_seed[s]['orth_err'] for s in SEEDS]))
    med['elapsed'] = float(np.median([by_seed[s]['elapsed'] for s in SEEDS]))
    med['inter'] = [float(np.median([np.mean(by_seed[s]['inter'] or [0])
                                     for s in SEEDS]))]
    all_results[key] = med

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 96)
print(f"Mean principal angle in degrees — median [min, max] over {len(SEEDS)} seeds")
print("=" * 96)
_head = f"{'experiment':<30}" + "".join(f"{_SHORT[d]:>22}" for d in DEFLATIONS)
rows = []
for rho in RHOS:
    print("\npredictor covariance: " +
          ("isotropic (rho = 0)" if rho == 0 else f"AR(1), rho = {rho}"))
    print(_head)
    print("-" * 96)
    for (k, p, nl, noise, n, label) in experiments:
        line = f"{label:<30}"
        for deflation in DEFLATIONS:
            med, lo, hi = _agg(label, rho, deflation, lambda m: m['mean_angle'])
            line += f"{med:8.2f} [{lo:5.2f},{hi:6.2f}]".rjust(22)
        print(line)
        for deflation in DEFLATIONS:
            for seed in SEEDS:
                m = per_seed[(label, rho, deflation)][seed]
                rows.append({
                    'seed': seed, 'config': label, 'k': k, 'p': p, 'n': n,
                    'nonlinearity': nl, 'noise': noise, 'rho': rho,
                    'deflation': deflation,
                    'mean_angle': round(float(m['mean_angle']), 4),
                    'inter_dcor': round(float(np.mean(m['inter'] or [0])), 6),
                    'orth_err': round(float(m['orth_err']), 8),
                    'elapsed': round(float(m['elapsed']), 3),
                })

_stamp = write_csv(RESULTS / 'results_deflation.csv', rows, seeds=SEEDS,
                              script='ch04_deflation_strategies.py',
                              rhos=", ".join(str(r) for r in RHOS),
                              deflations=", ".join(DEFLATIONS))
print("")
print(f"Wrote: results_deflation.csv ({len(rows)} rows)", flush=True)

# ── Honest documentation of failures ─────────────────────────────────────────
print("\n" + "=" * 65)
print("HONEST DIAGNOSIS")
print("=" * 65)
print("""
Issue 1 — X-deflation redundancy:
  After deflating X, weight vectors are orthogonal (||W^T W - I||_F ≈ 0)
  but the DATA-SPACE projections may be nearly identical. The optimizer
  finds directions that are orthogonal in R^p but whose projections Xβ_i
  are highly correlated in R^n. dCor^2(Xβ_1, Xβ_2) can be > 0.5 even when
  β_1 ⊥ β_2. This is the fundamental failure of X-deflation.

Issue 2 — Y-residual deflation limits:
  Y-deflation removes the linear projection of Y onto z_j = Xβ_j.
  But if Y = f(X^T β) with f nonlinear, the residual Y - (Y^T z)z/||z||^2
  still contains nonlinear information from β_j. The next direction can
  again be attracted to β_j (or its neighborhood). Observed as: β_2 ≈ β_1.

Issue 3 — Flat landscape in high dimensions:
  For p=20, k=2, the dCor^2 surface is nearly flat in many directions.
  Any 2D subspace containing the true β_1 is a near-optimum.
  The optimizer cannot distinguish which 2D subspace to commit to.
  Recovery is partly random — principal angles spread widely across restarts.
""", flush=True)
