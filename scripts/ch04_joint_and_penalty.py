"""
joint_optimization.py

Joint optimization over Stiefel manifold St(p,k).

Objective (all values are the U-statistic, bias-corrected dCor^2 returned by
`dcor_optimizer.dcor_u` — one scale everywhere, as in the thesis):
    max_{B in St(p,k)}  sum_j dCor^2(Y, X β_j)  -  λ · sum_{i<j} dCor^2(X β_i, X β_j)

  - Signal term: each column's projection maximizes dependence with Y.
  - Penalty: penalize inter-direction dCor^2 to discourage redundancy.
  - Optimizer: Riemannian Adam with QR retraction.
"""

# Thesis:   Chapter 4, tab:p1-joint
# Writes:   results/results_joint.csv
# Original: PP_Dcor/joint_optimization.py on the thesis branch.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import sys, time
import numpy as np

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import dcor as _dcor
from dpp.supervised.data_generator import data_generator
from dpp.supervised.dcor_optimizer import _make_B, _dcor_and_grad
from dpp.supervised.pp_helpers import seq_pp, eval_subspace


# ── Penalty gradient ──────────────────────────────────────────────────────────

def _penalty_grad_col(beta_i, beta_j, X):
    """Gradient of dCor^2(Xβ_i, Xβ_j) w.r.t. β_i using analytical formula."""
    z_j = X @ beta_j
    B_j, s_jj = _make_B(z_j)
    dc, g = _dcor_and_grad(beta_i, X, B_j, s_jj)
    return dc, g


# ── Joint optimizer ───────────────────────────────────────────────────────────

def joint_optimize(X, Y, k, lam=0.0, n_restarts=5, max_iter=150,
                   lr=0.05, lr_min=0.005, seed=0, verbose=False):
    """
    Riemannian Adam on St(p,k) = {B in R^{p x k} : B^T B = I_k}.
    Retraction: QR decomposition of (B + α * tangent_gradient).
    """
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    B_Y, s_yy = _make_B(Y)   # cached

    def _obj_grad(B):
        val = 0.0
        grad = np.zeros_like(B)
        for j in range(k):
            dc_j, g_j = _dcor_and_grad(B[:, j], X, B_Y, s_yy)
            val += dc_j
            grad[:, j] += g_j
        if lam > 0:
            for i in range(k):
                for jj in range(i+1, k):
                    dc_ij, g_i = _penalty_grad_col(B[:, i], B[:, jj], X)
                    _,     g_j = _penalty_grad_col(B[:, jj], B[:, i], X)
                    val -= lam * dc_ij
                    grad[:, i] -= lam * g_i
                    grad[:, jj] -= lam * g_j
        return val, grad

    def _run(B0):
        B = B0.copy()
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        M = np.zeros_like(B); V = np.zeros_like(B)
        prev = -np.inf
        for t in range(1, max_iter+1):
            lr_t = lr_min + 0.5*(lr-lr_min)*(1 + np.cos(np.pi*(t-1)/max_iter))
            val, g = _obj_grad(B)
            g_tan = g - B @ (B.T @ g)          # project to tangent space
            M = b1*M + (1-b1)*g_tan
            V = b2*V + (1-b2)*g_tan**2
            Mh = M/(1-b1**t); Vh = V/(1-b2**t)
            step = Mh/(np.sqrt(Vh)+eps_a)
            sn = np.linalg.norm(step)
            if sn > 1: step = step/sn
            B_new = B + lr_t*step
            B, _ = np.linalg.qr(B_new)         # retraction
            if abs(val-prev) < 1e-7 and t > 20: break
            prev = val
        fv, _ = _obj_grad(B)
        return B, fv

    best_B, best_val = None, -np.inf
    restart_vals = []
    for r in range(n_restarts):
        B0, _ = np.linalg.qr(rng.standard_normal((p, k)))
        Br, vr = _run(B0)
        restart_vals.append(vr)
        if vr > best_val:
            best_val, best_B = vr, Br.copy()
        if verbose:
            print(f"    restart {r+1}: val={vr:.4f}", flush=True)

    return best_B, best_val, {
        'restart_vals': restart_vals,
        'spread': max(restart_vals)-min(restart_vals)
    }


# ── Experiment configs ────────────────────────────────────────────────────────

LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0]
CONFIGS = [
    (2, 5,  'product',     0.0, 200, 'k2_p5_product'),
    (2, 5,  'sum_squares', 0.0, 200, 'k2_p5_sumSq'),
    (2, 10, 'sum_squares', 0.0, 300, 'k2_p10_sumSq'),
    (2, 5,  'product',     0.5, 200, 'k2_p5_product_noisy'),
]
N_RESTARTS = 5
#: The joint search gets the larger budget, for the reason set out in
#: unified_grid.py: St(p,k) has more dimensions than the sphere each sequential
#: step searches, and sequential extraction already spends N_RESTARTS ascents per
#: direction where the joint search spends its budget once.  Raised from 4 to 20 on
#: 2026-08-17; the two budgets now match the master grid.
JOINT_RESTARTS = 20
MAX_ITER   = 150

#: Five data seeds instead of the single seed 42 this script used until 2026-08-11.
#: This is the site of findings F1 and F2 of thesis/REVISION_PLAN_2026-08-11.md: the
#: reported joint-versus-sequential reversal was an artefact of an older
#: degenerate-point clamp, and it was also a one-seed event — at seed 42 sequential
#: gave 36.81° on sum_squares, at the four other seeds 4.92°–8.76°.
SEEDS = (42, 7, 123, 2024, 5)

CSV = RESULTS / "results_joint.csv"

COLUMNS = ['seed', 'config', 'n', 'p', 'k', 'nonlinearity', 'noise',
           'strategy', 'lam', 'mean_angle', 'inter_dcor', 'obj',
           'restart_spread', 'time']


def _median_range(values):
    """Median and the min/max over seeds, the presentation the rerun standardises on."""
    a = np.asarray(values, dtype=float)
    return float(np.median(a)), float(a.min()), float(a.max())


def run_config(seed, k, p, nl, noise, n, label):
    """One configuration at one seed: sequential, then joint at each λ."""
    X, Y, W_true = data_generator(n, p, k, nl, noise, seed=seed)
    base = dict(seed=seed, config=label, n=n, p=p, k=k,
                nonlinearity=nl, noise=noise)
    rows = []

    t0 = time.time()
    W_seq, _ = seq_pp(X, Y, k, deflation='X_deflation',
                      n_restarts=N_RESTARTS, max_iter=MAX_ITER, seed=0)
    m_seq = eval_subspace(W_seq, W_true, X, Y, label='sequential')
    rows.append({**base, 'strategy': 'sequential', 'lam': 0.0,
                 'mean_angle': m_seq['mean_angle'],
                 'inter_dcor': float(np.mean(m_seq['inter'] or [0])),
                 'obj': float('nan'), 'restart_spread': float('nan'),
                 'time': time.time() - t0})

    for lam in LAMBDAS:
        t0 = time.time()
        B_hat, obj_val, info = joint_optimize(
            X, Y, k, lam=lam, n_restarts=JOINT_RESTARTS,
            max_iter=MAX_ITER, seed=0, verbose=False)
        m_jt = eval_subspace(B_hat, W_true, X, Y, label=f'joint λ={lam}')
        rows.append({**base, 'strategy': 'joint', 'lam': lam,
                     'mean_angle': m_jt['mean_angle'],
                     'inter_dcor': float(np.mean(m_jt['inter'] or [0])),
                     'obj': float(obj_val),
                     'restart_spread': float(info['spread']),
                     'time': time.time() - t0})
    return rows


def _select(rows, label, strategy, lam=None):
    return [r for r in rows if r['config'] == label
            and r['strategy'] == strategy
            and (lam is None or r['lam'] == lam)]


def main():
    print("=" * 62)
    print("Joint Stiefel optimisation against sequential PP")
    print(f"{len(CONFIGS)} configurations x {len(LAMBDAS)} lambda values x "
          f"{len(SEEDS)} seeds")
    print("=" * 62, flush=True)

    rows = []
    t_start = time.time()
    for seed in SEEDS:
        for cfg in CONFIGS:
            k, p, nl, noise, n, label = cfg
            print(f"\n  seed {seed}  {label}  "
                  f"(n={n}, p={p}, k={k}, nl={nl}, noise={noise})", flush=True)
            new = run_config(seed, *cfg)
            rows.extend(new)
            seq = new[0]['mean_angle']
            jt0 = next(r['mean_angle'] for r in new
                       if r['strategy'] == 'joint' and r['lam'] == 0.0)
            print(f"    sequential {seq:6.2f}°   joint(λ=0) {jt0:6.2f}°   "
                  f"[{time.time() - t_start:.0f}s]", flush=True)

    record = write_csv(CSV, rows, seeds=SEEDS, fieldnames=COLUMNS,
                                  script='ch04_joint_and_penalty.py',
                                  n_restarts=N_RESTARTS,
                                  joint_restarts=JOINT_RESTARTS,
                                  max_iter=MAX_ITER, lambdas=LAMBDAS)
    print(f"\nWrote {CSV.name}: {len(rows)} rows", flush=True)

    # ── Summary: median over seeds, with the seed range beside it ────────────────
    # No argmin-over-λ column. Picking the best λ per configuration on a single seed
    # is what produced the withdrawn "Best λ" column of tab:p1-joint; with five seeds
    # the honest object is the median at each fixed λ.
    print("\n" + "=" * 78)
    print(f"Mean principal angle in degrees — median [min, max] over "
          f"{len(SEEDS)} seeds")
    print("=" * 78)
    header = ['sequential'] + [f'joint λ={l}' for l in LAMBDAS]
    print(f"{'config':<24}" + "".join(f"{h:>21}" for h in header))
    for (k, p, nl, noise, n, label) in CONFIGS:
        cells = []
        med, lo, hi = _median_range([r['mean_angle']
                                     for r in _select(rows, label, 'sequential')])
        cells.append(f"{med:6.2f} [{lo:5.2f},{hi:6.2f}]")
        for lam in LAMBDAS:
            med, lo, hi = _median_range([r['mean_angle'] for r in
                                         _select(rows, label, 'joint', lam)])
            cells.append(f"{med:6.2f} [{lo:5.2f},{hi:6.2f}]")
        print(f"{label:<24}" + "".join(f"{c:>21}" for c in cells))

    print("\nRedundancy, inter-direction dCor² — median [min, max]")
    print(f"{'config':<24}" + "".join(f"{h:>21}" for h in header))
    for (k, p, nl, noise, n, label) in CONFIGS:
        cells = []
        med, lo, hi = _median_range([r['inter_dcor']
                                     for r in _select(rows, label, 'sequential')])
        cells.append(f"{med:6.4f} [{lo:.4f},{hi:.4f}]")
        for lam in LAMBDAS:
            med, lo, hi = _median_range([r['inter_dcor'] for r in
                                         _select(rows, label, 'joint', lam)])
            cells.append(f"{med:6.4f} [{lo:.4f},{hi:.4f}]")
        print(f"{label:<24}" + "".join(f"{c:>21}" for c in cells))

    all_results = rows   # the figure below reads the same rows as the tables

    # ── Per-configuration comparison at fixed λ = 0 ──────────────────────────────
    # Sequential against joint at λ = 0 is the like-for-like comparison: same
    # objective, same data, one optimised by deflation and one on the Stiefel
    # manifold. λ > 0 is a separate question, answered by the λ columns above.
    print("\n--- Sequential against joint at λ = 0, per configuration ---")
    for (k, p, nl, noise, n, label) in CONFIGS:
        seq_vals = [r['mean_angle'] for r in _select(all_results, label, 'sequential')]
        jt_vals = [r['mean_angle'] for r in _select(all_results, label, 'joint', 0.0)]
        s_med, s_lo, s_hi = _median_range(seq_vals)
        j_med, j_lo, j_hi = _median_range(jt_vals)
        # A difference is only reported as such if it survives the seed spread. The
        # withdrawn claim rested on a gap that four of five seeds did not show.
        overlap = not (s_hi < j_lo or j_hi < s_lo)
        verdict = ('indistinguishable over these seeds' if overlap else
                   ('joint lower' if j_med < s_med else 'sequential lower'))
        seq_t = float(np.median([r['time'] for r in
                                 _select(all_results, label, 'sequential')]))
        jt_t = float(np.median([r['time'] for r in
                                _select(all_results, label, 'joint', 0.0)]))
        print(f"  {label}:")
        print(f"    sequential {s_med:6.2f}° [{s_lo:.2f}, {s_hi:.2f}]   "
              f"joint {j_med:6.2f}° [{j_lo:.2f}, {j_hi:.2f}]   -> {verdict}")
        print(f"    median wall-clock: sequential {seq_t:.1f}s, joint {jt_t:.1f}s")

    print("\n--- Does λ > 0 beat λ = 0? ---")
    for (k, p, nl, noise, n, label) in CONFIGS:
        base_med, base_lo, base_hi = _median_range(
            [r['mean_angle'] for r in _select(all_results, label, 'joint', 0.0)])
        parts = []
        for lam in LAMBDAS[1:]:
            med, lo, hi = _median_range([r['mean_angle'] for r in
                                         _select(all_results, label, 'joint', lam)])
            mark = '' if hi <= base_hi else '  (worst seed worse)'
            parts.append(f"λ={lam}: {med:6.2f} [{lo:.2f}, {hi:.2f}]{mark}")
        inter0, _, _ = _median_range(
            [r['inter_dcor'] for r in _select(all_results, label, 'joint', 0.0)])
        print(f"  {label}: λ=0 {base_med:6.2f} [{base_lo:.2f}, {base_hi:.2f}], "
              f"inter-direction dCor² at λ=0 {inter0:.4f}")
        for part in parts:
            print(f"      {part}")
    print("\nThere is nothing to interpret here that the numbers do not say. The "
          "reading of them belongs in the thesis, against the seed ranges above.",
          flush=True)


if __name__ == "__main__":
    main()
