"""multigroup_study.py — recovery when the latent plane holds several groups.

Written 2026-08-23, for the supervisor mark on printed page 61 of the Chapter 5
feedback: "I think GMM should here deserve more attention. Perhaps even GMM with
multiple groups. Then seq. direction might separate different groups."

What the construction is, and why it is not ``gaussian_mix`` again
------------------------------------------------------------------
``gaussian_mix`` at k = 2 already produces four clusters, but they sit at the
corners of an axis-aligned square: the two latent coordinates are independent
two-point mixtures, so the group structure factorises and each latent axis splits
the same pair of groups twice over. That is a product of two one-dimensional
problems, and it cannot answer the question the mark asks.

``dist="ring_mix"`` places ``n_groups`` equally weighted components on a circle in
the latent plane. For ``n_groups != 4`` the centres cannot factorise across
coordinates at all, so different directions in the plane separate different pairs
of groups, and "does the second direction split a different pair than the first"
becomes a question with an answer.

What is measured
----------------
Primary, and on the same scale as every other table in Chapter 5: the mean squared
sine of the principal angles between the recovered pair of directions and the true
signal plane, for dVar-PP (sequential and joint), PCA and FastICA.

Secondary, and specific to this mark: for each recovered direction, which pair of
group centres it separates best. Each unordered pair of centres defines a
separating axis in the latent plane, mapped into observed space by ``W_true``; the
pair a direction is assigned to is the one whose axis it aligns with most closely,
and ``sep_cos`` records that alignment. Two directions that carry the same
information land on the same pair; two that carry different information do not.

Runtime: about a minute per seed, so a few minutes for the five.

Run:  python scripts/ch05_multigroup.py     ->  results/results_multigroup.csv
"""

# Thesis:   Chapter 5, §5.5
# Writes:   results/results_multigroup.csv
# Original: Dvar-PP/multigroup_study.py on the thesis branch.
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
from itertools import combinations
from pathlib import Path

import numpy as np


from dpp.unsupervised.data_generator import generate_data                   # noqa: E402
from dpp.unsupervised.dvar_optimizer import pp_dvar                         # noqa: E402
from dpp.unsupervised.evaluation import (mss_principal, pca_directions,     # noqa: E402
                        fastica_directions)

OUT = RESULTS / "results_multigroup.csv"

SEEDS = (42, 7, 123, 2024, 5)

#: Three and five groups, both of which break the axis-aligned factorisation that
#: four groups on a ring would restore.
GROUP_COUNTS = (3, 5)
DIMENSIONS = (10, 50)
SIGMAS = (1.0, 2.0, 4.0)
N = 400
K = 2

N_STARTS = 20
MAX_ITER = 300
LR = 0.05
INDEX_FUN = "plain"
STRATEGIES = ("sequential", "joint")

COLUMNS = ["seed", "n_groups", "p", "sigma", "strategy",
           "MSS_dVar", "MSS_PCA", "MSS_FastICA",
           "pair_dir1", "pair_dir2", "sep_cos_dir1", "sep_cos_dir2",
           "distinct_pairs", "restart_spread", "elapsed"]


def separating_axes(centres: np.ndarray, W_true: np.ndarray):
    """One unit axis in observed space per unordered pair of group centres.

    The axis separating two groups is the direction of the difference of their
    centres, carried from the latent plane into observed space by ``W_true``.
    """
    axes, labels = [], []
    for i, j in combinations(range(centres.shape[0]), 2):
        d = W_true @ (centres[i] - centres[j])
        nrm = np.linalg.norm(d)
        if nrm > 0:
            axes.append(d / nrm)
            labels.append(f"{i}-{j}")
    return np.column_stack(axes), labels


def assign_pair(w: np.ndarray, axes: np.ndarray, labels: list[str]):
    """The group pair whose separating axis direction ``w`` aligns with best."""
    cos = np.abs(axes.T @ (w / np.linalg.norm(w)))
    best = int(np.argmax(cos))
    return labels[best], float(cos[best])


def run_one(seed: int, n_groups: int, p: int, sigma: float,
            strategy: str) -> dict:
    dat = generate_data(n=N, p=p, k=K, sigma_signal=sigma, dist="ring_mix",
                        n_groups=n_groups, seed=seed)
    X, W_true, centres = dat["X"], dat["W_true"], dat["centres"]

    t0 = time.time()
    res = pp_dvar(X, k=K, index_fun=INDEX_FUN, strategy=strategy,
                  n_starts=N_STARTS, max_iter=MAX_ITER, lr=LR, seed=seed)
    elapsed = time.time() - t0
    W_hat = res["W"]

    axes, labels = separating_axes(centres, W_true)
    p1, c1 = assign_pair(W_hat[:, 0], axes, labels)
    p2, c2 = assign_pair(W_hat[:, 1], axes, labels)

    spread = res.get("spread")
    if isinstance(spread, list) and spread:
        width = float(max(hi - lo for lo, hi in spread))
    elif spread is not None:
        lo, hi = spread
        width = float(hi - lo)
    else:
        width = float("nan")

    return {
        "seed": seed, "n_groups": n_groups, "p": p, "sigma": sigma,
        "strategy": strategy,
        "MSS_dVar": mss_principal(W_true, W_hat),
        "MSS_PCA": mss_principal(W_true, pca_directions(X, K)),
        "MSS_FastICA": mss_principal(W_true, fastica_directions(X, K, seed=seed)),
        "pair_dir1": p1, "pair_dir2": p2,
        "sep_cos_dir1": c1, "sep_cos_dir2": c2,
        "distinct_pairs": bool(p1 != p2),
        "restart_spread": width,
        "elapsed": elapsed,
    }


def main() -> None:
    configs = [dict(n_groups=g, p=p, sigma=s, strategy=st)
               for g in GROUP_COUNTS for p in DIMENSIONS for s in SIGMAS
               for st in STRATEGIES]
    print(f"{len(configs)} configurations x {len(SEEDS)} seeds = "
          f"{len(configs) * len(SEEDS)} runs "
          f"({len(GROUP_COUNTS)} group counts x {len(DIMENSIONS)} dimensions x "
          f"{len(SIGMAS)} signal strengths x {len(STRATEGIES)} strategies)",
          flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for seed in SEEDS:
        for cfg in configs:
            rows.append(run_one(seed=seed, **cfg))
            r = rows[-1]
            print(f"  seed {seed:5d}  G={r['n_groups']} p={r['p']:<3} "
                  f"sigma={r['sigma']:<4} {r['strategy']:10s} "
                  f"MSS={r['MSS_dVar']:.4f} (PCA {r['MSS_PCA']:.4f}, "
                  f"ICA {r['MSS_FastICA']:.4f})  pairs {r['pair_dir1']}/"
                  f"{r['pair_dir2']}  [{time.time() - t0:.0f}s]", flush=True)

    record = write_csv(OUT, rows, seeds=SEEDS, fieldnames=COLUMNS,
                                  script="ch05_multigroup.py",
                                  n=N, k=K, n_starts=N_STARTS,
                                  max_iter=MAX_ITER, index_fun=INDEX_FUN)
    print(f"\nWrote {OUT.name}: {len(rows)} rows", flush=True)

    print("\nmedian MSS over the seeds")
    for g in GROUP_COUNTS:
        for st in STRATEGIES:
            sub = [r for r in rows if r["n_groups"] == g and r["strategy"] == st]
            v = np.array([r["MSS_dVar"] for r in sub])
            pca = np.array([r["MSS_PCA"] for r in sub])
            frac = float(np.mean([r["distinct_pairs"] for r in sub]))
            print(f"  G={g} {st:10s} dVar {np.median(v):.3f} "
                  f"[{v.min():.3f}, {v.max():.3f}]  PCA {np.median(pca):.3f}  "
                  f"distinct pairs in {frac:.0%} of runs")


if __name__ == "__main__":
    main()
