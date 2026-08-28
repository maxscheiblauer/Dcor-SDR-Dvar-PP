"""Why dVar-PP runs on the raw projection rather than the whitened one (Ch. 5 footnote).

Projection pursuit conventionally whitens before maximising the index, so that the
index reads shape stripped of variance.  In the factor model of ``data_generator``
that removes the variance lift ``sigma_signal**2 + 1`` which distinguishes the signal
direction, and the optimiser is left chasing finite-sample accidents in the pure-noise
dimensions.  This script measures the effect at the configuration the footnote quotes.

Run:  python scripts/ch05_whitening.py     ->  results/results_whitening.csv
"""

# Thesis:   Chapter 5, the §5.2 footnote
# Writes:   results/results_whitening.csv
# Original: Dvar-PP/whitening_check.py on the thesis branch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys
from pathlib import Path

import numpy as np


from dpp.unsupervised.data_generator import generate_data                   # noqa: E402
from dpp.unsupervised.dvar_optimizer import pp_dvar                         # noqa: E402

SEEDS = (42, 7, 123, 2024, 5)

# The configuration the Chapter 5 footnote names.
N, P, K = 120, 20, 1
SIGMA_SIGNAL = 2.0
DIST = "gaussian_mix"


def mss(W_true: np.ndarray, W_hat: np.ndarray) -> float:
    """Mean squared sine of the principal angles between the two subspaces."""
    s = np.linalg.svd(W_true.T @ W_hat, compute_uv=False)
    return float(np.mean(1.0 - s ** 2))


def pca_directions(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    return np.linalg.svd(Xc, full_matrices=False)[2][:k].T


def main() -> None:
    rows = []
    for seed in SEEDS:
        d = generate_data(n=N, p=P, k=K, sigma_signal=SIGMA_SIGNAL,
                          dist=DIST, seed=seed)
        X, W_true = d["X"], d["W_true"]
        row = {
            "seed": seed, "n": N, "p": P, "k": K,
            "sigma": SIGMA_SIGNAL, "dist": DIST,
            "MSS_raw": mss(W_true, pp_dvar(X, k=K, whiten=False)["W"]),
            "MSS_whitened": mss(W_true, pp_dvar(X, k=K, whiten=True)["W"]),
            "MSS_PCA": mss(W_true, pca_directions(X, K)),
        }
        rows.append(row)
        print(f"seed {seed:5d}  raw={row['MSS_raw']:.4f}  "
              f"whitened={row['MSS_whitened']:.4f}  PCA={row['MSS_PCA']:.4f}")

    print("\nmedian [min, max] over the seeds")
    for col in ("MSS_raw", "MSS_whitened", "MSS_PCA"):
        v = np.array([r[col] for r in rows])
        print(f"  {col:13s} {np.median(v):.3f}  [{v.min():.3f}, {v.max():.3f}]")

    write_csv(RESULTS / "results_whitening.csv", rows, seeds=SEEDS,
                         n=N, p=P, k=K, sigma_signal=SIGMA_SIGNAL, dist=DIST,
                         n_starts=20, max_iter=300)


if __name__ == "__main__":
    main()
