"""The equal-variance experiment of Chapter 5, over the five standard seeds.

Two latent variables of equal unit variance, one Gaussian and one bimodal, rotated
by 35 degrees into the observed plane with no noise dimensions.  The covariance
matrix is then close to the identity, so no direction stands out by spread and the
only thing distinguishing the bimodal axis is the shape of its projection.

``make_ch5_figures.py`` draws this construction at one seed (51) for
``step1_scatter.png`` and prints that seed's angles.  One replicate is not a claim,
so the numbers Chapter 5 quotes come from here instead: five seeds, median with
[min, max].

FastICA is included because it is the obvious competitor in this setting and the
factor-model argument for why it coincides with PCA does not apply here.  It does
read the shape and recovers the axis, which bounds the claim the section makes.

Run:  python scripts/ch05_equal_variance.py     ->  results/results_equalvar.csv
"""

# Thesis:   Chapter 5, §5.4
# Writes:   results/results_equalvar.csv
# Original: Dvar-PP/equalvar_study.py on the thesis branch.
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
from sklearn.decomposition import FastICA


from dpp.unsupervised.dvar_optimizer import pp_dvar                         # noqa: E402

SEEDS = (42, 7, 123, 2024, 5)

N = 1200
THETA_DEG = 35.0
#: The seed ``make_ch5_figures.py`` draws; reported alongside but not part of the
#: five-seed summary, so that the figure and the text can be checked against
#: each other.
FIGURE_SEED = 51


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Acute angle between two directions, in degrees (sign-insensitive)."""
    c = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def construct(seed: int):
    """The rotated equal-variance pair, and the true bimodal axis.

    Both columns are scaled to unit *population* variance by a fixed constant rather
    than by their sample standard deviation, so their sample variances fluctuate.
    Forcing them exactly equal would pin the PCA eigenvectors to a degenerate 45
    degrees instead of letting PCA pick the near-random axis the example is about.
    """
    rng = np.random.default_rng(seed)
    z_gauss = rng.standard_normal(N)
    sign = rng.choice([-1.0, 1.0], size=N)
    z_bimodal = (sign * 2.0 + rng.standard_normal(N)) / np.sqrt(5.0)
    Z = np.column_stack([z_gauss, z_bimodal])

    th = np.deg2rad(THETA_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Z @ R.T, R[:, 1]


def one_seed(seed: int) -> dict:
    X, true_axis = construct(seed)

    evals = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    w_pca = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)[2][0]

    kw = dict(k=1, n_starts=16, max_iter=200, lr=0.05, n_jobs=2, seed=0)
    w_raw = pp_dvar(X, whiten=False, **kw)["W"][:, 0]
    w_whitened = pp_dvar(X, whiten=True, **kw)["W"][:, 0]

    # FastICA returns both components; the axis it recovers is whichever of the two
    # mixing columns lies closer to the bimodal one.
    mixing = FastICA(n_components=2, random_state=0, whiten="unit-variance",
                     max_iter=1000).fit(X).mixing_
    ica = min(angle_deg(mixing[:, 0], true_axis),
              angle_deg(mixing[:, 1], true_axis))

    return {
        "seed": seed, "n": N, "theta_deg": THETA_DEG,
        "eigengap": float(abs(evals[1] - evals[0])),
        "angle_PCA": angle_deg(w_pca, true_axis),
        "angle_dvar_raw": angle_deg(w_raw, true_axis),
        "angle_dvar_whitened": angle_deg(w_whitened, true_axis),
        "angle_FastICA": ica,
    }


COLUMNS = ("eigengap", "angle_PCA", "angle_dvar_raw",
           "angle_dvar_whitened", "angle_FastICA")


def main() -> None:
    rows = [one_seed(s) for s in SEEDS]
    for r in rows:
        print(f"seed {r['seed']:5d}  gap={r['eigengap']:.3f}  "
              f"PCA={r['angle_PCA']:5.1f}  raw={r['angle_dvar_raw']:4.1f}  "
              f"whitened={r['angle_dvar_whitened']:4.1f}  "
              f"FastICA={r['angle_FastICA']:4.1f}")

    print("\nmedian [min, max] over the seeds")
    for col in COLUMNS:
        v = np.array([r[col] for r in rows])
        print(f"  {col:20s} {np.median(v):6.2f}  [{v.min():.2f}, {v.max():.2f}]")

    fig = one_seed(FIGURE_SEED)
    print(f"\nfigure seed {FIGURE_SEED} (step1_scatter.png, not in the summary): "
          f"gap={fig['eigengap']:.3f}  PCA={fig['angle_PCA']:.1f}  "
          f"raw={fig['angle_dvar_raw']:.1f}  "
          f"whitened={fig['angle_dvar_whitened']:.1f}")
    rows.append(fig)

    write_csv(RESULTS / "results_equalvar.csv", rows, seeds=SEEDS,
                         n=N, theta_deg=THETA_DEG, figure_seed=FIGURE_SEED,
                         n_starts=16, max_iter=200)


if __name__ == "__main__":
    main()
