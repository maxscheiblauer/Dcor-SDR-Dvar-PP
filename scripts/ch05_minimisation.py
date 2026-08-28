"""minimisation_study.py — what the index finds when it is minimised.

Written 2026-08-23, for the supervisor mark on printed page 56 of the Chapter 5
feedback: "Would this suggest that one could try to both minimize and maximize it?
E.g. like kurtosis in ICA? Then see what is more interesting."

The question the study answers
------------------------------
Distance variance at unit variance orders distributions by how far their mass sits
from the centre, and the Gaussian sits in the middle of that ordering (0.634):
bimodal 0.754 and uniform 0.730 above it, Laplace 0.540 and t3 0.476 below it.
Chapter 5 states this and draws a consequence from it, that a maximiser can only
find the shapes on the upper side, but it never measures the consequence. That is
what this file does.

Two equal-variance constructions, both rotated by the same angle into the observed
plane, both with no variance gap for PCA to use:

  ``bimodal``  one Gaussian axis and one bimodal axis (dVar 0.754, above baseline)
  ``heavy``    one Gaussian axis and one t3 axis      (dVar 0.476, below baseline)

On ``bimodal`` the maximiser should find the interesting axis and the minimiser
should not. On ``heavy`` the prediction reverses: the interesting axis is the one
the maximiser is by construction searching away from, and only the minimiser can
be expected to land on it. FastICA is reported alongside because its own criterion
is non-Gaussianity, which is symmetric about the Gaussian in a way distance
variance is not, so it should find both.

Each column is scaled to unit *population* variance by a fixed constant rather than
by its sample standard deviation, as in ``equalvar_study.py``: forcing the sample
variances exactly equal would pin the PCA eigenvectors at 45 degrees rather than
letting PCA pick the near-random axis these constructions are about.

Runtime: about a minute for the five seeds.

Run:  python scripts/ch05_minimisation.py     ->  results/results_minimisation.csv
"""

# Thesis:   Chapter 5, §5.4
# Writes:   results/results_minimisation.csv
# Original: Dvar-PP/minimisation_study.py on the thesis branch.
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


from dpp.unsupervised.dvar_optimizer import pp_dvar, dvar                   # noqa: E402

OUT = RESULTS / "results_minimisation.csv"

SEEDS = (42, 7, 123, 2024, 5)

N = 1200
THETA_DEG = 35.0
CONSTRUCTIONS = ("bimodal", "heavy")
SENSES = ("max", "min")

N_STARTS = 16
MAX_ITER = 200
LR = 0.05

COLUMNS = ["seed", "construction", "sense", "n", "theta_deg", "eigengap",
           "angle_target", "angle_gaussian_axis", "dvar_at_solution",
           "dvar_target_axis", "dvar_gaussian_axis",
           "angle_PCA", "angle_FastICA"]


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Acute angle between two directions, in degrees (sign-insensitive)."""
    c = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def construct(seed: int, construction: str):
    """The rotated equal-variance pair, its target axis and its Gaussian axis."""
    rng = np.random.default_rng(seed)
    z_gauss = rng.standard_normal(N)
    if construction == "bimodal":
        sign = rng.choice([-1.0, 1.0], size=N)
        z_target = (sign * 2.0 + rng.standard_normal(N)) / np.sqrt(5.0)
    elif construction == "heavy":
        # Student t3 has population variance 3.
        z_target = rng.standard_t(df=3, size=N) / np.sqrt(3.0)
    else:
        raise ValueError(f"Unknown construction: {construction!r}.")

    Z = np.column_stack([z_gauss, z_target])
    th = np.deg2rad(THETA_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Z @ R.T, R[:, 1], R[:, 0]


def one_run(seed: int, construction: str, sense: str) -> dict:
    X, target_axis, gauss_axis = construct(seed, construction)

    evals = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    w_pca = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)[2][0]

    w = pp_dvar(X, k=1, whiten=False, n_starts=N_STARTS, max_iter=MAX_ITER,
                lr=LR, n_jobs=2, seed=0, sense=sense)["W"][:, 0]

    mixing = FastICA(n_components=2, random_state=0, whiten="unit-variance",
                     max_iter=1000).fit(X).mixing_
    ica = min(angle_deg(mixing[:, 0], target_axis),
              angle_deg(mixing[:, 1], target_axis))

    return {
        "seed": seed, "construction": construction, "sense": sense,
        "n": N, "theta_deg": THETA_DEG,
        "eigengap": float(abs(evals[1] - evals[0])),
        "angle_target": angle_deg(w, target_axis),
        "angle_gaussian_axis": angle_deg(w, gauss_axis),
        "dvar_at_solution": float(dvar(X @ w)),
        "dvar_target_axis": float(dvar(X @ target_axis)),
        "dvar_gaussian_axis": float(dvar(X @ gauss_axis)),
        "angle_PCA": angle_deg(w_pca, target_axis),
        "angle_FastICA": ica,
    }


def main() -> None:
    rows = [one_run(s, c, sense)
            for s in SEEDS for c in CONSTRUCTIONS for sense in SENSES]
    for r in rows:
        print(f"  seed {r['seed']:5d}  {r['construction']:8s} {r['sense']:3s}  "
              f"angle to target {r['angle_target']:5.1f}  "
              f"to Gaussian axis {r['angle_gaussian_axis']:5.1f}  "
              f"dVar {r['dvar_at_solution']:.3f} "
              f"(target {r['dvar_target_axis']:.3f}, "
              f"Gaussian {r['dvar_gaussian_axis']:.3f})", flush=True)

    print("\nmedian [min, max] over the seeds, angle to the target axis")
    for c in CONSTRUCTIONS:
        for sense in SENSES:
            v = np.array([r["angle_target"] for r in rows
                          if r["construction"] == c and r["sense"] == sense])
            print(f"  {c:8s} {sense:3s}  {np.median(v):5.1f}  "
                  f"[{v.min():.1f}, {v.max():.1f}]")
        ica = np.array([r["angle_FastICA"] for r in rows
                        if r["construction"] == c and r["sense"] == "max"])
        pca = np.array([r["angle_PCA"] for r in rows
                        if r["construction"] == c and r["sense"] == "max"])
        print(f"  {c:8s} ICA  {np.median(ica):5.1f}  "
              f"[{ica.min():.1f}, {ica.max():.1f}]   "
              f"PCA {np.median(pca):5.1f}  [{pca.min():.1f}, {pca.max():.1f}]")

    record = write_csv(OUT, rows, seeds=SEEDS, fieldnames=COLUMNS,
                                  script="ch05_minimisation.py",
                                  n=N, theta_deg=THETA_DEG, n_starts=N_STARTS,
                                  max_iter=MAX_ITER)
    print(f"\nWrote {OUT.name}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
