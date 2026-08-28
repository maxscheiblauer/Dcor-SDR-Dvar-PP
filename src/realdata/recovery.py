"""
outlier_recovery.py — can a variance criterion, a robust variance criterion, a
distance-variance criterion or its bounded variant single out the two months of
the September 2001 shock, when each is fitted on the *whole* sample?

Motivation
----------
Chapter 6 reports an unplanned finding: the direction that maximises distance
variance on the training half of the published Ludvigson-Ng factor block places
September and October 2001 far outside everything the training half contains,
while the first principal component of the same block places them at ordinary
values, well inside the training range. Two qualifications were attached to it.
The direction was fitted on the training half, so the two months had no influence
on the choice of axis, and the finding was never posed as a detection problem: no
method was asked to look for those months.

This script poses it as one. The train/test split is dropped — every method sees
all 480 months, 1964:01 to 2003:12 — and the question becomes whether a criterion
that is free to use the two shock months when choosing its direction produces a
direction on which those two months are the most extreme observations in the
sample. Four criteria are compared:

    PCA          the leading eigenvectors of the full-sample panel covariance
    MCD-PCA      the leading eigenvectors of the minimum-covariance-determinant
                 covariance; the robust Mahalanobis distance of the same fit is
                 recorded separately, since MCD is a detector in its own right
                 and "the projection misses the pair but the distance finds it"
                 is a different statement from both missing it
    dVar-PP      maximise dVar(X w) over the sphere, sequential and joint
    dVar-PP      the same with the Leyder-Raymaekers-Rousseeuw biloop transform
    (biloop)     inserted before the index, which bounds the influence of a
                 single observation on the objective

Input representation
--------------------
All four run inside the leading 8-dimensional subspace of the standardised macro
panel — the space Chapter 6 works in — but that subspace is fed to them in two
bases, because the basis is not neutral:

    unit    the published factor block f1..f8 of RFS2009.xls, normalised so that
            F'F / T = I. This is exactly Chapter 6's input. Its sample covariance
            is 1.0021 * I to four decimals, so *classical* PCA inside this basis
            is degenerate: all eight eigenvalues coincide and the eigenvectors an
            eigensolver returns are an arbitrary rotation. PCA is therefore never
            solved inside this basis. Its directions are the panel's principal
            axes, which in this basis are the coordinate axes themselves — the
            j-th direction is f_j — and the script asserts the degeneracy rather
            than letting an eigensolver hide it.
    scaled  the same subspace in its natural principal-component coordinates,
            each column carrying its own eigenvalue as variance. Classical PCA is
            well posed here and returns the coordinate axes by construction.

The two bases differ by a diagonal rescaling. PCA is unaffected by it, so its
rows are identical in both. dVar is scale-equivariant and MCD is not
scale-invariant either, so both can and do respond to the rescaling: `scaled`
answers the obvious objection that whatever dVar-PP finds in the whitened basis
is an artefact of the whitening. The wider ~130-series panel is out of scope
here, because MCD is not usable at p = 130 with n = 480 and the comparison would
stop being four criteria on one input.

What is measured
----------------
For every direction w of every method the score g = X w is standardised with the
median and the MAD (scaled to be consistent at the Gaussian) and, for each of the
two months, the script records the standardised score and the rank of |z| among
all 480 months, rank 1 being the most extreme month in the sample. Median and MAD
are used rather than mean and standard deviation because the quantity being
measured is how far two observations sit from the bulk, and the mean and standard
deviation of the sample are themselves moved by those two observations.

A method "isolates the pair" when the two months occupy ranks 1 and 2.
`sep_margin` is the smaller of the pair's two |z| values divided by the largest
|z| among the other 478 months: above 1 the pair is the two most extreme months,
and how far above says how cleanly they stand apart. The five most extreme months
are written out as dates, so what a criterion singles out *instead* is visible
when it does not single out the pair.

Three seeds, as Rule 5d of CLAUDE.md requires. They move the dVar-PP restarts and
the MCD subsampling; PCA is deterministic and its rows repeat unchanged, which is
worth having in the file.

Note on vocabulary: dVar is a dispersion / shape index, not a measure of
non-Gaussianity. Nothing here assumes otherwise — the reason a direction that
isolates two points is a candidate maximiser is that dVar rises as mass moves
away from the centre.

Run:  python scripts/ch06_outlier_recovery.py
Writes: results/outlier_recovery.csv, results/outlier_recovery_agg.csv,
        thesis/graphics/ch6_outlier_recovery.png
"""
from __future__ import annotations

from csvout import RESULTS, write_csv

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                     # Real Data Experiment/
_REPO = _ROOT.parent                     # Python/

import realdata.dataio as dio                                           # noqa: E402
from realdata.pca import standardize, pc_T                              # noqa: E402
from dpp.unsupervised.dvar_optimizer import pp_dvar, dvar, dvar_biloop          # noqa: E402

warnings.filterwarnings("ignore")

#: The two months of the September 2001 shock — the pair every method is scored on.
TARGET_MONTHS = ("2001-09-01", "2001-10-01")

#: Directions reported per method. Three answers whether a criterion finds the pair
#: on its leading direction, on a later one, or not at all, and keeps the biloop
#: path — whose gradient is a central difference, so p index evaluations per step —
#: at about a minute per configuration.
K = 3

#: Dimension of the subspace, matching Chapter 6's factor block.
NFAC = 8

SEEDS = (42, 7, 123)

#: Restart and iteration budgets. The plain path has an analytical gradient and gets
#: the budget used elsewhere in the project; the biloop path costs p times more per
#: step and gets the reduced budget that `dvar_outlier_diagnostic.py` and
#: `robustness.py` both use for it. Measured on this machine at n = 480, p = 8:
#: 5 s per plain configuration, 65 s per biloop configuration.
BUDGET = {
    "plain": dict(n_starts=20, max_iter=300),
    "robust": dict(n_starts=8, max_iter=80),
}

MAD_SCALE = 1.4826       # makes the MAD consistent for the Gaussian sd

#: Relative eigenvalue spread below which the sample covariance has no separated
#: leading eigenvalue, so solving for principal axes inside the basis is meaningless.
#: Used to assert the `unit` basis really is the degenerate case the module docstring
#: claims it is. Measured there: 2.47e-05, against 8.45e-01 in the `scaled` basis —
#: four orders of magnitude apart, so the threshold is not a delicate choice.
ISOTROPY_TOL = 1e-3


# --------------------------------------------------------------------- input
def blocks():
    """The two bases of the leading 8-dimensional subspace, and the date index.

    `unit` is read from RFS2009.xls rather than recomputed: it is the published
    object and `stage0_validate.py` is the gate that it agrees with a local
    re-extraction to |corr| = 1.00000. `scaled` is that re-extraction, kept in its
    natural principal-component coordinates.
    """
    a = dio.aligned_panel()
    tgt = a["targets"]
    F_unit = tgt[[f"f{i}" for i in range(1, NFAC + 1)]].to_numpy(dtype=float)
    assert not np.isnan(F_unit).any(), "published factor block has missing values"

    Ystd, _, _ = standardize(a["Y"])
    fhat, lam, sv = pc_T(Ystd, nfac=NFAC)
    # pc_T returns unit-variance scores; restore the eigenvalue scaling so that the
    # covariance of the block has NFAC distinct eigenvalues and PCA is well posed.
    eig = sv[:NFAC] / Ystd.shape[0]
    F_scaled = fhat * np.sqrt(eig)

    dates = pd.to_datetime(pd.Series(a["dates"])).to_numpy()
    return {"unit": F_unit, "scaled": F_scaled}, dates, eig


def target_rows(dates):
    """Row indices of the two months, in the order of TARGET_MONTHS."""
    d = pd.to_datetime(pd.Series(dates))
    rows = []
    for s in TARGET_MONTHS:
        hit = np.where(d == pd.Timestamp(s))[0]
        assert hit.size == 1, f"{s} not found exactly once in the sample"
        rows.append(int(hit[0]))
    return rows


# ------------------------------------------------------------------ scoring
def score_direction(g, rows, dates):
    """Extremeness of the two months on one projection score vector."""
    g = np.asarray(g, dtype=float).reshape(-1)
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med)))
    scale = MAD_SCALE * mad if mad > 1e-12 else float(g.std(ddof=1))
    z = (g - med) / scale
    absz = np.abs(z)

    order = np.argsort(-absz, kind="stable")      # rank 1 = most extreme month
    rank = np.empty(len(g), dtype=int)
    rank[order] = np.arange(1, len(g) + 1)

    others = np.delete(absz, rows)
    return {
        "z_sep2001": float(z[rows[0]]),
        "z_oct2001": float(z[rows[1]]),
        "rank_sep2001": int(rank[rows[0]]),
        "rank_oct2001": int(rank[rows[1]]),
        "worse_rank": int(max(rank[rows[0]], rank[rows[1]])),
        "pair_isolated": bool(set(rank[rows].tolist()) == {1, 2}),
        "sep_margin": float(absz[rows].min() / others.max()),
        "dvar": dvar(g),
        "dvar_biloop": dvar_biloop(g),
        "top5_months": " ".join(
            pd.Timestamp(dates[i]).strftime("%Y-%m") for i in order[:5]),
    }


def frame_records(method, basis, strategy, seed, X, W, order, crit, rows, dates):
    """One record per reported direction of one fitted method."""
    recs = []
    for pos, j in enumerate(order[:K]):
        rec = dict(method=method, basis=basis, strategy=strategy, seed=seed,
                   direction=pos + 1, criterion=float(crit[j]))
        rec.update(score_direction(X @ W[:, j], rows, dates))
        rec["detail"] = " ".join(f"{v:+.3f}" for v in W[:, j])
        recs.append(rec)
    return recs


# ------------------------------------------------------------------ methods
def pca_frame(X):
    """Principal axes of the panel, expressed in the basis `X` is given in.

    Both bases are already principal-component coordinates of the standardised
    panel, so the principal axes are the coordinate axes and the j-th direction is
    the j-th column. Ordering comes from the panel eigenvalues, which the `scaled`
    basis carries as its column variances. Solving an eigenproblem on cov(X) is
    deliberately avoided: in the `unit` basis that covariance is isotropic and the
    eigenvectors returned would be an arbitrary rotation.
    """
    p = X.shape[1]
    return np.eye(p), np.arange(p), X.var(axis=0, ddof=1)


def isotropy(X):
    """Relative spread of the eigenvalues of cov(X): 0 means perfectly isotropic."""
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    return float((ev.max() - ev.min()) / ev.max())


def fit_mcd(X, seed):
    """MCD covariance eigenvectors, ordered by robust eigenvalue, plus the fit.

    `support_fraction=None` takes scikit-learn's default (n + p + 1) / 2, the
    highest-breakdown choice. Unlike the classical covariance in the `unit` basis
    this is not isotropic: the subsample it is computed on excludes the months
    that inflate particular coordinates, so its eigenvalues separate.
    """
    mcd = MinCovDet(support_fraction=None, random_state=seed).fit(X)
    ev, V = np.linalg.eigh(mcd.covariance_)
    return V, np.argsort(ev)[::-1], ev, mcd


def fit_dvar(X, index_fun, strategy, seed):
    """dVar-PP on the full sample; directions ordered by their own index value."""
    fit = pp_dvar(X, k=K, index_fun=index_fun, strategy=strategy, whiten=False,
                  n_jobs=-1, seed=seed, **BUDGET[index_fun])
    crit = np.asarray(fit["obj"], dtype=float)
    return fit["W"], np.argsort(crit)[::-1], crit


def mahalanobis_record(mcd, X, basis, rows, dates, seed):
    """MCD robust distance as a plain detector — no projection involved.

    Recorded with direction 0 to keep it distinguishable from the projection rows.
    `z_*` hold the robust distance itself, not a standardised score, and the two
    dVar columns are left empty: there is no single direction to evaluate an index
    on. `detail` reports how many months exceed the 0.975 quantile of chi-squared
    on 8 degrees of freedom, the conventional cutoff.
    """
    d2 = mcd.mahalanobis(X)
    order = np.argsort(-d2, kind="stable")
    rank = np.empty(len(d2), dtype=int)
    rank[order] = np.arange(1, len(d2) + 1)
    cut = float(chi2.ppf(0.975, X.shape[1]))
    return dict(
        method="MCD distance", basis=basis, strategy="-", seed=seed, direction=0,
        criterion=cut,
        z_sep2001=float(np.sqrt(d2[rows[0]])), z_oct2001=float(np.sqrt(d2[rows[1]])),
        rank_sep2001=int(rank[rows[0]]), rank_oct2001=int(rank[rows[1]]),
        worse_rank=int(max(rank[rows[0]], rank[rows[1]])),
        pair_isolated=bool(set(rank[rows].tolist()) == {1, 2}),
        sep_margin=float(d2[rows].min() / np.delete(d2, rows).max()),
        dvar=float("nan"), dvar_biloop=float("nan"),
        top5_months=" ".join(pd.Timestamp(dates[i]).strftime("%Y-%m")
                             for i in order[:5]),
        detail=f"flagged at chi2_0.975: {int((d2 > cut).sum())} of {len(d2)} months",
    )


DVAR_CONFIGS = (("plain", "dVar-PP"), ("robust", "dVar-PP (biloop)"))


def run_seed(X, basis, seed, rows, dates):
    """Every method on one basis at one seed."""
    recs = []

    V, order, crit = pca_frame(X)
    recs += frame_records("PCA", basis, "-", seed, X, V, order, crit, rows, dates)

    V, order, ev, mcd = fit_mcd(X, seed)
    recs += frame_records("MCD-PCA", basis, "-", seed, X, V, order, ev, rows, dates)
    recs.append(mahalanobis_record(mcd, X, basis, rows, dates, seed))

    for index_fun, label in DVAR_CONFIGS:
        for strategy in ("sequential", "joint"):
            W, order, crit = fit_dvar(X, index_fun, strategy, seed)
            recs += frame_records(label, basis, strategy, seed, X, W, order, crit,
                                  rows, dates)
    return recs


# ----------------------------------------------------------------------- main
def main():
    B, dates, eig = blocks()
    rows = target_rows(dates)
    print(f"input: leading {NFAC} PCs of the standardised panel, "
          f"{B['unit'].shape[0]} months "
          f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}")
    print(f"target pair: {TARGET_MONTHS[0][:7]}, {TARGET_MONTHS[1][:7]} "
          f"at rows {rows}")
    print(f"panel eigenvalues: {np.round(eig, 4)}")
    print(f"isotropy of cov(X): unit={isotropy(B['unit']):.2e}  "
          f"scaled={isotropy(B['scaled']):.2e}")
    assert isotropy(B["unit"]) < ISOTROPY_TOL, (
        "the unit basis is not isotropic — the docstring's reason for not solving "
        "an eigenproblem there no longer holds")
    assert isotropy(B["scaled"]) > 0.1, "the scaled basis has no separated axes"
    print()

    recs = []
    for basis in ("unit", "scaled"):
        for seed in SEEDS:
            new = run_seed(B[basis], basis, seed, rows, dates)
            recs += new
            for r in new:
                if r["direction"] in (0, 1):
                    print(f"  {basis:7s} seed {seed:4d}  {r['method']:18s} "
                          f"{r['strategy']:10s} dir {r['direction']}  ranks "
                          f"({r['rank_sep2001']:3d},{r['rank_oct2001']:3d})  "
                          f"margin {r['sep_margin']:6.2f}  "
                          f"isolated={r['pair_isolated']}")
            print(flush=True)

    df = pd.DataFrame(recs)
    write_csv(RESULTS / "outlier_recovery.csv", df,
                         seeds=list(SEEDS),
                         script="ch06_outlier_recovery.py",
                         K=K, budget=str(BUDGET))

    # Median with [min, max] over the seeds. aggfunc is named explicitly: pandas
    # defaults to the mean, which on three replicates with one outlier is the wrong
    # summary (CLAUDE.md Rule 5d).
    agg = (df.groupby(["basis", "method", "strategy", "direction"])
             .agg(worse_rank_med=("worse_rank", "median"),
                  worse_rank_min=("worse_rank", "min"),
                  worse_rank_max=("worse_rank", "max"),
                  margin_med=("sep_margin", "median"),
                  margin_min=("sep_margin", "min"),
                  margin_max=("sep_margin", "max"),
                  isolated_seeds=("pair_isolated", "sum"),
                  n_seeds=("seed", "nunique"))
             .reset_index())
    print("across seeds — worse_rank is the poorer of the two months' ranks, "
          "rank 1 being the most extreme of the 480 months:")
    print(agg.to_string(index=False))
    write_csv(RESULTS / "outlier_recovery_agg.csv", agg,
                         seeds=list(SEEDS),
                         script="ch06_outlier_recovery.py")

    print("\nSaved ch6_outlier_recovery.png")


if __name__ == "__main__":
    main()
