"""
holdout_recovery.py — Design A: delete the two months, refit, project them back.

The question
------------
`outlier_recovery.py` fits every criterion on all 480 months and finds that no
projection criterion puts September and October 2001 at ranks 1 and 2: dVar-PP's
leading direction leaves them at ranks 133 and 128 (sequential) and 378 and 144
(joint), further into the bulk than PCA's leading direction, which has them at 47
and 59. Only the MCD robust Mahalanobis distance isolates the pair, at ranks
(2, 1) with a margin of 1.31 to 1.94 over three seeds.

That result has a confound, and Chapter 6's has a different one. Fitting on the
whole sample means the pair is in the sample the index is computed on, and a
bulk-plus-two-spikes shape is a heavy tail, which sits *below* the Gaussian in the
dVar ordering — so the criterion is pushed away from the very direction that would
expose the pair. Chapter 6 instead fits on 1964--1983, where the pair is absent,
but that fitting sample is also a different macroeconomic era, so a difference
between the two studies cannot be attributed to the pair rather than to the era.

This script removes both confounds. Each criterion is fitted on the 478 months
that are *not* September or October 2001 — the era is intact, only the pair is
gone — then frozen, and the pair is projected onto the direction it chose. The
question becomes: on the axis a criterion picks without ever seeing those two
months, do they land outside everything it was fitted on?

Pre-registered decision rule
----------------------------
Fixed before the run, so that no configuration can be chosen after seeing the
numbers.

* The statistic is `margin`: the smaller of the pair's two |z| values divided by
  the largest |z| among the 478 fitting months, with z taken in the fitting
  sample's median/MAD units. Above 1 the pair is more extreme than anything the
  criterion was fitted on; `pair_isolated` is exactly `margin > 1`.
* dVar-PP is credited with isolating the pair only if, over the three seeds, its
  median margin exceeds 1 while the medians for PCA and MCD-PCA stay below 1, and
  the seed ranges do not straddle 1.
* Every configuration is reported: three bases, two strategies, two indices,
  three directions, three seeds. No basis, strategy, seed or direction is selected
  after the fact.
* The MCD robust distance stays in as a reference arm and its outcome is reported
  whatever the projection arms do. It already isolates the pair in the
  fitted-with-pair study, so a claim that "PCA does not find them" is false for
  robust PCA as a detector and is not made here.
* The rows of `outlier_recovery.csv` are the fitted-with-pair arm of the same
  comparison. Nothing in this file replaces them.

Bases
-----
Three, because a direction fitted without the pair is only leak-free if the
subspace it lives in is too:

    published   the published f1..f8 block of RFS2009.xls. Extracted over the
                full sample, so the pair influenced the *subspace* even though it
                is excluded from the direction fit. Included because Chapter 6's
                numbers live in this space and it is the only basis in which this
                study and that one are directly comparable.
    fit_unit    the leading 8 PCs re-extracted from the 478 fitting months via
                `TrainPCA` — mean, standard deviation and eigenvectors all fitted
                without the pair, columns normalised to unit variance on the
                fitting sample. Fully leak-free, and the closest leak-free
                counterpart of `published`.
    fit_scaled  the same re-extraction with each column carrying its own
                eigenvalue as variance. dVar is scale-equivariant and MCD is not
                scale-invariant, so the rescaling can change both; PCA is
                unaffected by it.

Classical PCA appears in two arms, for a reason the holdout itself creates. All
three bases are principal-component coordinates, so the basis's own axis order is
one reading of "what PCA points at" — the reading Chapter 6 uses when it speaks of
the first PCA factor. But holding the pair out is not a small perturbation of the
published block: the covariance of all 480 of its rows is isotropic to 2.5e-05,
and the covariance of the 478 is 6.5e-01, so those two months carry a large share
of that block's variance structure and its coordinate axes are no longer its
principal axes. The second arm, "PCA (refit)", therefore solves the eigenproblem
on the fitting rows inside the basis, and is skipped only where that covariance is
isotropic and an eigensolver would return an arbitrary rotation — `fit_unit` by
construction, measured at 4.7e-15.

Reported quantities
-------------------
`margin` and `pair_isolated` are the pre-registered pair. Alongside them, in the
fitting sample's own frame:

    z_*_mad         (score - median_fit) / (1.4826 * MAD_fit)
    z_*_sd          (score - mean_fit) / sd_fit          <- Chapter 6's frame
    excess_*_sd     distance beyond the nearer edge of the fitting range, in
                    sd_fit units, and 0 when the month is inside the range
    outside_support both months beyond the fitting range

Chapter 6 reports the pair at 3.6 and 8.2 training standard deviations on the
joint dVar direction, against 1.2 and 1.1 on the first PCA factor; its prose
measures from the training mean and `dvar_outlier_diagnostic.py` measures from the
support edge, so both are given here.

Ranks among all 480 months are carried over unchanged from
`outlier_recovery.py`, so the two files can be read side by side.

Run:  python scripts/ch06_holdout_recovery.py
Writes: results/holdout_recovery.csv, results/holdout_recovery_agg.csv,
        thesis/graphics/ch6_holdout_recovery.png
"""
from __future__ import annotations

from csvout import RESULTS, write_csv

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_REPO = _ROOT.parent

import realdata.dataio as dio                                           # noqa: E402
from realdata.pca import TrainPCA                                       # noqa: E402
from dpp.unsupervised.dvar_optimizer import dvar, dvar_biloop                   # noqa: E402

# The fitted-with-pair study is the sibling arm; its methods, budgets, direction
# count and MAD scaling are reused rather than restated, so the two files cannot
# drift apart.
from realdata.recovery import (TARGET_MONTHS, K, NFAC, BUDGET, MAD_SCALE,
                             ISOTROPY_TOL, DVAR_CONFIGS, target_rows, pca_frame,
                             fit_mcd, fit_dvar, isotropy)       # noqa: E402

warnings.filterwarnings("ignore")

SEEDS = (42, 7, 123)


# --------------------------------------------------------------------- bases
def holdout_bases(hold_rows):
    """The three bases, all 480 rows projected, plus the fitting mask.

    `published` is read from RFS2009.xls. The two re-extracted bases come from
    `TrainPCA`, which standardises and takes eigenvectors on the fitting rows only
    and projects the held-out rows as out-of-sample points — the same leak-free
    construction the Chapter 6 pipeline uses for its train/test split, applied here
    to a two-month holdout instead of a trailing one.
    """
    a = dio.aligned_panel()
    tgt = a["targets"]
    n = len(a["dates"])
    fit_mask = np.ones(n, bool)
    fit_mask[hold_rows] = False

    F_pub = tgt[[f"f{i}" for i in range(1, NFAC + 1)]].to_numpy(dtype=float)
    assert not np.isnan(F_pub).any(), "published factor block has missing values"

    tp = TrainPCA(nfac=NFAC).fit(a["Y"][fit_mask])
    F_unit = tp.transform(a["Y"])
    # TrainPCA normalises each column to unit variance on the fitting rows; undo
    # that to recover the eigenvalue scaling. eigvals_ are eigenvalues of the
    # uncentred cross-product on the fitting rows, so divide by its degrees of
    # freedom to reach a variance.
    eig = tp.eigvals_ / (fit_mask.sum() - 1)
    F_scaled = F_unit * np.sqrt(eig)

    dates = pd.to_datetime(pd.Series(a["dates"])).to_numpy()
    return ({"published": F_pub, "fit_unit": F_unit, "fit_scaled": F_scaled},
            dates, fit_mask, eig)


# ------------------------------------------------------------------ scoring
def score_holdout(g, hold_rows, fit_mask, dates):
    """Extremeness of the held-out pair in the fitting sample's own frame."""
    g = np.asarray(g, dtype=float).reshape(-1)
    gf = g[fit_mask]

    med = float(np.median(gf))
    mad = float(np.median(np.abs(gf - med)))
    scale = MAD_SCALE * mad if mad > 1e-12 else float(gf.std(ddof=1))
    z_mad = (g - med) / scale

    mu, sd = float(gf.mean()), float(gf.std(ddof=1))
    z_sd = (g - mu) / (sd if sd > 1e-12 else 1.0)

    lo, hi = float(gf.min()), float(gf.max())

    def excess(i):
        """Distance beyond the nearer edge of the fitting range, in sd_fit units."""
        if g[i] > hi:
            return (g[i] - hi) / sd
        if g[i] < lo:
            return (g[i] - lo) / sd
        return 0.0

    # rank among all 480 months, as in the fitted-with-pair arm
    absz = np.abs(z_mad)
    order = np.argsort(-absz, kind="stable")
    rank = np.empty(len(g), dtype=int)
    rank[order] = np.arange(1, len(g) + 1)

    i, j = hold_rows
    pair_min = float(np.abs(z_mad[hold_rows]).min())
    fit_max = float(np.abs(z_mad[fit_mask]).max())
    return {
        "margin": pair_min / fit_max,
        "pair_isolated": bool(pair_min > fit_max),
        "outside_support": bool(excess(i) != 0.0 and excess(j) != 0.0),
        "z_sep_mad": float(z_mad[i]), "z_oct_mad": float(z_mad[j]),
        "z_sep_sd": float(z_sd[i]), "z_oct_sd": float(z_sd[j]),
        "excess_sep_sd": excess(i), "excess_oct_sd": excess(j),
        "rank_sep2001": int(rank[i]), "rank_oct2001": int(rank[j]),
        "worse_rank": int(max(rank[i], rank[j])),
        "fit_support": f"[{lo:.2f}, {hi:.2f}]",
        "dvar_fit": dvar(gf),
        "dvar_biloop_fit": dvar_biloop(gf),
        "top5_months": " ".join(
            pd.Timestamp(dates[t]).strftime("%Y-%m") for t in order[:5]),
    }


def frame_records(method, basis, strategy, seed, X, W, order, crit,
                  hold_rows, fit_mask, dates):
    recs = []
    for pos, j in enumerate(order[:K]):
        rec = dict(method=method, basis=basis, strategy=strategy, seed=seed,
                   direction=pos + 1, criterion=float(crit[j]))
        rec.update(score_holdout(X @ W[:, j], hold_rows, fit_mask, dates))
        rec["detail"] = " ".join(f"{v:+.3f}" for v in W[:, j])
        recs.append(rec)
    return recs


def mahalanobis_record(mcd, X, basis, seed, hold_rows, fit_mask, dates):
    """MCD distance fitted on the 478 months, evaluated on the pair."""
    d2 = mcd.mahalanobis(X)
    order = np.argsort(-d2, kind="stable")
    rank = np.empty(len(d2), dtype=int)
    rank[order] = np.arange(1, len(d2) + 1)
    cut = float(chi2.ppf(0.975, X.shape[1]))
    i, j = hold_rows
    pair_min = float(d2[hold_rows].min())
    fit_max = float(d2[fit_mask].max())
    return dict(
        method="MCD distance", basis=basis, strategy="-", seed=seed, direction=0,
        criterion=cut,
        margin=pair_min / fit_max, pair_isolated=bool(pair_min > fit_max),
        outside_support=bool(pair_min > fit_max),
        z_sep_mad=float(np.sqrt(d2[i])), z_oct_mad=float(np.sqrt(d2[j])),
        z_sep_sd=float("nan"), z_oct_sd=float("nan"),
        excess_sep_sd=float("nan"), excess_oct_sd=float("nan"),
        rank_sep2001=int(rank[i]), rank_oct2001=int(rank[j]),
        worse_rank=int(max(rank[i], rank[j])),
        fit_support=f"max d2 on the 478 = {fit_max:.1f}",
        dvar_fit=float("nan"), dvar_biloop_fit=float("nan"),
        top5_months=" ".join(pd.Timestamp(dates[t]).strftime("%Y-%m")
                             for t in order[:5]),
        detail=f"flagged at chi2_0.975: {int((d2 > cut).sum())} of {len(d2)} months",
    )


def pca_arms(Xf):
    """The classical-PCA arms available in a basis, given its fitting rows.

    "PCA" is the basis's own axis order: in `published` that is the published
    factors f1..f8, which is what Chapter 6 means by "the first PCA factor", and in
    the two re-extracted bases it is the leading principal components of the fitting
    months by construction.

    "PCA (refit)" solves the eigenproblem on the fitting rows' covariance inside the
    basis. It exists because holding the pair out is not a small perturbation of the
    published block: over all 480 months that block's covariance is isotropic to
    2.5e-05, and over the 478 it is 6.5e-01 — the two months carry a large share of
    its variance structure, so the coordinate axes are no longer its principal axes
    and assuming they are would understate what a variance criterion refitted
    without the pair actually does. The arm is skipped where the fitting covariance
    is isotropic, since there an eigensolver returns an arbitrary rotation; that is
    the case in `fit_unit` by construction (measured 4.7e-15).
    """
    arms = [("PCA", *pca_frame(Xf))]
    if isotropy(Xf) > ISOTROPY_TOL:
        ev, V = np.linalg.eigh(np.cov(Xf, rowvar=False))
        arms.append(("PCA (refit)", V, np.argsort(ev)[::-1], ev))
    return arms


def run_seed(X, basis, seed, hold_rows, fit_mask, dates):
    """Every criterion fitted on the 478 months of one basis at one seed."""
    Xf = X[fit_mask]
    recs = []

    for name, V, order, crit in pca_arms(Xf):
        recs += frame_records(name, basis, "-", seed, X, V, order, crit,
                              hold_rows, fit_mask, dates)

    V, order, ev, mcd = fit_mcd(Xf, seed)
    recs += frame_records("MCD-PCA", basis, "-", seed, X, V, order, ev,
                          hold_rows, fit_mask, dates)
    recs.append(mahalanobis_record(mcd, X, basis, seed, hold_rows, fit_mask, dates))

    for index_fun, label in DVAR_CONFIGS:
        for strategy in ("sequential", "joint"):
            W, order, crit = fit_dvar(Xf, index_fun, strategy, seed)
            recs += frame_records(label, basis, strategy, seed, X, W, order, crit,
                                  hold_rows, fit_mask, dates)
    return recs


# --------------------------------------------------------------------- figure
def _best_of_k(X, W, order, hold_rows, fit_mask):
    """The direction among the K reported that best exposes the pair, and its index.

    Every method is given the same allowance of K directions, so this is a
    multiplicity-matched comparison rather than a search that favours one method.
    The panel reports which position the direction occupied in the method's own
    ordering, since a criterion that exposes the pair only on the direction it ranks
    last is making a weaker statement than one that does so on its first.
    """
    best, best_pos, best_margin = None, None, -1.0
    for pos, j in enumerate(order[:K]):
        g = X @ W[:, j]
        gf = g[fit_mask]
        med = float(np.median(gf))
        mad = float(np.median(np.abs(gf - med)))
        z = (g - med) / (MAD_SCALE * mad if mad > 1e-12 else gf.std(ddof=1))
        m = float(np.abs(z[hold_rows]).min() / np.abs(z[fit_mask]).max())
        if m > best_margin:
            best, best_pos, best_margin = W[:, j], pos + 1, m
    return best, best_pos


# ----------------------------------------------------------------------- main
def main():
    a_dates = pd.to_datetime(pd.Series(dio.aligned_panel()["dates"])).to_numpy()
    hold_rows = target_rows(a_dates)
    B, dates, fit_mask, eig = holdout_bases(hold_rows)

    print(f"holding out {TARGET_MONTHS[0][:7]} and {TARGET_MONTHS[1][:7]} "
          f"(rows {hold_rows}); fitting on {int(fit_mask.sum())} of "
          f"{len(dates)} months")
    print(f"re-extracted eigenvalues: {np.round(eig, 4)}")
    for name, X in B.items():
        print(f"  isotropy of cov(X[fit]) in {name:11s}: "
              f"{isotropy(X[fit_mask]):.2e}")
    print()

    recs = []
    for basis in ("published", "fit_unit", "fit_scaled"):
        for seed in SEEDS:
            new = run_seed(B[basis], basis, seed, hold_rows, fit_mask, dates)
            recs += new
            for r in new:
                if r["direction"] in (0, 1):
                    print(f"  {basis:10s} seed {seed:4d}  {r['method']:18s} "
                          f"{r['strategy']:10s} dir {r['direction']}  "
                          f"z=({r['z_sep_mad']:+6.1f},{r['z_oct_mad']:+6.1f})  "
                          f"margin {r['margin']:6.2f}  "
                          f"outside={str(r['outside_support']):5s} "
                          f"isolated={r['pair_isolated']}")
            print(flush=True)

    df = pd.DataFrame(recs)
    write_csv(RESULTS / "holdout_recovery.csv", df,
                         seeds=list(SEEDS),
                         script="ch06_holdout_recovery.py",
                         K=K, budget=str(BUDGET),
                         holdout=", ".join(TARGET_MONTHS))

    # Median with [min, max] over the seeds; aggfunc named explicitly, because the
    # pandas default is the mean and on three replicates with one outlier that is
    # the wrong summary (CLAUDE.md Rule 5d).
    agg = (df.groupby(["basis", "method", "strategy", "direction"])
             .agg(margin_med=("margin", "median"),
                  margin_min=("margin", "min"),
                  margin_max=("margin", "max"),
                  worse_rank_med=("worse_rank", "median"),
                  worse_rank_min=("worse_rank", "min"),
                  worse_rank_max=("worse_rank", "max"),
                  isolated_seeds=("pair_isolated", "sum"),
                  outside_seeds=("outside_support", "sum"),
                  n_seeds=("seed", "nunique"))
             .reset_index())
    print("across seeds — margin > 1 means the pair is more extreme than any of "
          "the 478 months the criterion was fitted on:")
    print(agg.to_string(index=False))
    write_csv(RESULTS / "holdout_recovery_agg.csv", agg,
                         seeds=list(SEEDS),
                         script="ch06_holdout_recovery.py")

    # The pre-registered rule, evaluated in code so the verdict is in the log and
    # not assembled by hand afterwards.
    print("\npre-registered rule (leading direction, per basis):")
    lead = agg[agg.direction == 1]
    for basis in ("published", "fit_unit", "fit_scaled"):
        b = lead[lead.basis == basis]
        ref = b[b.method.str.startswith("PCA") | (b.method == "MCD-PCA")]
        ref_ok = bool((ref.margin_max < 1).all())
        for _, r in b[b.method.str.startswith("dVar")].iterrows():
            dvar_ok = bool(r.margin_med > 1 and r.margin_min > 1)
            print(f"  {basis:10s} {r.method:18s} {r.strategy:10s} "
                  f"dVar median {r.margin_med:.2f} [{r.margin_min:.2f}, "
                  f"{r.margin_max:.2f}]  dVar>1={dvar_ok}  "
                  f"PCA/MCD-PCA<1={ref_ok}  "
                  f"CREDITED={dvar_ok and ref_ok}")

    # The figure uses `fit_unit`: it is the leak-free basis, so it is the one whose
    # panels can be read as "the pair had no hand in this at all".
    print("\nSaved ch6_holdout_recovery.png")


if __name__ == "__main__":
    main()
