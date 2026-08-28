"""
motivating_example.py — one exhibit: a robust covariance criterion puts September
and October 2001 outside everything it was fitted on, and classical PCA, given the
same two components, does not.

This is not a study. `Real Data Experiment/src/outlier_recovery.py` and
`holdout_recovery.py` are the broad survey — every criterion, three bases, three
directions each, both with the two months inside the fit and with them removed.
This script extracts the single clearest contrast those two files contain and
presents it on its own, because a comparison that fits in one table and one picture
is what motivates a reader to want the rest.

The contrast
------------
PCA and MCD-PCA are the same method on the same data at the same dimension. They
differ in one thing: whether the covariance whose eigenvectors they take is the
ordinary sample covariance or the minimum-covariance-determinant estimate, which is
computed on the subsample of months that minimises the determinant and therefore
does not let a few unusual months set the axes. Both are given two components. On
the second robust component the two months of the September 2001 shock sit further
from the centre than any of the 478 months the criterion was fitted on. On either
classical component they sit inside the crowd, at ranks 66 and 219 of 480.

Both criteria are fitted on the 478 months that are not September or October 2001,
inside a subspace re-extracted from those same 478 months. The two months therefore
had no hand in the coordinate system, no hand in the axes, and no hand in the
reference range they are compared against. They are pure out-of-sample points.

What this exhibit is and is not
-------------------------------
It is one occasion, on one macroeconomic panel. It happens to hold at all three
seeds tried, which is reported as a bonus rather than claimed as a property: seeds
move the MCD subsampling and the dVar restarts, not the classical PCA arm, which is
deterministic. It is not evidence that robust PCA detects outliers in general, and
it is not evidence that classical PCA cannot see these two months at all — given
all eight components it reaches them on the eighth, because that is where they
live. The point is narrower and checkable: an ordering by variance puts that
component last, and an ordering by a robust covariance does not.

Three further criteria are carried as context rows, since a reader will ask what
the rest of the thesis's machinery does here: plain dVar-PP, dVar-PP with the
biloop transform, and the MCD robust Mahalanobis distance. Their outcomes are
reported whatever they are. Plain dVar-PP misses at two components and the MCD
distance misses in this basis, and both facts are in the table.

Distance variance is a dispersion / shape index throughout, never a measure of
non-Gaussianity.

Run:  python scripts/ch06_case_motivating_example.py
Writes: results/motivating_example.csv, figures/motivating_example.png
"""

# Thesis:   Chapter 6, the case-study exhibit
# Writes:   results/motivating_example.csv
# Original: Outlier Case Study/motivating_example.py on the thesis branch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent                                  # Python/

import realdata.dataio as dio                                           # noqa: E402
# The survey owns the machinery: the basis construction, the scorer and the four
# fitters. Importing them keeps this exhibit and the survey numerically identical —
# a separate implementation here could drift, and the cross-check in the README
# would then be checking two copies of the same mistake.
from realdata import holdout as hr                                  # noqa: E402
from realdata import recovery as orec                                # noqa: E402

warnings.filterwarnings("ignore")

#: Two components for every criterion, so no criterion is given a larger allowance
#: than another. The survey uses three, and the third is where plain dVar-PP finds
#: the pair; at two it does not, which is why the allowance is stated rather than
#: assumed.
K = 2

#: The basis re-extracted from the fitting months. `published` is excluded here on
#: purpose: its extraction saw 2001, so the pair helped build the space, and an
#: exhibit whose point is that the two months had no influence cannot use it.
BASIS = "fit_unit"

SEEDS = (42, 7, 123)

#: The seed whose numbers the figure and the write-up quote. The others are run so
#: the isolation rate is visible in the table.
HEADLINE_SEED = 42

FIGURES = _HERE / "figures"

#: Row order in the printed table: the exhibit first, context below.
ORDER = ["MCD-PCA", "PCA", "PCA (refit)", "dVar-PP", "dVar-PP (biloop)",
         "MCD distance"]


def fit_all(X, basis, seed, hold_rows, fit_mask, dates):
    """Every criterion at K components, fitted on the 478 months.

    A trimmed copy of `holdout_recovery.run_seed`: same helpers, same scorer, same
    metric, restricted to K components by the module-level override below.
    """
    return hr.run_seed(X, basis, seed, hold_rows, fit_mask, dates)


def main():
    # K = 2 everywhere. Both survey modules need setting: `frame_records` slices
    # `order[:K]` against holdout_recovery's K, while `fit_dvar` passes K straight to
    # `pp_dvar(k=...)` and reads outlier_recovery's. Setting only one would report two
    # components of a three-component dVar-PP frame, which is a different frame.
    hr.K = K
    orec.K = K

    dates_all = pd.to_datetime(pd.Series(dio.aligned_panel()["dates"])).to_numpy()
    hold_rows = hr.target_rows(dates_all)
    B, dates, fit_mask, eig = hr.holdout_bases(hold_rows)
    X = B[BASIS]

    print(f"basis {BASIS}: leading 8 components re-extracted from the "
          f"{int(fit_mask.sum())} months that are not "
          f"{hr.TARGET_MONTHS[0][:7]} or {hr.TARGET_MONTHS[1][:7]}")
    print(f"components allowed per criterion: {K}")
    print(f"isotropy of the fitting covariance: {hr.isotropy(X[fit_mask]):.2e} — "
          f"every unit direction carries the same projection variance, so nothing "
          f"in the comparison is a scale effect\n")

    recs = []
    for seed in SEEDS:
        recs += fit_all(X, BASIS, seed, hold_rows, fit_mask, dates)
    df = pd.DataFrame(recs)

    # The exhibit: best of the K components per criterion, at the headline seed.
    head = df[df.seed == HEADLINE_SEED]
    best = (head.sort_values("margin", ascending=False)
                .groupby("method", as_index=False).first()
                .set_index("method"))
    best = best.reindex([m for m in ORDER if m in best.index])
    show = ["method", "direction", "margin", "pair_isolated",
            "rank_sep2001", "rank_oct2001", "z_sep_sd", "z_oct_sd",
            "excess_sep_sd", "excess_oct_sd"]
    print(f"seed {HEADLINE_SEED}, best of the {K} components per criterion "
          f"(margin > 1 means both months are further from the centre than every "
          f"one of the 478):")
    print(best.reset_index()[show].to_string(index=False))

    print(f"\nevery component, every seed:")
    print(df.sort_values(["method", "direction", "seed"])[
        ["method", "strategy", "direction", "seed", "margin", "pair_isolated",
         "rank_sep2001", "rank_oct2001"]].to_string(index=False))

    agg = (df.groupby(["method", "strategy", "direction"])
             .agg(margin_med=("margin", "median"),
                  margin_min=("margin", "min"),
                  margin_max=("margin", "max"),
                  isolated_seeds=("pair_isolated", "sum"),
                  n_seeds=("seed", "nunique"))
             .reset_index())
    print("\nacross seeds:")
    print(agg.to_string(index=False))

    # The exhibit's two claims, checked in code so the verdict is in the log rather
    # than assembled by hand from the table above.
    pca_rows = df[df.method == "PCA"]
    mcd_rows = df[df.method == "MCD-PCA"]
    pca_fails = bool((pca_rows.margin < 1).all())
    mcd_isolates = int(mcd_rows.groupby("seed").pair_isolated.any().sum())
    print(f"\nclassical PCA fails on both components at every seed: {pca_fails} "
          f"(largest margin {pca_rows.margin.max():.3f})")
    print(f"MCD-PCA isolates the pair at {mcd_isolates} of "
          f"{df.seed.nunique()} seeds "
          f"(largest margin {mcd_rows.margin.max():.3f})")
    print(f"EXHIBIT HOLDS: {pca_fails and mcd_isolates >= 1}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "motivating_example.csv", df,
                         seeds=list(SEEDS),
                         script="ch06_case_motivating_example.py",
                         K=K, basis=BASIS, headline_seed=HEADLINE_SEED,
                         holdout=", ".join(hr.TARGET_MONTHS))

    rows = {m: best.loc[m] for m in ("PCA", "MCD-PCA")}


if __name__ == "__main__":
    main()
