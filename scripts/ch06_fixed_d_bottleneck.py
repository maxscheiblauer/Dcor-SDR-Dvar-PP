"""
fixed_d_bottleneck.py — point 4, fixed-d variant (author-specified design).

Question: hold each method's second-stage dimension FIXED at its own operating
point, then widen the input PC block B in {8,12,16,20} (train-only, leak-free,
no CP). Does exposing more of the panel's total variance change anything when the
methods are NOT allowed to spend extra directions?

Fixed operating points (from results/grid_full.csv, block=PCA_ours, leak-free):
  * dCor-SDR(seqX)+lin at d=2  — its best-over-d optimum every maturity at B=8.
  * dCor-SDR(seqX)+lin at d=4  — pays off only once the block is widened.
  * INPCA+lin at d=8 = the PUBLISHED config: 4 reshaped + 4 linear (k_nl=4).
  * PCA+poly and PCA+lin at d=8 — the LN benchmark reference.

Because PCA is nested, PCA(d=8) and INPCA(4 reshaped + 4 linear, d=8) look ONLY
at the first 8 PCs and are therefore FLAT across B by construction — honest
controls. dCor(d=2) searches the full B-dim space for its 2 directions, so it is
the only method that can move as B grows. Expectation: small drift (each extra PC
carries little variance), but the direction of the drift is the point of interest.
"""

# Thesis:   Chapter 6, fig:rd-bottleneck
# Writes:   results/fixed_d_bottleneck.csv
# Original: Real Data Experiment/src/fixed_d_bottleneck.py on the thesis branch.
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

import realdata.dataio as dio
from realdata.pca import TrainPCA
from realdata.regression import predict
from realdata.metrics import mspe, campbell_thompson_r2
import realdata.sdr_registry as reg
from realdata.inpca import INPCA

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent

BS = (8, 12, 16, 20)

#: The one optimiser seed Chapter 6 reports. The panel is fixed, so a seed varies only
#: the starting points, and the chapter reports a single run (decision of 2026-08-19).
#: widening the list still turns this into a stability check.
SEEDS = (0,)
D_DCOR = 2      # dCor(seqX)+lin operating point
D_INPCA = 8     # published INPCA: 4 reshaped + 4 linear
D_PCA = 8       # LN benchmark reference uses the 8-PC block


def r2_at_d(F, finder, d, spec, mat, tgt, train, test, tr, te):
    y = tgt[f"yr{mat}"].to_numpy()
    ytr = y[tr + dio.H]; yte = y[te + dio.H]
    bench = float(((yte - ytr.mean()) ** 2).mean())
    Z = finder.project(F, d)
    if Z is None or Z.shape[1] < d:
        return np.nan
    pred = predict(Z, d, spec, train, test, ytr)
    return campbell_thompson_r2(mspe(pred - yte), bench)


def main():
    a = dio.aligned_panel()
    tgt = a["targets"]; Y = a["Y"]
    train, test, tr, te = dio.split_masks(a["dates"])

    rows = []
    for seed in SEEDS:
        for B in BS:
            # build dCor to 4 dirs once; d=2 = first 2 cols (sequential deflation)
            reg.KMAX = 4
            F = TrainPCA(nfac=B).fit(Y[train]).transform(Y)
            for mat in dio.MATS:
                ytr = tgt[f"yr{mat}"].to_numpy()[tr + dio.H]
                sup = reg.build_supervised(F[train], ytr, seed=seed)
                dcor = sup["DcorSDR(seqX)"]
                inpca = reg.InpcaFinder(INPCA(k_nl=4, seed=seed).fit(F[train]))
                pca = reg.Identity()
                rows.append({
                    "seed": seed, "B": B, "mat": mat,
                    "PCA+poly(d8)": r2_at_d(F, pca, D_PCA, "poly", mat, tgt, train, test, tr, te),
                    "PCA+lin(d8)":  r2_at_d(F, pca, D_PCA, "linear", mat, tgt, train, test, tr, te),
                    "INPCA+lin(4+4)": r2_at_d(F, inpca, D_INPCA, "linear", mat, tgt, train, test, tr, te),
                    "dCor+lin(d2)": r2_at_d(F, dcor, 2, "linear", mat, tgt, train, test, tr, te),
                    "dCor+lin(d4)": r2_at_d(F, dcor, 4, "linear", mat, tgt, train, test, tr, te),
                })
        print(f"  seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    cols = ["PCA+poly(d8)", "PCA+lin(d8)", "INPCA+lin(4+4)", "dCor+lin(d2)", "dCor+lin(d4)"]

    def block(title, selector):
        print("=" * 90)
        print(f"Fixed-d bottleneck — OOS R^2, leak-free, no CP  |  {title}")
        print(f"median over {len(SEEDS)} optimiser seeds")
        print("=" * 90)
        print(f"  {'B':>3s} " + "".join(f"{c:>16s}" for c in cols))
        for B in BS:
            r = selector(B)
            print(f"  {B:>3d} " + "".join(f"{r[c]:>+16.3f}" for c in cols))
        print()

    for mat in dio.MATS:
        block(f"rx{mat}",
              lambda B, m=mat: df[(df.B == B) & (df.mat == m)][cols].median())
    block("4-maturity MEAN", lambda B: df[df.B == B][cols].median())

    # Does dCor+lin(d2) keep the lower MSPE at every widening and maturity, at every
    # seed? The chapter states this as a clean sweep, so the count is printed rather
    # than left to be read off a table of medians.
    wins = 0
    total = 0
    for (seed, B, mat), g in df.groupby(["seed", "B", "mat"]):
        r = g.iloc[0]
        total += 1
        if r["dCor+lin(d2)"] > max(r["PCA+lin(d8)"], r["INPCA+lin(4+4)"]):
            wins += 1
    print(f"dCor+lin(d2) beats PCA+lin(d8) and INPCA+lin(4+4) on R^2 in "
          f"{wins} of {total} (seed, widening, maturity) combinations")

    out = RESULTS / "fixed_d_bottleneck.csv"
    write_csv(out, df, seeds=SEEDS,
                         script="ch06_fixed_d_bottleneck.py")
    print(f"\nwrote {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
