"""
widen_bottleneck.py — point 4: widen the PCA bottleneck from 8 to 12/16/20
train-only PCs and see how PCA+poly, INPCA+lin, and dCor-SDR behave when the
methods can see more of the panel variance (leak-free factors, no CP).

For each bottleneck B, all methods are fit on the SAME train-only B-dim PC block.
PCA / INPCA may use up to B directions; the SDR finders (dCor, PLS, SIR, SAVE)
are ALSO allowed up to B directions here (reg.KMAX raised to B per bottleneck),
so the head-to-head is fair — no method is silently capped below its competitors.
Reports best-over-d OOS Campbell-Thompson R^2, rx2 (GS focus) and the 4-maturity
mean.
"""

# Thesis:   Chapter 6, §6.7
# Writes:   results/widen_bottleneck.csv
# Original: Real Data Experiment/src/widen_bottleneck.py on the thesis branch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import sys, warnings
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

BS = (8, 12, 16, 20)

#: The one optimiser seed Chapter 6 reports. The panel is fixed, so a seed varies only
#: the starting points, and the chapter reports a single run (decision of 2026-08-19).
#: widening the list still turns this into a stability check.
SEEDS = (0,)


def build_finders(Xtr, ytr, seed=0):
    sup = reg.build_supervised(Xtr, ytr, seed=seed)   # PLS,SIR,SAVE,Dcor*
    f = {"PCA": reg.Identity(),
         "INPCA": reg.InpcaFinder(INPCA(k_nl=4, seed=seed).fit(Xtr)),
         "dCor(seqX)": sup["DcorSDR(seqX)"],
         "SIR": sup["SIR"], "PLS": sup["PLS"], "SAVE": sup["SAVE"]}
    return f


def best_r2(F, finder, spec, B, mat, tgt, train, test, tr, te):
    y = tgt[f"yr{mat}"].to_numpy()
    ytr = y[tr + dio.H]; yte = y[te + dio.H]
    bench = float(((yte - ytr.mean()) ** 2).mean())
    best = -np.inf
    for d in range(1, B + 1):
        Z = finder.project(F, d)
        if Z is None or Z.shape[1] < d:
            continue
        pred = predict(Z, d, spec, train, test, ytr)
        r2 = campbell_thompson_r2(mspe(pred - yte), bench)
        best = max(best, r2)
    return best


def main():
    a = dio.aligned_panel()
    tgt = a["targets"]; Y = a["Y"]
    train, test, tr, te = dio.split_masks(a["dates"])

    rows = []
    for seed in SEEDS:
        for B in BS:
            # fair cap: SDR finders may use up to B directions, like PCA/INPCA
            reg.KMAX = B
            F = TrainPCA(nfac=B).fit(Y[train]).transform(Y)
            for mat in dio.MATS:
                ytr = tgt[f"yr{mat}"].to_numpy()[tr + dio.H]
                fin = build_finders(F[train], ytr, seed=seed)
                entry = {"seed": seed, "B": B, "mat": mat}
                entry["PCA+poly"] = best_r2(F, fin["PCA"], "poly", B, mat, tgt, train, test, tr, te)
                entry["INPCA+lin"] = best_r2(F, fin["INPCA"], "linear", B, mat, tgt, train, test, tr, te)
                entry["dCor+lin"] = best_r2(F, fin["dCor(seqX)"], "linear", B, mat, tgt, train, test, tr, te)
                entry["dCor+poly"] = best_r2(F, fin["dCor(seqX)"], "poly", B, mat, tgt, train, test, tr, te)
                entry["SIR+poly"] = best_r2(F, fin["SIR"], "poly", B, mat, tgt, train, test, tr, te)
                entry["PLS+lin"] = best_r2(F, fin["PLS"], "linear", B, mat, tgt, train, test, tr, te)
                rows.append(entry)
        print(f"  seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    cols = ["PCA+poly", "INPCA+lin", "dCor+lin", "dCor+poly", "SIR+poly", "PLS+lin"]

    print("=" * 78)
    print("Widened bottleneck — best-over-d OOS R^2, leak-free, no CP,  rx2 (GS focus)")
    print(f"median over {len(SEEDS)} optimiser seeds")
    print("=" * 78)
    print(f"  {'B':>3s} " + "".join(f"{c:>11s}" for c in cols))
    for B in BS:
        r = df[(df.B == B) & (df.mat == 2)][cols].median()
        print(f"  {B:>3d} " + "".join(f"{r[c]:>+11.3f}" for c in cols))

    print("\n" + "=" * 78)
    print("Widened bottleneck — 4-maturity MEAN best-over-d OOS R^2")
    print(f"(mean over maturities of the per-seed median)")
    print("=" * 78)
    print(f"  {'B':>3s} " + "".join(f"{c:>11s}" for c in cols))
    for B in BS:
        r = (df[df.B == B].groupby("mat")[cols].median()).mean()
        print(f"  {B:>3d} " + "".join(f"{r[c]:>+11.3f}" for c in cols))
    out = RESULTS / "widen_bottleneck.csv"
    write_csv(out, df, seeds=SEEDS,
                         script="ch06_widen_bottleneck.py")
    print("\nwrote results/widen_bottleneck.csv")


if __name__ == "__main__":
    main()
