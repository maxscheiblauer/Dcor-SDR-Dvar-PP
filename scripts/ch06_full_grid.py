"""
run_grid.py — the systematic, leak-free comparison (PLAN_rebuild.md §4/§8.4).

Iterates the full grid
    Layer-1 finder x sub-variant x d(1..8) x Layer-2 spec(linear/poly) x CP(0/1)
    x maturity(2,3,4,5)
on the train-only PCA_ours factors, plus a benchmark column of the published
Ludvigson-Ng look-ahead factors (PCA_LN) so the cost of removing the leak is
visible.  Every direction is fit train-only and frozen; every score is evaluated
out-of-sample with train-frozen regression coefficients.

Outputs:
    results/grid_full.csv       one row per (maturity, method, d, spec, cp)
    results/grid_full_dm.csv    DM tests vs the two anchors PCA+poly / INPCA+lin
    reports/report_02_full_grid.md
    figures/grid_headline.png
"""

# Thesis:   Chapter 6, fig:rd-headline
# Writes:   results/grid_full.csv, results/grid_full_dm.csv
# Original: Real Data Experiment/src/run_grid.py on the thesis branch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import dcor as _dcor

import realdata.dataio as dio
from realdata.pca import TrainPCA
from realdata.regression import predict
from realdata.metrics import mspe, dm_test
import realdata.sdr_registry as reg

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

DIMS = tuple(range(1, 9))
SPECS = ("linear", "poly")

#: Single optimiser seed, deliberately, unlike the rest of the Chapter 6 chain.
#: This script is the leak-free rebuild of the whole grid: 2 blocks x every finder x
#: 8 dimensions x 2 specifications x CP on/off, and its outputs feed report_03 and
#: fig:rd-headline rather than any quoted number. Every figure the chapter quotes comes
#: from ch6_bestd_dm.py, peer_dm_published.py, fixed_d_bottleneck.py or
#: widen_bottleneck.py, and those four do run five optimiser seeds. Multi-seeding this
#: grid as well would multiply a 6-minute run by five for no claim that rests on it.
SEED = 0


def factor_blocks():
    """Return (dates, PCA_ours[n,8], PCA_LN[n,8], targets, train, test)."""
    a = dio.aligned_panel()
    dates, Y, tgt = a["dates"], a["Y"], a["targets"]
    train, test, _, _ = dio.split_masks(dates)
    pca = TrainPCA(nfac=8).fit(Y[train])
    F_ours = pca.transform(Y)
    F_ln = tgt[[f"f{i}" for i in range(1, 9)]].to_numpy()   # published look-ahead
    return dates, F_ours, F_ln, tgt, train, test


def _dcor_pair(z, ymask_tr, ymask_te, ytr, yte):
    try:
        return (float(_dcor.distance_correlation(z[ymask_tr], ytr)),
                float(_dcor.distance_correlation(z[ymask_te], yte)))
    except Exception:
        return (np.nan, np.nan)


def run_block(X, block_name, tgt, train, test, unsup, seed=SEED, want_dcor=True):
    """Evaluate the full finder x d x spec x cp grid on one factor block."""
    rows, err = [], {}
    tr_idx = np.where(train)[0]; te_idx = np.where(test)[0]
    for a in dio.MATS:
        y = tgt[f"yr{a}"].to_numpy()
        ytr = y[tr_idx + dio.H]; yte = y[te_idx + dio.H]
        bench = float(((yte - ytr.mean()) ** 2).mean())
        cp_all = tgt["CP"].to_numpy()
        sup = reg.build_supervised(X[train], ytr, seed=seed)
        finders = {**unsup, **sup}
        for mname, finder in finders.items():
            for d in DIMS:
                Z = finder.project(X, d)
                if Z is None or Z.shape[1] < d:
                    continue
                dctr, dcte = ((np.nan, np.nan) if not want_dcor
                              else _dcor_pair(Z[:, 0], train, test, ytr, yte))
                for spec in SPECS:
                    for cp in (False, True):
                        cpcol = cp_all if cp else None
                        pred = predict(Z, d, spec, train, test, ytr, cp=cpcol)
                        e = pred - yte
                        rows.append(dict(block=block_name, maturity=a,
                                         method=mname, d=d, spec=spec, cp=cp,
                                         mspe=mspe(e), r2=1.0 - mspe(e) / bench,
                                         bench=bench, dcor_tr=dctr, dcor_te=dcte))
                        err[(a, mname, d, spec, cp)] = e
    return pd.DataFrame(rows), err


def main(seed=SEED):
    t0 = time.time()
    dates, F_ours, F_ln, tgt, train, test = factor_blocks()
    print(f"n={len(dates)}  train={train.sum()}  test={test.sum()}  "
          f"PCA_ours{F_ours.shape}")

    print("building unsupervised finders (PCA/dVar/INPCA) on PCA_ours ...")
    unsup_ours = reg.build_unsupervised(F_ours[train], seed=seed)
    print(f"  done ({time.time()-t0:.0f}s).  running PCA_ours grid ...")
    res_ours, err_ours = run_block(F_ours, "PCA_ours", tgt, train, test,
                                   unsup_ours, seed=seed)
    print(f"  PCA_ours grid done ({time.time()-t0:.0f}s, {len(res_ours)} rows). "
          f"running PCA_LN benchmark grid ...")
    unsup_ln = reg.build_unsupervised(F_ln[train], seed=seed)
    res_ln, err_ln = run_block(F_ln, "PCA_LN", tgt, train, test, unsup_ln,
                               seed=seed, want_dcor=False)
    print(f"  PCA_LN grid done ({time.time()-t0:.0f}s).")

    res = pd.concat([res_ours, res_ln], ignore_index=True)
    write_csv(RESULTS / "grid_full.csv", res, seeds=(seed,),
                         script="ch06_full_grid.py")

    # ---- DM tests vs the two anchors (PCA_ours block) ----
    dm_rows = []
    anchors = {"PCA+poly(LN-winner)": ("PCA", "poly"),
               "INPCA(k3)+lin(G&S)": ("INPCA(k3)", "linear")}
    methods = sorted({k[1] for k in err_ours})
    for a in dio.MATS:
        for d in DIMS:
            for aname, (am, aspec) in anchors.items():
                for cp in (False, True):
                    key_a = (a, am, d, aspec, cp)
                    if key_a not in err_ours:
                        continue
                    e_a = err_ours[key_a]
                    for m in methods:
                        for spec in SPECS:
                            key_b = (a, m, d, spec, cp)
                            if key_b not in err_ours:
                                continue
                            e_b = err_ours[key_b]
                            dm, p = dm_test(e_a, e_b)
                            dm_rows.append(dict(maturity=a, d=d, cp=cp,
                                anchor=aname, method=f"{m}+{spec}", DM=dm, p=p,
                                mspe_anchor=mspe(e_a), mspe_method=mspe(e_b)))
    dm = pd.DataFrame(dm_rows)
    write_csv(RESULTS / "grid_full_dm.csv", dm, seeds=(seed,),
                         script="ch06_full_grid.py")

    print(f"\nDONE ({time.time()-t0:.0f}s).  wrote results/grid_full.csv "
          f"({len(res)} rows), results/grid_full_dm.csv ({len(dm)} rows), "
          f"reports/report_02_full_grid.md, figures/grid_headline.png")
    return res, dm


# ----------------------------------------------------------------- report
def _best_over_d(res, block, cp):
    sub = res[(res.block == block) & (res.cp == cp)]
    best = sub.groupby(["maturity", "method", "spec"])["r2"].max().reset_index()
    return best


if __name__ == "__main__":
    main()
