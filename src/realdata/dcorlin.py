"""
dcorlin_experiments.py — two targeted experiments requested for the thesis.

EXP 1 (leak-free PCA_ours): rigorous DM test dCor-SDR(seqX)+poly vs +lin,
       every maturity x d. Focus: does the cube add anything over the linear
       second stage? Sign convention dm_test(e_poly, e_lin) > 0 => poly worse
       => lin better.

EXP 2 (leaked PCA_LN, the published Ludvigson-Ng full-sample factors — the same
       block INPCA used): dCor-SDR(seqX)+lin vs INPCA+lin and vs PCA+poly, on the
       paper's OWN factors and the paper's OWN scoring (best-over-d Campbell-
       Thompson OOS R^2, and the "match/beat PCA+poly" delta), plus the DM test
       the paper did not run.

Reuses the exact rebuild machinery (dataio, pca, regression, metrics, the same
finder adapters) so numbers are identical to grid_full.csv.
"""
from __future__ import annotations

from csvout import RESULTS, write_csv
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

import realdata.dataio as dio
from realdata.pca import TrainPCA
from realdata.regression import predict
from realdata.metrics import mspe, dm_test, campbell_thompson_r2
import realdata.sdr_registry as reg
from realdata.inpca import INPCA

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

#: Single optimiser seed, deliberately. These two experiments are diagnostics of the
#: second stage (polynomial against linear, and the leaked-block comparison); the
#: chapter quotes no number from their CSVs. The five-seed rule is carried by
#: ch6_bestd_dm.py, peer_dm_published.py, fixed_d_bottleneck.py and
#: widen_bottleneck.py, which produce everything the chapter states.
SEED = 0
DIMS = range(1, 9)


def blocks():
    a = dio.aligned_panel()
    dates, Y, tgt = a["dates"], a["Y"], a["targets"]
    train, test, tr, te = dio.split_masks(dates)
    F_ours = TrainPCA(nfac=8).fit(Y[train]).transform(Y)
    F_ln = tgt[[f"f{i}" for i in range(1, 9)]].to_numpy()  # published look-ahead
    return dates, F_ours, F_ln, tgt, train, test, tr, te


def finders_for_block(X, ytr, train, seed=SEED):
    """Only the finders these experiments need (skip the slow dVar grid)."""
    sup = reg.build_supervised(X[train], ytr, seed=seed)  # gives DcorSDR(seqX), PLS, SIR, SAVE
    f = {"DcorSDR(seqX)": sup["DcorSDR(seqX)"],
         "PCA": reg.Identity()}
    for k in (2, 3, 4):
        f[f"INPCA(k{k})"] = reg.InpcaFinder(INPCA(k_nl=k, seed=seed).fit(X[train]))
    return f


def errors(finders, X, tgt, train, test, tr, te, specs=("linear", "poly")):
    """err[(mat, method, d, spec)] = per-test-month forecast error vector; also r2."""
    err, r2 = {}, {}
    cp = None  # no CP throughout (matches the INPCA headline setting)
    for mat in dio.MATS:
        y = tgt[f"yr{mat}"].to_numpy()
        ytr = y[tr + dio.H]; yte = y[te + dio.H]
        bench = float(((yte - ytr.mean()) ** 2).mean())
        for mname, finder in finders.items():
            for d in DIMS:
                Z = finder.project(X, d)
                if Z is None or Z.shape[1] < d:
                    continue
                for spec in specs:
                    e = predict(Z, d, spec, train, test, ytr, cp=cp) - yte
                    err[(mat, mname, d, spec)] = e
                    r2[(mat, mname, d, spec)] = campbell_thompson_r2(mspe(e), bench)
    return err, r2


def best_over_d(r2, method, spec):
    return {mat: max(r2[(mat, method, d, spec)] for d in DIMS
                     if (mat, method, d, spec) in r2) for mat in dio.MATS}


def argbest_over_d(r2, method, spec):
    out = {}
    for mat in dio.MATS:
        cand = [(d, r2[(mat, method, d, spec)]) for d in DIMS
                if (mat, method, d, spec) in r2]
        out[mat] = max(cand, key=lambda t: t[1])
    return out


# ======================================================================= EXP 1
def exp1(err_ours, r2_ours):
    print("\n" + "=" * 78)
    print("EXP 1 — dCor-SDR(seqX): poly vs lin, leak-free PCA_ours, no CP")
    print("  DM = dm_test(e_poly, e_lin);  DM>0 => poly WORSE (lin better);")
    print("  DM<0 => poly better;  p<0.10 => the two stages differ significantly.")
    print("=" * 78)
    rows = []
    m = "DcorSDR(seqX)"
    for mat in dio.MATS:
        for d in DIMS:
            kp = (mat, m, d, "poly"); kl = (mat, m, d, "linear")
            if kp not in err_ours or kl not in err_ours:
                continue
            dm, p = dm_test(err_ours[kp], err_ours[kl])
            rows.append(dict(maturity=mat, d=d,
                             r2_poly=r2_ours[kp], r2_lin=r2_ours[kl],
                             mspe_poly=mspe(err_ours[kp]), mspe_lin=mspe(err_ours[kl]),
                             DM=dm, p=p))
    df = pd.DataFrame(rows)
    write_csv(RESULTS / "exp1_dcor_poly_vs_lin.csv", df, seeds=(SEED,),
                         script="ch06_dcor_linear_vs_poly.py")

    # best-over-d summary
    b_poly = best_over_d(r2_ours, m, "poly")
    b_lin = best_over_d(r2_ours, m, "lin".replace("lin", "linear"))
    print("\nbest-over-d OOS R^2:")
    print(f"  {'maturity':9s} {'dCor+lin':>9s} {'dCor+poly':>10s} {'gain(poly-lin)':>15s}")
    for mat in dio.MATS:
        print(f"  rx{mat:<7d} {b_lin[mat]:9.3f} {b_poly[mat]:10.3f} "
              f"{b_poly[mat]-b_lin[mat]:15.3f}")

    print("\nper-d DM (operating range d<=3 shown; full table in CSV):")
    print(f"  {'mat':>3s} {'d':>2s} {'R2_lin':>7s} {'R2_poly':>7s} "
          f"{'DM':>7s} {'p':>6s}  {'verdict':s}")
    for _, r in df[df.d <= 3].iterrows():
        v = ("lin better" if r.DM > 0 else "poly better") if r.p < 0.10 else "TIE"
        print(f"  {int(r.maturity):>3d} {int(r.d):>2d} {r.r2_lin:>7.3f} "
              f"{r.r2_poly:>7.3f} {r.DM:>7.2f} {r.p:>6.3f}  {v}")
    nsig = ((df.p < 0.10)).sum()
    print(f"\n  cells where poly and lin differ at p<0.10: {nsig}/{len(df)} "
          f"(all d);  d<=3: {((df.p<0.10)&(df.d<=3)).sum()}/{(df.d<=3).sum()}")
    print(f"  min p over all d: {df.p.min():.3f} (at "
          f"{df.loc[df.p.idxmin(), ['maturity','d']].to_dict()})")
    print(f"  min p over d<=3 : {df[df.d<=3].p.min():.3f}")
    return df


# ======================================================================= EXP 2
def exp2(err_ln, r2_ln):
    print("\n" + "=" * 78)
    print("EXP 2 — leaked PCA_LN (published LN full-sample factors, INPCA's own block)")
    print("  Can dCor-SDR(seqX)+lin beat INPCA at its own game, on its own factors?")
    print("=" * 78)

    dcor_lin = best_over_d(r2_ln, "DcorSDR(seqX)", "linear")
    dcor_lin_arg = argbest_over_d(r2_ln, "DcorSDR(seqX)", "linear")
    pca_poly = best_over_d(r2_ln, "PCA", "poly")
    inpca = {k: best_over_d(r2_ln, f"INPCA(k{k})", "linear") for k in (2, 3, 4)}
    # INPCA best across k (the paper reports its best entropy-rank block)
    inpca_best = {mat: max(inpca[k][mat] for k in (2, 3, 4)) for mat in dio.MATS}

    print("\n--- (A) The INPCA paper's OWN scoring: best-over-d Campbell-Thompson")
    print("    OOS R^2 levels + the match/beat-PCA+poly delta (no significance test).")
    print(f"  {'maturity':9s} {'PCA+poly':>9s} {'INPCA+lin':>10s} {'dCor+lin':>9s} "
          f"{'dCor-INPCA':>11s} {'dCor-PCA+poly':>13s}")
    for mat in dio.MATS:
        print(f"  rx{mat:<7d} {pca_poly[mat]:9.3f} {inpca_best[mat]:10.3f} "
              f"{dcor_lin[mat]:9.3f} {dcor_lin[mat]-inpca_best[mat]:11.3f} "
              f"{dcor_lin[mat]-pca_poly[mat]:13.3f}")
    print("  (INPCA+lin = best over k in {2,3,4}; per-k below)")
    for k in (2, 3, 4):
        print(f"    INPCA(k{k})+lin: " +
              "  ".join(f"rx{mat}={inpca[k][mat]:.3f}" for mat in dio.MATS))
    print("  dCor+lin best-d (d*, R^2): " +
          "  ".join(f"rx{mat}=(d{dcor_lin_arg[mat][0]},{dcor_lin_arg[mat][1]:.3f})"
                    for mat in dio.MATS))

    # --- (B) DM tests the paper did NOT run, on the leaked block
    print("\n--- (B) DM test on the SAME leaked factors (the check G&S did not do).")
    print("    DM = dm_test(e_INPCA, e_dcorlin) > 0 => INPCA worse (dCor+lin better).")
    rows = []
    for mat in dio.MATS:
        for d in DIMS:
            kd = (mat, "DcorSDR(seqX)", d, "linear")
            if kd not in err_ln:
                continue
            e_d = err_ln[kd]
            for anchor, am, aspec in [("INPCA(k3)+lin", "INPCA(k3)", "linear"),
                                      ("INPCA(k4)+lin", "INPCA(k4)", "linear"),
                                      ("PCA+poly", "PCA", "poly")]:
                ka = (mat, am, d, aspec)
                if ka not in err_ln:
                    continue
                dm, p = dm_test(err_ln[ka], e_d)
                rows.append(dict(maturity=mat, d=d, anchor=anchor,
                                 r2_anchor=r2_ln[ka], r2_dcorlin=r2_ln[kd],
                                 mspe_anchor=mspe(err_ln[ka]), mspe_dcorlin=mspe(e_d),
                                 DM=dm, p=p))
    dm_df = pd.DataFrame(rows)
    write_csv(RESULTS / "exp2_leaked_dcorlin_dm.csv", dm_df, seeds=(SEED,),
                         script="ch06_dcor_linear_vs_poly.py")

    for anchor in ["INPCA(k3)+lin", "INPCA(k4)+lin", "PCA+poly"]:
        s = dm_df[(dm_df.anchor == anchor) & (dm_df.d <= 3)]
        better = ((s.DM > 0) & (s.p < 0.10)).sum()
        worse = ((s.DM < 0) & (s.p < 0.10)).sum()
        print(f"\n  dCor+lin vs {anchor}, d<=3 ({len(s)} cells): "
              f"dCor sig-better {better}, sig-worse {worse}, tie {len(s)-better-worse}; "
              f"min p={s.p.min():.3f}")
        # show d=1 explicitly (the low-dim operating point)
        for _, r in s[s.d == 1].iterrows():
            tag = ("dCor better" if r.DM > 0 else "anchor better")
            print(f"    rx{int(r.maturity)} d=1: R2 anchor={r.r2_anchor:.3f} "
                  f"dCor+lin={r.r2_dcorlin:.3f}  MSPE {r.mspe_anchor:.2f} vs "
                  f"{r.mspe_dcorlin:.2f}  DM={r.DM:.2f} p={r.p:.3f} ({tag})")
    return dm_df


def main():
    dates, F_ours, F_ln, tgt, train, test, tr, te = blocks()
    print(f"n={len(dates)} train={train.sum()} test={test.sum()}")

    # errors need supervised finders per maturity; build per block once via a
    # representative ytr is wrong (dCor is per-maturity) -> rebuild inside errors.
    # So we construct finders per maturity here.
    def block_errors(X):
        err, r2 = {}, {}
        cp = None
        for mat in dio.MATS:
            y = tgt[f"yr{mat}"].to_numpy()
            ytr = y[tr + dio.H]; yte = y[te + dio.H]
            bench = float(((yte - ytr.mean()) ** 2).mean())
            f = finders_for_block(X, ytr, train)
            for mname, finder in f.items():
                for d in DIMS:
                    Z = finder.project(X, d)
                    if Z is None or Z.shape[1] < d:
                        continue
                    for spec in ("linear", "poly"):
                        e = predict(Z, d, spec, train, test, ytr, cp=cp) - yte
                        err[(mat, mname, d, spec)] = e
                        r2[(mat, mname, d, spec)] = campbell_thompson_r2(mspe(e), bench)
        return err, r2

    print("building + scoring leak-free PCA_ours block ...")
    err_ours, r2_ours = block_errors(F_ours)
    exp1(err_ours, r2_ours)

    print("\nbuilding + scoring leaked PCA_LN block ...")
    err_ln, r2_ln = block_errors(F_ln)
    exp2(err_ln, r2_ln)

    print("\nwrote results/exp1_dcor_poly_vs_lin.csv, "
          "results/exp2_leaked_dcorlin_dm.csv")


if __name__ == "__main__":
    main()
