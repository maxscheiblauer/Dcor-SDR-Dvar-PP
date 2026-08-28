"""
ch6_bestd_dm.py — best-over-d comparisons for the revised Chapter 6.

The revised chapter compares methods ONLY at their own best-over-d operating
point (per the author's instruction: for a nested/rotational reduction the
number of directions is an internal tuning choice, so only the best matters).
This script produces, on the PUBLISHED (Ludvigson-Ng) factor block, no CP:

  * best-over-d Campbell-Thompson OOS R^2 and the d* at which it is attained,
    for every method the chapter reports;
  * the Diebold-Mariano (Newey-West 18) test between each pair of interest,
    each method taken at its OWN best d;
  * the 2001 leverage-point anatomy for the dVar-PP direction (scores, z, the
    factor loadings of the selected direction).

Outputs: results/ch6_bestd_dm.csv, results/ch6_bestd_levels.csv, stdout report.

The panel is fixed, so a seed here varies only the optimiser's starting points, not
the data. There is one dataset, one split and one set of factors, so there is no
sampling distribution to average over: this chapter is a proof of concept on a single
benchmark rather than a simulation study of the kind Chapters 4 and 5 run, and it
reports one run. SEEDS is therefore (0,) by decision of 2026-08-19, and seed 0 is what
every number printed in Chapter 6 comes from. Widening the list turns this into a
stability check, which the aggregations below survive; the chapter quotes the single
seed.
"""

# Thesis:   Chapter 6, tab:rd-bestd
# Writes:   results/ch6_bestd_levels.csv, results/ch6_bestd_dm.csv, results/ch6_dvar_anatomy.csv
# Original: Real Data Experiment/src/ch6_bestd_dm.py on the thesis branch.
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
from scipy.stats import kurtosis

import realdata.dataio as dio
from realdata.regression import predict
from realdata.metrics import mspe, dm_test, campbell_thompson_r2
import realdata.sdr_registry as reg
from realdata.inpca import INPCA

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

SEED = 0                 # the reference seed, used for the 2001 anatomy below

#: The one optimiser seed Chapter 6 reports. See the module docstring.
SEEDS = (0,)
DIMS = list(range(1, 9))

# GS hybrid: reshape the leading min(kappa, 4) factors; registry has k in {2,3,4}
def inpca_k(d):
    return min(max(d, 2), 4)


def main():
    a = dio.aligned_panel()
    dates = pd.to_datetime(pd.Series(a["dates"]))
    tgt = a["targets"]
    train, test, tr, te = dio.split_masks(a["dates"])
    F = tgt[[f"f{i}" for i in range(1, 9)]].to_numpy()   # published block

    err, r2 = {}, {}
    for seed in SEEDS:
        # The unsupervised finders depend only on the training block and the seed, not
        # on the maturity, so they are fitted once per seed rather than once per
        # maturity. Identical inputs, identical finders -- four times less work, which
        # is what makes five seeds affordable here.
        fin_unsup = reg.build_unsupervised(F[train], seed=seed)
        for mat in dio.MATS:
            y = tgt[f"yr{mat}"].to_numpy()
            ytr = y[tr + dio.H]; yte = y[te + dio.H]
            bench = float(((yte - ytr.mean()) ** 2).mean())
            finders = dict(reg.build_supervised(F[train], ytr, seed=seed))
            finders.update(fin_unsup)
            for mname, finder in finders.items():
                for d in DIMS:
                    Z = finder.project(F, d)
                    if Z is None or Z.shape[1] < d:
                        continue
                    for spec in ("linear", "poly"):
                        e = predict(Z, d, spec, train, test, ytr, cp=None) - yte
                        err[(seed, mat, mname, d, spec)] = e
                        r2[(seed, mat, mname, d, spec)] = campbell_thompson_r2(
                            mspe(e), bench)
            # INPCA hybrid curve: k varies with d
            for d in DIMS:
                src = (seed, mat, f"INPCA(k{inpca_k(d)})", d, "linear")
                if src in err:
                    err[(seed, mat, "INPCA-hybrid", d, "linear")] = err[src]
                    r2[(seed, mat, "INPCA-hybrid", d, "linear")] = r2[src]
        print(f"  seed {seed} fitted", flush=True)

    # ------------------------------------------------------------ levels
    REPORT = [
        ("PCA", "poly", "PCA+poly"),
        ("DcorSDR(seqX)", "linear", "dCor-SDR+lin"),
        ("INPCA-hybrid", "linear", "INPCA+lin (hybrid)"),
        ("INPCA(k4)", "linear", "INPCA(k4)+lin"),
        ("INPCA(k2)", "linear", "INPCA(k2)+lin"),
        ("SIR", "linear", "SIR+lin"),
        ("SAVE", "linear", "SAVE+lin"),
        ("PLS", "linear", "PLS+lin"),
        ("DvarSDR(seq,plain)", "linear", "dVar-PP(seq)+lin"),
        ("DvarSDR(joi,plain)", "linear", "dVar-PP(joint)+lin"),
        ("DvarSDR(seq,robust)", "linear", "dVar-PP(seq,rob)+lin"),
        ("PCA", "linear", "PCA+lin"),
        ("DvarSDR(joi,plain)", "poly", "dVar-PP(joint)+poly"),
    ]

    def best(seed, mat, m, spec):
        c = [(d, r2[(seed, mat, m, d, spec)]) for d in DIMS
             if (seed, mat, m, d, spec) in r2]
        return max(c, key=lambda t: t[1])   # (d*, r2*)

    rows = []
    for seed in SEEDS:
        for m, spec, lab in REPORT:
            for mat in dio.MATS:
                d, v = best(seed, mat, m, spec)
                rows.append(dict(seed=seed, label=lab, method=m, spec=spec,
                                 maturity=mat, d_star=d, r2=v,
                                 mspe=mspe(err[(seed, mat, m, d, spec)])))
    lev = pd.DataFrame(rows)
    write_csv(RESULTS / "ch6_bestd_levels.csv", lev, seeds=SEEDS,
                         script="ch06_best_d_dm_tests.py")
    piv = lev.pivot_table(index="label", columns="maturity", values="r2",
                          aggfunc="median")
    piv["mean"] = piv.mean(axis=1)
    print(f"\n=== best-over-d OOS R^2, published factors, no CP "
          f"(optimiser seeds {', '.join(map(str, SEEDS))}) ===")
    print(piv.sort_values("mean", ascending=False).round(3).to_string())
    if len(SEEDS) > 1:
        print("\nspread across seeds (max - min R^2), by method and maturity:")
        spread = (lev.pivot_table(index="label", columns="maturity", values="r2",
                                  aggfunc="max")
                  - lev.pivot_table(index="label", columns="maturity", values="r2",
                                    aggfunc="min"))
        print(spread.round(4).to_string())
        print(f"largest spread over any method and maturity: "
              f"{float(spread.to_numpy().max()):.4f} R^2")
    print("\nd* attained (median; a non-integer means the seeds disagree):")
    print(lev.pivot_table(index="label", columns="maturity",
                          values="d_star", aggfunc="median").to_string())

    # ------------------------------------------------- best-over-d DM tests
    PAIRS = [
        ("dCor-SDR+lin", ("DcorSDR(seqX)", "linear"), ("INPCA-hybrid", "linear")),
        ("dCor-SDR+lin", ("DcorSDR(seqX)", "linear"), ("INPCA(k4)", "linear")),
        ("dCor-SDR+lin", ("DcorSDR(seqX)", "linear"), ("PCA", "poly")),
        ("dCor-SDR+lin", ("DcorSDR(seqX)", "linear"), ("PCA", "linear")),
        ("dCor-SDR+lin", ("DcorSDR(seqX)", "linear"), ("SIR", "linear")),
        ("dCor-SDR+lin", ("DcorSDR(seqX)", "linear"), ("SAVE", "linear")),
        ("dCor-SDR+lin", ("DcorSDR(seqX)", "linear"), ("PLS", "linear")),
        ("dVar(seq)+lin", ("DvarSDR(seq,plain)", "linear"), ("PCA", "linear")),
        ("dVar(seq)+lin", ("DvarSDR(seq,plain)", "linear"), ("PCA", "poly")),
        ("dVar(seq)+lin", ("DvarSDR(seq,plain)", "linear"), ("INPCA-hybrid", "linear")),
        ("dVar(seq)+lin", ("DvarSDR(seq,plain)", "linear"), ("DcorSDR(seqX)", "linear")),
        ("dVar(seq)+lin", ("DvarSDR(seq,plain)", "linear"), ("SIR", "linear")),
        ("dVar(seq)+lin", ("DvarSDR(seq,plain)", "linear"), ("PLS", "linear")),
        ("dVar(seq)+lin", ("DvarSDR(seq,plain)", "linear"), ("SAVE", "linear")),
    ]
    out = []
    print(f"\n=== DM at each method's OWN best d (DM>0 => second arg worse), "
          f"optimiser seeds {', '.join(map(str, SEEDS))} ===")
    for seed in SEEDS:
        for lab, (ma, sa), (mb, sb) in PAIRS:
            for mat in dio.MATS:
                da, ra = best(seed, mat, ma, sa)
                db, rb = best(seed, mat, mb, sb)
                dm, p = dm_test(err[(seed, mat, mb, db, sb)],
                                err[(seed, mat, ma, da, sa)])
                out.append(dict(seed=seed, maturity=mat, focus=lab,
                                other=f"{mb}+{sb}",
                                d_focus=da, d_other=db, r2_focus=ra, r2_other=rb,
                                mspe_focus=mspe(err[(seed, mat, ma, da, sa)]),
                                mspe_other=mspe(err[(seed, mat, mb, db, sb)]),
                                DM=dm, p=p))
    dm_df = pd.DataFrame(out)
    write_csv(RESULTS / "ch6_bestd_dm.csv", dm_df, seeds=SEEDS,
                         script="ch06_best_d_dm_tests.py")
    # Median dR2 and median p, with the p range: a p-value that straddles a threshold
    # across seeds is not a significance result, and the range is what shows that.
    for lab in dm_df.focus.unique():
        s = dm_df[dm_df.focus == lab]
        print(f"\n-- focus {lab}")
        for other, g in s.groupby("other"):
            parts = []
            for mat, gm in g.groupby("maturity"):
                dr2 = (gm.r2_focus - gm.r2_other).median()
                parts.append(f"rx{int(mat)}: dR2={dr2:+.3f} "
                             f"p={gm.p.median():.3f}[{gm.p.min():.3f},"
                             f"{gm.p.max():.3f}]")
            print(f"   vs {other:26s} " + "  ".join(parts))

    # ------------------------------------------------ 2001 leverage anatomy
    # Reported at the reference seed, in detail; the chapter quotes these numbers.
    # The block below repeats the anatomy for every seed in SEEDS, which is that one
    # seed unless the list is widened for a stability check.
    print(f"\n=== 2001 leverage points on the published block (seed {SEED}) ===")
    fin_u = reg.build_unsupervised(F[train], seed=SEED)
    W = fin_u["DvarSDR(joi,plain)"].W
    w = W[:, 0]
    g = F @ w
    if (g[test] - g[train].mean()).min() < -(g[test] - g[train].mean()).max():
        g, w = -g, -w
    f1 = F[:, 0]
    idx = [int(np.where(dates == pd.Timestamp(s))[0][0])
           for s in ("2001-09-01", "2001-10-01")]
    for name, s in (("dVar joint dir", g), ("PCA factor 1", f1)):
        z = (s - s[train].mean()) / s[train].std()
        lo, hi = s[train].min(), s[train].max()
        print(f"  [{name}] train support [{lo:.2f},{hi:.2f}] "
              f"train kurt={kurtosis(s[train]):.2f} test kurt={kurtosis(s[test]):.2f}")
        for i in idx:
            print(f"    {dates.iloc[i].date()} score={s[i]:+.2f} z={z[i]:+.2f} "
                  f"cube={s[i]**3:+.0f} in-support={lo <= s[i] <= hi}")
        print(f"    largest |cube| train={np.abs(s[train]**3).max():.0f} "
              f"test={np.abs(s[test]**3).max():.0f}")
        # rank of the two months among test months by |z|
        te_idx = np.where(test)[0]
        order = sorted(te_idx, key=lambda i: -abs(z[i]))
        print("    top-4 |z| test months: " + ", ".join(
            f"{dates.iloc[i].date()}(z={z[i]:+.1f})" for i in order[:4]))
    print("  dVar joint direction loadings on f1..f8: "
          + " ".join(f"{v:+.2f}" for v in w))
    print("  PCA f1 loading (identity): 1 on f1")

    # ---- the same anatomy across every seed, as numbers rather than prose
    anat = []
    for seed in SEEDS:
        Wj = reg.build_unsupervised(F[train], seed=seed)["DvarSDR(joi,plain)"].W
        wj = Wj[:, 0]
        s = F @ wj
        if (s[test] - s[train].mean()).min() < -(s[test] - s[train].mean()).max():
            s, wj = -s, -wj
        z = (s - s[train].mean()) / s[train].std()
        anat.append(dict(seed=seed,
                         train_kurtosis=float(kurtosis(s[train])),
                         test_kurtosis=float(kurtosis(s[test])),
                         z_2001_09=float(z[idx[0]]), z_2001_10=float(z[idx[1]]),
                         train_min=float(s[train].min()),
                         train_max=float(s[train].max()),
                         loading_f1=float(wj[0]), loading_f3=float(wj[2]),
                         loading_f6=float(wj[5])))
    an = pd.DataFrame(anat)
    write_csv(RESULTS / "ch6_dvar_anatomy.csv", an, seeds=SEEDS,
                         script="ch06_best_d_dm_tests.py")
    print(f"\n=== dVar direction anatomy, seeds {', '.join(map(str, SEEDS))} ===")
    print(f"  test kurtosis  median {an.test_kurtosis.median():.1f} "
          f"[{an.test_kurtosis.min():.1f}, {an.test_kurtosis.max():.1f}]")
    print(f"  train kurtosis median {an.train_kurtosis.median():.2f} "
          f"[{an.train_kurtosis.min():.2f}, {an.train_kurtosis.max():.2f}]")
    print(f"  z(2001-09)     median {an.z_2001_09.median():+.1f} "
          f"[{an.z_2001_09.min():+.1f}, {an.z_2001_09.max():+.1f}]")
    print(f"  z(2001-10)     median {an.z_2001_10.median():+.1f} "
          f"[{an.z_2001_10.min():+.1f}, {an.z_2001_10.max():+.1f}]")
    print("  wrote results/ch6_dvar_anatomy.csv")


if __name__ == "__main__":
    main()
