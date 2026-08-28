"""
sheng_yin_ch6.py — the Chapter 6 forecast experiment run with Sheng & Yin's own
dCov-SDR solver in place of this thesis's dCor optimiser.

Same panel, same split, same second stage as `peer_dm_published.py`: the
published (look-ahead) Ludvigson-Ng factor block, a linear second stage, no CP,
excess bond returns at maturities 2-5 years, h = 12 months. The only thing that
changes is the direction finder:

    DcorSDR(seqX)   this thesis — sequential Riemannian dCor search
    ShengYin(d)     `PP_Dcor/sheng_yin.py` — their whitened SQP solve of the
                    d-dimensional dCov problem, refitted at each d

Both are supervised and are therefore refitted on the training half at every
maturity. Directions are frozen after fitting; the second-stage OLS is fit on
train and applied out of sample, exactly as in the rest of the chapter.

Reported per maturity and d: out-of-sample MSPE for each finder, and the
Diebold-Mariano test between them (positive DM => Sheng-Yin worse).

Writes results/sheng_yin_ch6.csv.
"""

# Thesis:   Chapter 6, their solver on this panel
# Writes:   results/sheng_yin_ch6.csv
# Original: Real Data Experiment/src/sheng_yin_ch6.py on the thesis branch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import realdata.dataio as dio
from realdata.regression import predict
from realdata.metrics import mspe, dm_test
import realdata.sdr_registry as reg
from realdata.dcorlin import blocks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from dpp.supervised.sheng_yin import sheng_yin_sdr                                  # noqa: E402

#: The one optimiser seed Chapter 6 reports, as in peer_dm_published.py. The panel is
#: fixed, so a seed varies only the starting points — theirs (the 500 perturbations
#: of `dcsol.m`) as well as ours — and the chapter reports a single run (decision of
#: 2026-08-19). widening the list still turns this into a stability check.
SEEDS = (0,)

N_PERTURB = 500
DIMS = range(1, 8)   # d = 8 would span the whole 8-factor block for both methods


def main(seeds=SEEDS):
    dates, F_ours, F_ln, tgt, train, test, tr, te = blocks()
    rows = []
    t0 = time.time()
    for seed in seeds:
        for mat in dio.MATS:
            y = tgt[f"yr{mat}"].to_numpy()
            ytr = y[tr + dio.H]
            yte = y[te + dio.H]
            sup = reg.build_supervised(F_ln[train], ytr, seed=seed)

            for d in DIMS:
                Z = sup["DcorSDR(seqX)"].project(F_ln, d)
                e_dcor = predict(Z, d, "linear", train, test, ytr, cp=None) - yte

                B, info = sheng_yin_sdr(F_ln[train], ytr, d=d,
                                        n_perturb=N_PERTURB, seed=seed)
                e_sy = predict(F_ln @ B, d, "linear", train, test, ytr,
                               cp=None) - yte

                dm, p = dm_test(e_sy, e_dcor)   # >0 => Sheng-Yin worse
                rows.append(dict(seed=seed, maturity=mat, d=d,
                                 mspe_dcor=mspe(e_dcor), mspe_sy=mspe(e_sy),
                                 DM=dm, p=p, sy_seconds=round(info["seconds"], 3),
                                 sy_objective=round(info["objective"], 6)))
            print(f"  seed {seed} maturity {mat} done ({time.time()-t0:.0f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    write_csv(RESULTS / "sheng_yin_ch6.csv", df, seeds=seeds,
                         script="ch06_shengyin_on_panel.py",
                         n_perturb=N_PERTURB)
    print(f"wrote results/sheng_yin_ch6.csv ({len(df)} rows)")

    print(f"\n=== Sheng-Yin vs DcorSDR(seqX), linear second stage, published "
          f"factors, {len(seeds)} seeds ===")
    print(f"{'d':>2} | {'MSPE dCor':>18} | {'MSPE Sheng-Yin':>18} | "
          f"{'SY sig-worse':>12} | {'SY sig-better':>13} | {'SY sec':>7}")
    for d, g in df.groupby("d"):
        worse = int(((g.DM > 0) & (g.p < 0.10)).sum())
        better = int(((g.DM < 0) & (g.p < 0.10)).sum())
        print(f"{d:>2} | {g.mspe_dcor.median():7.2f} "
              f"[{g.mspe_dcor.min():6.2f},{g.mspe_dcor.max():6.2f}] | "
              f"{g.mspe_sy.median():7.2f} "
              f"[{g.mspe_sy.min():6.2f},{g.mspe_sy.max():6.2f}] | "
              f"{worse:>4}/{len(g):<7} | {better:>5}/{len(g):<7} | "
              f"{g.sy_seconds.median():6.1f}")

    s = df[df.d <= 3]
    print(f"\nd <= 3 pooled: median MSPE dCor {s.mspe_dcor.median():.2f}, "
          f"Sheng-Yin {s.mspe_sy.median():.2f}; "
          f"Sheng-Yin significantly worse in "
          f"{int(((s.DM > 0) & (s.p < 0.10)).sum())} of {len(s)}, "
          f"significantly better in "
          f"{int(((s.DM < 0) & (s.p < 0.10)).sum())} of {len(s)}")


if __name__ == "__main__":
    main()
