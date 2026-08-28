"""
peer_dm_published.py — the poc_dm.py peer comparison re-run on the PUBLISHED
(look-ahead) Ludvigson-Ng factor block, with the LINEAR dCor-SDR second stage.

Chapter 6 was reduced to a single factor block (the published one) and dropped
the dCor+poly variant entirely, so the peer Diebold-Mariano table has to be
recomputed on that block with dCor-SDR(seqX)+linear as the method under test.
No CP anywhere. Writes results/peer_dm_published.csv.

The panel is fixed, so a seed varies only the optimiser's starting points, and Chapter 6
reports one run on one benchmark rather than a distribution over runs. SEEDS is therefore
(0,) by decision of 2026-08-19, matching the rest of the chapter. The per-seed counts
below still print, so widening the list turns this into a stability check; the chapter
quotes seed 0.
"""

# Thesis:   Chapter 6, the peer comparison on the published block
# Writes:   results/peer_dm_published.csv
# Original: Real Data Experiment/src/peer_dm_published.py on the thesis branch.
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
from realdata.regression import predict
from realdata.metrics import mspe, dm_test
import realdata.sdr_registry as reg
from realdata.dcorlin import blocks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

PEERS = [("PLS", "linear"), ("PLS", "poly"), ("SIR", "linear"),
         ("SIR", "poly"), ("SAVE", "linear"), ("SAVE", "poly")]

#: The one optimiser seed Chapter 6 reports. See the module docstring.
SEEDS = (0,)


def main(seeds=SEEDS):
    dates, F_ours, F_ln, tgt, train, test, tr, te = blocks()
    rows = []
    for seed in seeds:
        for mat in dio.MATS:
            y = tgt[f"yr{mat}"].to_numpy()
            ytr = y[tr + dio.H]; yte = y[te + dio.H]
            sup = reg.build_supervised(F_ln[train], ytr, seed=seed)

            def err(finder, d, spec):
                Z = finder.project(F_ln, d)
                if Z is None or Z.shape[1] < d:
                    return None
                return predict(Z, d, spec, train, test, ytr, cp=None) - yte

            for d in range(1, 9):
                e_dcor = err(sup["DcorSDR(seqX)"], d, "linear")
                if e_dcor is None:
                    continue
                for peer, spec in PEERS:
                    e_p = err(sup[peer], d, spec)
                    if e_p is None:
                        continue
                    dm, p = dm_test(e_p, e_dcor)   # >0 => peer worse (dCor better)
                    rows.append(dict(seed=seed, maturity=mat, d=d,
                                     peer=f"{peer}+{spec}",
                                     dcor="DcorSDR(seqX)+linear", DM=dm, p=p,
                                     mspe_peer=mspe(e_p), mspe_dcor=mspe(e_dcor)))
        print(f"  seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    write_csv(RESULTS / "peer_dm_published.csv", df, seeds=seeds,
                         script="ch06_peer_dm_published.py")
    print(f"wrote results/peer_dm_published.csv ({len(df)} rows)")

    s = df[df.d <= 3]
    print(f"\n=== dCor-SDR(seqX)+lin vs peers, published factors, d<=3, no CP, "
          f"optimiser seeds {', '.join(map(str, seeds))} ===")
    print(f"  median MSPE dCor = {s.mspe_dcor.median():.2f} "
          f"[{s.mspe_dcor.min():.2f}, {s.mspe_dcor.max():.2f}]")
    for peer in s.peer.unique():
        ss = s[s.peer == peer]
        per_seed = []
        for seed, g in ss.groupby("seed"):
            per_seed.append(int(((g.DM > 0) & (g.p < 0.10)).sum()))
        n_per_seed = len(ss) // max(len(seeds), 1)
        worse = ((ss.DM < 0) & (ss.p < 0.10)).sum()
        print(f"  {peer:14s}: dCor sig-better per seed {per_seed} of "
              f"{n_per_seed}, sig-worse {worse}/{len(ss)} pooled, "
              f"min p={ss.p.min():.3f}, "
              f"median MSPE peer={ss.mspe_peer.median():.2f}")


if __name__ == "__main__":
    main()
