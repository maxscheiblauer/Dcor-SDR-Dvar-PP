"""Does an informative (SIR) start help? — supervisor request, PDF p. 19.

She asked for "one informative initialization; e.g. linear SIR for supervised
approach or PCA for unsupervised one", against the paragraph that says the
optimiser is started from random directions.

The unsupervised half is already answered: PCA is the r = 1 start there.  This
script answers the supervised half, which OLS does not, because SIR is precisely
the estimator that sees the symmetric structure OLS is blind to.

Compared, at equal total restart budget:

  random  : all restarts random                       (current default)
  sir     : restart 0 seeded from SIR, rest random
  sir1    : SIR alone, no gradient ascent at all      (the baseline itself)

reported as the angle to the true direction and the attained dCor^2 (the
U-statistic of `dcor_u`, the scale used throughout the thesis).

Writes results_sir_init.csv.

Run:  python scripts/ch03_sir_initialisation.py
"""

# Thesis:   Chapter 3, §3.3 initialisation
# Writes:   results/results_sir_init.csv
# Original: PP_Dcor/sir_initialisation_study.py on the thesis branch.
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

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent


from dpp.supervised.data_generator import data_generator          # noqa: E402
from dpp.supervised.dcor_optimizer import dcor_u, optimize_dcor   # noqa: E402
from dpp.supervised.sdr_baselines import sir                      # noqa: E402

#: Five data seeds. The first is this study's original single seed, kept first so its
#: numbers stay identifiable; the rest are the project-wide list. Until 2026-08-11 the
#: §3.3 initialisation result rested on 20260808 alone.
SEEDS = (42, 7, 123, 2024, 5, 2026, 17, 99, 777, 31337)
N = 200
RESTARTS = 3
NONLINEARITIES = ("square", "sine", "abs", "cubic", "tanh")
NOISES = (0.0, 0.5, 1.0, 2.0)
PS = (5, 20)


def angle_to_truth(beta, true_dirs):
    v = np.asarray(true_dirs)[:, 0]
    c = abs(float(beta @ v)) / (np.linalg.norm(beta) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(min(1.0, c))))


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        for p in PS:
            for nl in NONLINEARITIES:
                for noise in NOISES:
                    X, Y, true_dirs = data_generator(N, p, 1, nl, noise, seed)
                    Yv = np.asarray(Y).reshape(-1)

                    # SIR on its own, no gradient step
                    b_sir = np.asarray(sir(X, Yv, k=1)).reshape(-1)
                    b_sir = b_sir / np.linalg.norm(b_sir)
                    rows.append(dict(
                        seed=seed, p=p, nonlinearity=nl, noise=noise, start="sir1",
                        angle=angle_to_truth(b_sir, true_dirs),
                        dcor2=float(dcor_u(X @ b_sir, Yv)), spread=np.nan))

                    # all-random restarts (current default)
                    b, v, info = optimize_dcor(X, Yv, init_method="random",
                                               n_restarts=RESTARTS, seed=seed,
                                               max_iter=300)
                    rows.append(dict(
                        seed=seed, p=p, nonlinearity=nl, noise=noise, start="random",
                        angle=angle_to_truth(b, true_dirs), dcor2=float(v),
                        spread=info["val_spread"]))

                    # SIR-seeded, same total budget
                    b, v, info = optimize_dcor(X, Yv, init_method="sir",
                                               n_restarts=RESTARTS, seed=seed,
                                               max_iter=300)
                    rows.append(dict(
                        seed=seed, p=p, nonlinearity=nl, noise=noise, start="sir",
                        angle=angle_to_truth(b, true_dirs), dcor2=float(v),
                        spread=info["val_spread"]))
        print(f"  seed {seed} done ({len(rows)} rows)", flush=True)

    df = pd.DataFrame(rows)
    write_csv(RESULTS / "results_sir_init.csv", df, seeds=SEEDS,
                         script="ch03_sir_initialisation.py",
                         n_restarts=RESTARTS)

    # Median over seeds, stated rather than left to pivot_table's default mean: with
    # five replicates and one possible outlier the mean is the wrong summary, and the
    # counts below are read straight off these tables.
    piv_a = df.pivot_table(index=["p", "nonlinearity", "noise"],
                           columns="start", values="angle", aggfunc="median")
    piv_d = df.pivot_table(index=["p", "nonlinearity", "noise"],
                           columns="start", values="dcor2", aggfunc="median")
    rng_a = df.pivot_table(index=["p", "nonlinearity", "noise"],
                           columns="start", values="angle",
                           aggfunc=["min", "max"])

    print("=" * 74)
    print(f"Recovery angle to the true direction in degrees, median over "
          f"{len(SEEDS)} seeds {list(SEEDS)}")
    print("=" * 74)
    print(f"{'p':>3s} {'response':10s} {'noise':>6s} "
          f"{'SIR only':>9s} {'random':>9s} {'SIR-seeded':>11s} {'change':>8s}")
    print("-" * 74)
    for (p, nl, noise), r in piv_a.iterrows():
        print(f"{p:3d} {nl:10s} {noise:6.1f} {r['sir1']:9.2f} "
              f"{r['random']:9.2f} {r['sir']:11.2f} "
              f"{r['sir'] - r['random']:+8.2f}")

    print()
    print("=" * 74)
    print("Summary")
    print("=" * 74)
    d = piv_a["sir"] - piv_a["random"]
    print(f"(medians over {len(SEEDS)} seeds; a difference is only meaningful if it "
          f"exceeds the seed spread, which the CSV carries per replicate)")
    print(f"configurations where the SIR start improves the angle by > 1 deg: "
          f"{int((d < -1).sum())} of {len(d)}")
    print(f"configurations where it worsens the angle by > 1 deg:            "
          f"{int((d > 1).sum())} of {len(d)}")
    print(f"largest improvement {d.min():+.2f} deg, largest deterioration "
          f"{d.max():+.2f} deg")
    dd = piv_d["sir"] - piv_d["random"]
    print(f"dCor^2 change: best {dd.max():+.4f}, worst {dd.min():+.4f}")
    print()
    print("The `sine` configurations, which are the failure the request targets:")
    for (p, nl, noise), r in piv_a.iterrows():
        if nl == "sine":
            print(f"  p={p:2d} noise={noise:.1f}: SIR only {r['sir1']:6.2f}, "
                  f"random {r['random']:6.2f}, SIR-seeded {r['sir']:6.2f}")
