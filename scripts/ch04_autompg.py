"""
autompg_ch4.py — the AutoMPG illustration of Chapter 4.

The supervisor's mark on the Sheng & Yin request reads: "Use also the AutoMPG
data from that paper to illustrate". This is that illustration, and it is the
one real data set in the chapter — the bond panel of Chapter 6 does different
work.

Data
----
The UCI *Auto MPG* set (Quinlan, 1993): 398 cars, fuel consumption in miles per
gallon against engine and body measurements. Six of the eight recorded fields
have missing values only in `horsepower`, and the six rows affected are dropped,
which leaves n = 392 — the sample size the SDR literature reports for this set.
The predictors are the five continuous ones (displacement, horsepower, weight,
acceleration, model year) together with the cylinder count, standardised to zero
mean and unit variance so that a direction's coefficients can be read against
each other. The file is cached under `Real Data Experiment/data/` on first run;
subsequent runs do not touch the network.

What is compared
----------------
Four ways of choosing a single index, at d = 1:

    dCor-PP     this thesis, Riemannian gradient search on the sphere
    Sheng-Yin   `PP_Dcor/sheng_yin.py`, their whitened SQP solve
    SIR         sliced inverse regression, the classical inverse-moment method
    OLS         the least-squares coefficient vector, normalised

and three things are reported for each: the direction itself, the dependence
its index retains (dCor^2_u, the estimator used everywhere in this thesis), and
a five-fold cross-validated R^2 of a cubic fit on the index — a prediction
number, so that the comparison does not rest solely on the criterion that one
of the four methods is built to maximise.

Writes results/autompg_ch4.csv and the figure `autompg_indices.png`.
"""

# Thesis:   Chapter 4, tab:p1-autompg
# Writes:   results/autompg_ch4.csv
# Original: Real Data Experiment/src/autompg_ch4.py on the thesis branch.
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
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

from dpp.supervised.dcor_optimizer import optimize_dcor, dcor_u           # noqa: E402
from dpp.supervised.sdr_baselines import sir                              # noqa: E402
from dpp.supervised.sheng_yin import sheng_yin_sdr                        # noqa: E402


URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
CACHE = DATA / "auto-mpg.data"

COLUMNS = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model_year", "origin", "name"]
PREDICTORS = ["cylinders", "displacement", "horsepower", "weight",
              "acceleration", "model_year"]

SEEDS = (42, 7, 123, 2024, 5)
N_RESTARTS = 5
MAX_ITER = 150
N_PERTURB = 500
N_FOLDS = 5
POLY_DEGREE = 3


def load():
    """The cached UCI file, downloaded once. Returns X (standardised), y, names."""
    if not CACHE.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as fh:
            CACHE.write_bytes(fh.read())
    raw = pd.read_csv(CACHE, sep=r"\s+", names=COLUMNS, na_values="?")
    df = raw.dropna(subset=["horsepower"]).reset_index(drop=True)
    X = df[PREDICTORS].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    y = df["mpg"].to_numpy(float)
    return X, y, PREDICTORS, len(raw)


def cv_r2(index, y, seed, folds=N_FOLDS, degree=POLY_DEGREE):
    """Cross-validated R^2 of a polynomial fit of y on a single index."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    parts = np.array_split(order, folds)
    pred = np.empty_like(y)
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        coef = np.polyfit(index[tr], y[tr], degree)
        pred[te] = np.polyval(coef, index[te])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def directions(X, y, seed):
    """One direction per method, each returned as a unit vector."""
    out = {}

    t = time.time()
    beta, _, _ = optimize_dcor(X, y, init_method='random',
                               optimizer='gradient_ascent',
                               n_restarts=N_RESTARTS, max_iter=MAX_ITER,
                               seed=seed)
    out['dCor-PP (this thesis)'] = (beta, time.time() - t)

    B, info = sheng_yin_sdr(X, y, d=1, n_perturb=N_PERTURB, seed=seed)
    b = B[:, 0] / np.linalg.norm(B[:, 0])
    out['Sheng-Yin (SQP)'] = (b, info['seconds'])

    t = time.time()
    b = sir(X, y, k=1)[:, 0]
    out['SIR'] = (b / np.linalg.norm(b), time.time() - t)

    t = time.time()
    b, *_ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
    out['OLS'] = (b / np.linalg.norm(b), time.time() - t)
    return out


def run():
    X, y, names, n_raw = load()
    print(f"  auto-mpg: {n_raw} rows read, {len(y)} kept after dropping "
          f"missing horsepower; p = {X.shape[1]}", flush=True)
    rows = []
    for seed in SEEDS:
        for method, (beta, secs) in directions(X, y, seed).items():
            beta = beta * np.sign(beta[names.index('weight')] or 1.0)  # fix the sign
            index = X @ beta
            row = dict(seed=seed, method=method, n=len(y), p=X.shape[1],
                       dcor2_u=round(float(dcor_u(index, y)), 6),
                       cv_r2=round(float(cv_r2(index, y, seed)), 6),
                       seconds=round(float(secs), 3))
            row.update({f'b_{nm}': round(float(v), 4)
                        for nm, v in zip(names, beta)})
            rows.append(row)
        print(f"  seed {seed} done", flush=True)
    return rows, (X, y, names)


def summarise(rows):
    df = pd.DataFrame(rows)
    print(f"\n=== AutoMPG, d = 1, median over {len(SEEDS)} seeds ===")
    print(df.groupby('method')[['dcor2_u', 'cv_r2', 'seconds']]
          .median().round(4).to_string())
    print("\n=== direction coefficients, median over seeds ===")
    coef_cols = [c for c in df.columns if c.startswith('b_')]
    print(df.groupby('method')[coef_cols].median().round(3).to_string())
    print("\n=== spread of the recovered direction across seeds "
          "(max angle to the first seed, degrees) ===")
    for method, g in df.groupby('method'):
        B = g[coef_cols].to_numpy(float)
        ref = B[0] / np.linalg.norm(B[0])
        ang = [np.degrees(np.arccos(min(1.0, abs(float(b @ ref) /
                                                 np.linalg.norm(b)))))
               for b in B]
        print(f"  {method:<24} {max(ang):.2f}")
    return df


if __name__ == '__main__':
    rows, (X, y, names) = run()
    df = summarise(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = write_csv(RESULTS / 'autompg_ch4.csv', df, seeds=SEEDS,
                                 script='ch04_autompg.py',
                                 source=URL, n_restarts=N_RESTARTS,
                                 n_perturb=N_PERTURB, folds=N_FOLDS,
                                 poly_degree=POLY_DEGREE)
    print(f"\nWrote results/autompg_ch4.csv ({len(df)} rows)")
