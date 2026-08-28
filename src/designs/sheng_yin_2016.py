"""
sheng_yin_2016_study.py

Replication of the simulation design of Sheng & Yin (2016) — the study the
supervisor asked for ("I would add another simulation study, the one from
Sheng 2016") — with this project's optimiser, their own SQP solver and SIR run
on identical data.

Where the design comes from
---------------------------
Sheng & Yin publish no code and the paper is not in `thesis/references/`.
Wu & Chen (2021, `wu2021mm`, pp. 12-13, in the reference set) state that they
use "the same simulation settings as in Sheng and Yin (2016)" and reproduce the
design in full: three models, three predictor kinds each, two (n, p) pairs, and
the subspace distance used to score them. That is the source transcribed here.
Any discrepancy with the original paper is a discrepancy with Wu & Chen's
restatement of it, which is stated in the thesis caption.

The design
----------
Let eps, eps1, eps2 be independent standard normal. With beta1, beta2, beta3
having first six components (1,0,0,0,0,0), (0,1,0,0,0,0), (1,0.5,1,0,0,0) and
zeros beyond,

    A:  Y = (beta1'X)^2 + (beta2'X) + 0.1 eps            d = 2
    B:  Y = sign(2 beta1'X + eps1) log|2 beta2'X + 4 + eps2|   d = 2
    C:  Y = exp(beta3'X) eps                             d = 1

Each model is run with three kinds of predictor: (1) standard normal,
(2) non-normal continuous, (3) **discrete**. Part (3) is the evidence for the
second supervisor mark on the same page — that continuous predictors are not a
requirement of the objective.

Two accuracy criteria are recorded per fit.

* `angle` — the **mean principal angle** to the true central subspace, in
  degrees, from `evaluation.principal_angles`. This is the criterion Chapter 4
  uses everywhere else (the deflation and joint-versus-sequential tables), and
  since 2026-08-14 it is the one the chapter's prose quotes for these models
  too, so that one metric runs through the whole chapter.
* `delta_m` — Sheng & Yin's own criterion, the spectral distance between the
  projections onto the estimated and the true central subspace,

      Delta_m(P_hat, P) = || P_hat - P ||_2,   P = B (B'B)^-1 B',

  which lies in [0, 1] for subspaces of equal dimension; 0 is exact recovery.
  Kept so that the run can be cross-checked against the published table, and
  deliberately **not** featured in the thesis prose.

`delta_m` is undefined for an estimate of the wrong rank, so it is NaN for the
OLS arm wherever d > 1 (models A and B): OLS returns a single direction, and the
spectral distance between a rank-1 and a rank-2 projection is 1 whatever the
direction. The principal angle handles the rank mismatch correctly — with a
(p, 1) estimate against a (p, 2) truth it returns the one angle between the OLS
line and the true plane — and that is what the OLS column reports there. The
thesis caption says so.

Four estimators are run on identical data: this project's gradient search, their
SQP solver, SIR and OLS.

Two blocks of configurations
----------------------------
`accuracy` — the paper's own design: three models x three predictor kinds x two
(n, p) pairs. The paper averages 100 datasets, which is affordable at
(n, p) = (100, 6) and is run here; at (500, 20) the SQP solver costs ~14 s per
fit, so 25 replicates are run instead and the count is reported with the
numbers. Both counts are literals below and are written into the header of the
CSV this produces.

`scaling` — added 2026-08-14 and new to this design: model A, part (1), at
n = 500 and p in {6, 20, 50, 100}, five replicates. It exists for two reasons
the paper's own (n, p) pairs cannot serve. First, the chapter claims the
closed-form gradient scales better in the predictor dimension than a
finite-difference constrained solve, and two p values are not a curve. Second,
nothing else in the chapter goes above p = 20, so the claim rested on the
p = 6-to-20 step alone. The gap the block measures is large but its size is
machine-dependent: the median fit at p = 100 cost 1.5 s for the gradient search
against 83.9 s for their solver on the machine the shipped CSV was produced on,
and 3.2 s against 113.1 s on another, so read the order of magnitude and not the
constant. The accuracy columns beside them do reproduce exactly.

Writes results_sheng_yin_2016.csv (one row per block, model, part, size,
replicate and method).
"""
from __future__ import annotations

from csvout import RESULTS, write_csv

import os
import sys
import time
from pathlib import Path

import numpy as np


from dpp.supervised.dcor_optimizer import dcor_u                    # noqa: E402
from dpp.supervised.evaluation import principal_angles              # noqa: E402
from dpp.supervised.pp_helpers import seq_pp                        # noqa: E402
from dpp.supervised.sdr_baselines import sir                        # noqa: E402
from dpp.supervised.sheng_yin import sheng_yin_sdr                  # noqa: E402

import warnings
warnings.filterwarnings("ignore")

#: Base of the replicate seeds. Each dataset is drawn from
#: ``default_rng([BASE_SEED, config_index, replicate])``, so a replicate is
#: reproducible on its own and no two configurations share a stream.
BASE_SEED = 20260813

#: (n, p) pairs of the paper, with the replicate count used for each.
SIZES = [(100, 6), (500, 20)]
REPS_DEFAULT = (100, 25)

#: The scaling block: model A, part (1), one n, four predictor dimensions. Two of
#: these repeat configurations of the accuracy block; they are re-run here so that
#: the curve is measured at one n throughout and is readable on its own.
SCALING_P = [6, 20, 50, 100]
SCALING_N = 500
SCALING_REPS = 5

#: Perturbation count of `dcsol.m`, unchanged from their code.
N_PERTURB = 500

#: Budget of this project's optimiser, the same as `step4_projection_pursuit.py`.
N_RESTARTS = 5
MAX_ITER = 150


def _reps() -> tuple[int, int]:
    """The replicate counts.  Literals, so shortening them is a visible edit."""
    return REPS_DEFAULT


# ── the three models ─────────────────────────────────────────────────────────

def _betas(p):
    """beta1, beta2, beta3 of the paper, padded with zeros to length p."""
    b1 = np.zeros(p); b1[0] = 1.0
    b2 = np.zeros(p); b2[1] = 1.0
    b3 = np.zeros(p); b3[0] = 1.0; b3[1] = 0.5; b3[2] = 1.0
    return b1, b2, b3


def _predictors(model, part, n, p, rng):
    """Part (1) standard normal, (2) non-normal continuous, (3) discrete."""
    if part == 1:
        return rng.standard_normal((n, p))
    if part == 2:
        if model == 'A':                       # (x_i + 2)/5 ~ Beta(0.75, 1)
            return 5.0 * rng.beta(0.75, 1.0, size=(n, p)) - 2.0
        if model == 'B':                       # x_i ~ U(-2, 2)
            return rng.uniform(-2.0, 2.0, size=(n, p))
        return 2.0 * rng.beta(1.5, 1.0, size=(n, p)) - 1.0   # (x_i + 1)/2 ~ Beta(1.5, 1)
    if part == 3:
        if model == 'A':
            return rng.poisson(1.0, size=(n, p)).astype(float)
        if model == 'B':
            return rng.binomial(10, 0.1, size=(n, p)).astype(float)
        X = rng.poisson(1.0, size=(n, p)).astype(float)      # model C
        X[:, 5] = rng.binomial(10, 0.3, size=n)              # x_6 is different
        return X
    raise ValueError(part)


def make_data(model, part, n, p, rng):
    """Return X, Y and the (p, d) true basis of the central subspace."""
    X = _predictors(model, part, n, p, rng)
    b1, b2, b3 = _betas(p)
    if model == 'A':
        Y = (X @ b1) ** 2 + (X @ b2) + 0.1 * rng.standard_normal(n)
        B = np.column_stack([b1, b2])
    elif model == 'B':
        e1 = rng.standard_normal(n)
        e2 = rng.standard_normal(n)
        Y = np.sign(2.0 * (X @ b1) + e1) * np.log(np.abs(2.0 * (X @ b2) + 4.0 + e2))
        B = np.column_stack([b1, b2])
    elif model == 'C':
        Y = np.exp(X @ b3) * rng.standard_normal(n)
        B = b3.reshape(-1, 1)
    else:
        raise ValueError(model)
    return X, Y, B


MODELS = {'A': 2, 'B': 2, 'C': 1}


# ── the two accuracy criteria ────────────────────────────────────────────────

def _as_matrix(B, p):
    return np.asarray(B, float).reshape(p, -1)


def mean_angle(B_hat, B_true):
    """Mean principal angle to the true central subspace, in degrees.

    Delegates to ``evaluation.principal_angles``, the utility the deflation and
    joint-versus-sequential tables of Chapter 4 already use, so that one
    implementation serves the whole chapter. With a rank-1 estimate against a
    rank-2 truth it returns a single angle — that of the estimated line to the
    true plane — which is what the OLS arm reports at d > 1.
    """
    p = len(B_true)
    return float(np.mean(principal_angles(_as_matrix(B_hat, p),
                                          _as_matrix(B_true, p))))


def delta_m(B_hat, B_true):
    """Spectral distance between the two orthogonal projections, ||P_hat - P||_2.

    NaN when the estimate and the truth have different rank: the quantity is 1
    for any rank-1 estimate of a rank-2 subspace and carries no information.
    """
    p = len(B_true)
    A, T = _as_matrix(B_hat, p), _as_matrix(B_true, p)
    if A.shape[1] != T.shape[1]:
        return float('nan')

    def proj(M):
        Q, _ = np.linalg.qr(M)
        return Q @ Q.T
    return float(np.linalg.norm(proj(A) - proj(T), 2))


def _mean_dcor2(B_hat, X, Y):
    """Mean per-column dCor^2_u of the estimated directions, for reference."""
    Z = X @ _as_matrix(B_hat, X.shape[1])
    return float(np.mean([dcor_u(Z[:, j], Y) for j in range(Z.shape[1])]))


def ols(X, Y, k=1):
    """The OLS direction, unit-normalised. A single direction whatever k is.

    OLS estimates one direction of the central subspace and there is no
    inverse-moment extension of it to a larger subspace, so ``k`` is accepted for
    signature symmetry with :func:`sdr_baselines.sir` and ignored. That the arm
    supplies one direction where the model has two is a property of OLS, and it
    is scored as such by :func:`mean_angle`.
    """
    b, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(Y, float), rcond=None)
    nrm = np.linalg.norm(b)
    return (b / nrm if nrm > 1e-12 else b).reshape(-1, 1)


# ── the run ──────────────────────────────────────────────────────────────────

def _score(rows, common, method, B_hat, X, Y, B_true, seconds):
    """Append one scored row. A failed fit lands as NaN rather than being dropped."""
    if B_hat is None:
        rows.append(dict(common, method=method, angle=float('nan'),
                         delta_m=float('nan'), dcor2_u=float('nan'),
                         seconds=round(seconds, 3)))
        return
    rows.append(dict(common, method=method,
                     angle=round(mean_angle(B_hat, B_true), 4),
                     delta_m=round(delta_m(B_hat, B_true), 6),
                     dcor2_u=round(_mean_dcor2(B_hat, X, Y), 6),
                     seconds=round(seconds, 3)))


def _one_replicate(rows, common, X, Y, B_true, d, rep):
    """Run and score all four estimators on one data set."""
    t = time.time()
    W, _ = seq_pp(X, Y, d, deflation='X_deflation',
                  n_restarts=N_RESTARTS, max_iter=MAX_ITER, seed=rep)
    _score(rows, common, 'dCor-PP (this thesis)', W, X, Y, B_true, time.time() - t)

    B_sy, info = sheng_yin_sdr(X, Y, d=d, n_perturb=N_PERTURB, seed=rep)
    _score(rows, common, 'Sheng-Yin (SQP)', B_sy, X, Y, B_true, info['seconds'])

    for method, fn in (('SIR', sir), ('OLS', ols)):
        t = time.time()
        try:
            B = fn(X, Y, k=d)
        except Exception:                         # singular slice covariance
            B = None
        _score(rows, common, method, B, X, Y, B_true, time.time() - t)


def run():
    reps_small, reps_large = _reps()
    rows = []
    t_start = time.time()

    # ── the paper's own design ───────────────────────────────────────────────
    configs = [(m, part, n, p)
               for m in MODELS for part in (1, 2, 3) for (n, p) in SIZES]
    for cid, (model, part, n, p) in enumerate(configs):
        d = MODELS[model]
        n_rep = reps_small if n == 100 else reps_large
        for rep in range(n_rep):
            rng = np.random.default_rng([BASE_SEED, cid, rep])
            X, Y, B_true = make_data(model, part, n, p, rng)
            _one_replicate(rows,
                           dict(block='accuracy', model=model, part=part,
                                n=n, p=p, d=d, rep=rep),
                           X, Y, B_true, d, rep)
        print(f"  [accuracy] model {model} part {part}  (n,p)=({n},{p})  "
              f"{n_rep} replicates done  [{time.time() - t_start:.0f}s]", flush=True)

    # ── the scaling block: one model, one n, four dimensions ────────────────
    # The stream base is offset past the accuracy block's config indices so that
    # no replicate of the two shares an RNG stream.
    base = len(configs)
    for k, p in enumerate(SCALING_P):
        d = MODELS['A']
        for rep in range(SCALING_REPS):
            rng = np.random.default_rng([BASE_SEED, base + k, rep])
            X, Y, B_true = make_data('A', 1, SCALING_N, p, rng)
            _one_replicate(rows,
                           dict(block='scaling', model='A', part=1,
                                n=SCALING_N, p=p, d=d, rep=rep),
                           X, Y, B_true, d, rep)
        print(f"  [scaling]  model A part 1  (n,p)=({SCALING_N},{p})  "
              f"{SCALING_REPS} replicates done  [{time.time() - t_start:.0f}s]",
              flush=True)

    return rows, (reps_small, reps_large)


def _by_config(df, column, fmt):
    """One line per configuration, one entry per method: mean (SE), median [min, max]."""
    for (model, part, n, p), g in df.groupby(['model', 'part', 'n', 'p'], sort=True):
        entries = []
        for method, h in g.groupby('method'):
            v = h[column].dropna()
            if v.empty:
                entries.append(f"{method}: n/a")
                continue
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            entries.append(f"{method}: {fmt % v.mean()}({fmt % se}) "
                           f"med {fmt % v.median()} "
                           f"[{fmt % v.min()},{fmt % v.max()}]")
        print(f"  {model} ({part})  n={n:<4d} p={p:<3d}  " + " | ".join(entries))


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    acc = df[df.block == 'accuracy']
    sca = df[df.block == 'scaling']

    # The thesis quotes the angle. Delta_m is printed after it, for cross-checking
    # against the published table only.
    print("\n=== [accuracy] mean principal angle (deg): mean (SE), median [min, max] ===")
    _by_config(acc, 'angle', '%.2f')
    print("\n=== [accuracy] Delta_m — cross-check against the published table only ===")
    _by_config(acc, 'delta_m', '%.3f')

    print("\n=== [scaling] model A part 1, n=500: mean principal angle (deg) ===")
    _by_config(sca, 'angle', '%.2f')

    # aggfunc is passed explicitly: pivot_table defaults to the mean, and on five
    # replicates with one outlier that is the wrong summary (Rule 5d).
    print("\n=== [scaling] seconds per fit, median over "
          f"{SCALING_REPS} replicates ===")
    print(sca.pivot_table(index='method', columns='p', values='seconds',
                          aggfunc='median').round(3).to_string())
    print("\n=== [scaling] seconds per fit, min and max over replicates ===")
    for agg in ('min', 'max'):
        print(f"-- {agg}")
        print(sca.pivot_table(index='method', columns='p', values='seconds',
                              aggfunc=agg).round(3).to_string())

    print("\n=== [accuracy] seconds per fit, median ===")
    print(acc.pivot_table(index='method', columns='n', values='seconds',
                          aggfunc='median').round(3).to_string())
    return df


def main():
    rows, (reps_small, reps_large) = run()
    df = summarise(rows)
    out = RESULTS / 'results_sheng_yin_2016.csv'
    stamp = write_csv(
        out, df, seeds=(BASE_SEED,),
        script='ch04_shengyin_2016_design.py',
        replicates=f"accuracy block: {reps_small} at n=100, {reps_large} at n=500; "
                   f"scaling block: {SCALING_REPS} at each p",
        n_perturb=N_PERTURB, n_restarts=N_RESTARTS, max_iter=MAX_ITER,
        scaling_block=f"model A part 1, n={SCALING_N}, p in "
                      f"{{{', '.join(str(v) for v in SCALING_P)}}}",
        methods='dCor-PP (this thesis), Sheng-Yin (SQP), SIR, OLS',
        design_source='wu2021mm pp.12-13, restating Sheng & Yin (2016)')
    print(f"\nWrote {out.name} ({len(df)} rows")


if __name__ == "__main__":
    main()
