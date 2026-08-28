"""step3_1d_grid.py — the k = 1 grid behind Chapter 5's recovery table and heatmap.

Writes ``results_1d.csv``: one row per (seed, latent distribution, signal strength,
dimension), reporting how well a single dVar direction recovers the true axis against
PCA, FastICA and a random direction.

Why this file exists
--------------------
``results_1d.csv`` was in the repository with **no generator** — the script that wrote
it (``step3_1d_grid.py``, this name) was lost, and ``make_ch5_figures.py`` says so in a
comment. Chapter 5's grid table, its heatmap and two figures therefore could not be
reproduced at all. Written 2026-08-11 as prerequisite P3 of
``thesis/REVISION_PLAN_2026-08-11.md``.

Agreement with the lost script
------------------------------
Checked against the old ``results_1d.csv`` at seed 42 over seven configurations, one
per latent distribution and one per dimension.

**The data pipeline is exactly reproduced.** ``MSS_PCA`` and ``MSS_FastICA`` depend on
the data and not on the search, and they agree to 13–15 significant digits at every
configuration checked. So the generator, the standardisation, the ``W_true`` draw and
the rng consumption order are right.

**The search is reproduced only to within restart noise.** ``angle_deg``, ``MSS_dVar``
and ``dvar_obj`` agree to 14 digits at two of the seven configurations and differ at the
other five — by 2e-3 to 2e-2 relative on the angle, and by 3e-7 to 1e-5 relative on the
objective. The cause is not a pipeline difference but a flat objective: at
(p=20, sigma=1, gaussian_mix), optimiser seeds 0, 1, 7, 42, 123, 2024 and 20251 at 10,
20 and 30 restarts give objective values spanning 0.942422–0.942452 and angles spanning
43.60°–44.49°, and the old file's 43.754° sits inside that spread. No seed reproduces
it. Which restart wins is arbitrary at the 1e-5 level of the objective, and about ±0.5°
of the reported angle is optimiser noise on top of sampling noise. This generator's
objective is equal or higher than the old file's at all seven configurations, so nothing
suggests it searches worse.

That is a reason to report a median and a range over seeds rather than one number, which
is what the rerun does.

Three columns are not comparable at all:

* ``MSS_random`` and ``dvar_random_mean`` are reproducible in *form* — a search over
  seeds 0–999 and every draw count up to 200 found each old target exactly, at 20 draws,
  but under two different and unrelated seeds (59 and 53). The old script's seeding of
  its random baselines is not recoverable. This file fixes an explicit convention,
  documented at ``_random_seed`` below. They are baselines, not results.
* ``restart_spread`` was **0.0 in all 61 rows of the old file, and that is not a
  measurement.** ``pp_dvar_sequential`` discarded the spread that
  ``optimise_one_direction`` computes, so no code path could have filled the column. The
  true value at (p=5, sigma=2, gaussian_mix) is 1.585e-04 (min 1.58904133819829, max
  1.58919979299742 over 20 restarts), and it reaches 9.8e-02 at (p=20, sigma=1) — small
  where the problem is easy, not small where it is not, and in no case a rounding of zero
  next to a file that stores sixteen digits in every other column. Any claim that the
  restart spread vanishes rests on a column that was never measured. Recorded in
  ``thesis/REVISION_LOG.md``; the spread is now plumbed out of the sequential path in
  ``dvar_optimizer.pp_dvar_sequential`` and measured here.

Runtime: about 4 minutes per seed on two cores, so ~20 minutes for the five seeds.
"""

# Thesis:   Chapter 5, tab:p2-grid
# Writes:   results/results_1d.csv
# Original: Dvar-PP/step3_1d_grid.py on the thesis branch.
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
import zlib
from pathlib import Path

import numpy as np


from dpp.unsupervised.data_generator import generate_data                   # noqa: E402
from dpp.unsupervised.dvar_optimizer import optimise_one_direction, make_index  # noqa: E402
from dpp.unsupervised.evaluation import (angle_deg, mss_principal, pca_directions,  # noqa: E402
                        fastica_directions, random_dvar_baseline,
                        random_mss_baseline)

OUT = RESULTS / "results_1d.csv"

#: Five data seeds. One replicate is not a claim — finding F2 of the revision plan is
#: a rule inferred from a single seed that four other seeds contradict.
SEEDS = (42, 7, 123, 2024, 5)

#: Six latent distributions. ``gaussian`` was added on 2026-08-23 at the supervisor's
#: request ("give Gaussian too") and is the control the grid never had: its distance
#: variance at unit variance is 0.634, exactly that of the noise it is buried in, so the
#: index has no shape difference to climb towards and whatever recovery remains is
#: carried by the variance lift sigma_signal^2 + 1 alone.
DISTRIBUTIONS = ("gaussian_mix", "t3", "uniform", "skewed", "chisq", "gaussian")
SIGMAS = (0.5, 1.0, 2.0, 4.0)
DIMENSIONS = (5, 20, 50)
N_MAIN = 200

#: One extra configuration at n = 500. The old file carried it as a 61st row beside the
#: 5 x 4 x 3 = 60 of the main grid; it is what the sample-size remark in Chapter 5
#: rests on.
EXTRA = ({"n": 500, "p": 20, "sigma": 2.0, "dist": "gaussian_mix"},)

N_STARTS = 20
MAX_ITER = 300
LR = 0.05
N_RAND = 20          # draws in both random baselines, as in the old file
INDEX_FUN = "plain"

COLUMNS = ["seed", "n", "p", "sigma", "dist", "angle_deg",
           "MSS_dVar", "MSS_PCA", "MSS_FastICA", "MSS_random",
           "dvar_obj", "dvar_random_mean", "restart_spread",
           "beats_PCA", "beats_FastICA", "beats_random"]


def _random_seed(data_seed: int, p: int, dist: str, offset: int) -> int:
    """Seed for a random baseline, derived from the configuration.

    Fixed convention, chosen because the old script's is unrecoverable: the baseline
    varies with the configuration (so it is not one lucky draw reused sixty times)
    and is reproducible from the row itself (so any row can be recomputed without the
    rest of the grid). ``offset`` separates the MSS baseline from the dVar baseline.

    Uses a CRC of the configuration string rather than ``hash()``: Python salts the
    hash of a string per process, so ``hash()`` here would silently give a different
    grid on every run — the precise failure this script exists to rule out.
    """
    key = f"{data_seed}|{p}|{dist}|{offset}".encode("utf-8")
    return zlib.crc32(key) % (2**31 - 1)


def run_one(seed: int, n: int, p: int, sigma: float, dist: str) -> dict:
    dat = generate_data(n=n, p=p, k=1, sigma_signal=sigma, dist=dist, seed=seed)
    X, W_true = dat["X"], dat["W_true"]

    res = optimise_one_direction(X, index_fun=INDEX_FUN, n_starts=N_STARTS,
                                 max_iter=MAX_ITER, lr=LR, seed=seed)
    W_hat = res["w"][:, None]
    lo, hi = res["spread"]

    mss_dvar = mss_principal(W_true, W_hat)
    mss_pca = mss_principal(W_true, pca_directions(X, 1))
    mss_ica = mss_principal(W_true, fastica_directions(X, 1))
    mss_rand = float(random_mss_baseline(
        W_true, n_rand=N_RAND, seed=_random_seed(seed, p, dist, 0)).mean())
    dvar_rand = float(random_dvar_baseline(
        X, n_rand=N_RAND, whitened=False, index_fun=INDEX_FUN,
        seed=_random_seed(seed, p, dist, 1)).mean())

    return {
        "seed": seed, "n": n, "p": p, "sigma": sigma, "dist": dist,
        "angle_deg": angle_deg(W_hat, W_true),
        "MSS_dVar": mss_dvar, "MSS_PCA": mss_pca, "MSS_FastICA": mss_ica,
        "MSS_random": mss_rand,
        "dvar_obj": float(res["obj"]),
        "dvar_random_mean": dvar_rand,
        # max - min across the restarts, not the (min, max) pair: what the text
        # calls the restart spread is the width.
        "restart_spread": float(hi - lo),
        "beats_PCA": bool(mss_dvar < mss_pca),
        "beats_FastICA": bool(mss_dvar < mss_ica),
        "beats_random": bool(mss_dvar < mss_rand),
    }


def main() -> None:
    configs = [{"n": N_MAIN, "p": p, "sigma": s, "dist": d}
               for d in DISTRIBUTIONS for s in SIGMAS for p in DIMENSIONS]
    configs += [dict(c) for c in EXTRA]

    total = len(configs) * len(SEEDS)
    print(f"{len(configs)} configurations x {len(SEEDS)} seeds = {total} runs "
          f"({len(DISTRIBUTIONS)} latent distributions x {len(SIGMAS)} signal "
          f"strengths x {len(DIMENSIONS)} dimensions, plus "
          f"{len(EXTRA)} at n = {EXTRA[0]['n']})", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for seed in SEEDS:
        for i, cfg in enumerate(configs, start=1):
            rows.append(run_one(seed=seed, **cfg))
            r = rows[-1]
            print(f"  seed {seed:5d}  {r['dist']:13s} sigma={r['sigma']:<4} "
                  f"p={r['p']:<3} angle={r['angle_deg']:6.2f}  "
                  f"MSS={r['MSS_dVar']:.4f}  spread={r['restart_spread']:.2e}  "
                  f"[{i}/{len(configs)}, {time.time() - t0:.0f}s]", flush=True)

    record = write_csv(OUT, rows, seeds=SEEDS, fieldnames=COLUMNS,
                                 script="ch05_single_direction_grid.py",
                                 n_starts=N_STARTS, max_iter=MAX_ITER,
                                 n_rand=N_RAND, index_fun=INDEX_FUN)
    print(f"\nWrote {OUT.name}: {len(rows)} rows", flush=True)

    spread = np.array([r["restart_spread"] for r in rows])
    print(f"restart spread across all runs: median {np.median(spread):.3e}, "
          f"max {spread.max():.3e}, exactly zero in "
          f"{int((spread == 0).sum())}/{len(spread)} runs", flush=True)


if __name__ == "__main__":
    main()
