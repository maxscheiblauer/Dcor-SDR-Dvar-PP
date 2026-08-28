"""step4_multi_dir.py — the k > 1 scenarios behind Chapter 5's multi-direction table.

Writes ``results_4.csv``: one row per (seed, scenario, strategy), comparing sequential
rank-1 deflation against joint optimisation on the Stiefel manifold.

Like ``step3_1d_grid.py`` this replaces a generator that was lost; see that file's
header for what could and could not be reconstructed. The scenarios, and the column
set, are read off the old ``results_4.csv``: five scenarios x two strategies, the last
of them with dependent latent factors.

The comparison this table supports is between strategies, so both run on identical
data. There is no penalty term anywhere on this side of the project: the multi-direction
constraint is held by rank-1 deflation in the sequential path and by the QR retraction
in the joint one, and ``pp_dvar_joint`` takes no penalty argument. The remedy for the
marginal objective's blindness to dependence *between* projections is the multivariate
index, not a penalty.

Runtime: about 90 s per seed per strategy pair, so ~15 minutes for five seeds.
"""

# Thesis:   Chapter 5, the multi-direction table
# Writes:   results/results_4.csv
# Original: Dvar-PP/step4_multi_dir.py on the thesis branch.
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
from pathlib import Path

import numpy as np


from dpp.unsupervised.data_generator import generate_data           # noqa: E402
from dpp.unsupervised.dvar_optimizer import pp_dvar                 # noqa: E402
from dpp.unsupervised.evaluation import (evaluate, inter_direction_dcor)  # noqa: E402

OUT = RESULTS / "results_4.csv"

SEEDS = (42, 7, 123, 2024, 5)
STRATEGIES = ("sequential", "joint")

#: The five scenarios, as tabulated in ``THESIS_NOTES.md`` §"Step 4" (the authoritative
#: record) and confirmed against ``report_experiments.md``. ``S1_dep`` is the one with
#: higher-order dependent latent factors, which is what separates "finds independent
#: directions" from "finds merely orthogonal ones"; it is not in the notes table
#: because it was added later, and its knobs are S1_basic's with the dependent
#: generator switched on.
SCENARIOS = {
    "S1_basic":     dict(n=200, p=50, k=2, sigma_signal=2.0, dist="gaussian_mix"),
    "S2_highdim":   dict(n=200, p=100, k=2, sigma_signal=3.0, dist="gaussian_mix"),
    "S4_weak":      dict(n=200, p=50, k=2, sigma_signal=0.8, dist="gaussian_mix"),
    "S5_multi_dir": dict(n=300, p=80, k=4, sigma_signal=2.0, dist="t3"),
    "S1_dep":       dict(n=200, p=50, k=2, sigma_signal=2.0, dist="gaussian_mix",
                         dependent_factors=True),
}

N_STARTS = 20
MAX_ITER = 300
LR = 0.05
INDEX_FUN = "plain"

COLUMNS = ["seed", "scenario", "strategy", "dependent", "MSS_dVar", "MSS_PCA",
           "MSS_FastICA", "ortho_defect", "inter_dcor_max", "per_dir_dvar",
           "restart_spread", "elapsed"]


def _spread_width(fit: dict) -> float:
    """Restart spread as a single width, whichever strategy produced the fit.

    ``pp_dvar_joint`` reports one ``(min, max)`` pair over restarts of the joint
    objective. ``pp_dvar_sequential`` reports one pair per direction (plumbed out on
    2026-08-11 — it used to discard them). The widest is reported, so the column
    answers "how far apart did the restarts land" in both cases.
    """
    spread = fit.get("spread")
    if spread is None:
        return float("nan")
    pairs = [spread] if isinstance(spread[0], (int, float)) else list(spread)
    return float(max(hi - lo for lo, hi in pairs))


def run_one(seed: int, name: str, cfg: dict, strategy: str) -> dict:
    dat = generate_data(seed=seed, **cfg)
    t0 = time.time()
    fit = pp_dvar(dat["X"], k=cfg["k"], index_fun=INDEX_FUN, strategy=strategy,
                  whiten=False, n_starts=N_STARTS, max_iter=MAX_ITER, lr=LR,
                  seed=seed)
    elapsed = time.time() - t0

    ev = evaluate(fit["W"], dat["X"], dat["W_true"], index_fun=INDEX_FUN,
                  whitened_eval=False)
    inter = inter_direction_dcor(dat["X"], fit["W"])
    k = cfg["k"]
    off_diag = [inter[i, j] for i in range(k) for j in range(i + 1, k)]

    return {
        "seed": seed, "scenario": name, "strategy": strategy,
        "dependent": bool(cfg.get("dependent_factors", False)),
        "MSS_dVar": ev["MSS_dVar"], "MSS_PCA": ev["MSS_PCA"],
        "MSS_FastICA": ev["MSS_FastICA"],
        "ortho_defect": ev["ortho_defect"],
        # Biased, unsquared V-statistic: evaluation.inter_direction_dcor calls
        # dcor.distance_correlation. Chapter 5 quotes none of these values and this
        # is deliberately left as it was found -- changing the estimator is a logic
        # change and belongs after the rerun, not during it.
        "inter_dcor_max": float(max(off_diag)) if off_diag else 0.0,
        "per_dir_dvar": np.asarray(ev["dvar_est"]).round(12).tolist(),
        "restart_spread": _spread_width(fit),
        "elapsed": elapsed,
    }


def main() -> None:
    total = len(SCENARIOS) * len(STRATEGIES) * len(SEEDS)
    print(f"{len(SCENARIOS)} scenarios x {len(STRATEGIES)} strategies x "
          f"{len(SEEDS)} seeds = {total} runs", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for seed in SEEDS:
        for name, cfg in SCENARIOS.items():
            for strategy in STRATEGIES:
                rows.append(run_one(seed, name, cfg, strategy))
                r = rows[-1]
                print(f"  seed {seed:5d}  {name:13s} {strategy:11s} "
                      f"MSS={r['MSS_dVar']:.4f} (PCA {r['MSS_PCA']:.4f})  "
                      f"ortho={r['ortho_defect']:.2e}  "
                      f"{r['elapsed']:5.1f}s  [{time.time() - t0:.0f}s]", flush=True)

    record = write_csv(OUT, rows, seeds=SEEDS, fieldnames=COLUMNS,
                                 script="ch05_multi_direction.py",
                                 n_starts=N_STARTS, max_iter=MAX_ITER,
                                 index_fun=INDEX_FUN)
    print(f"\nWrote {OUT.name}: {len(rows)} rows", flush=True)

    for strategy in STRATEGIES:
        mss = np.array([r["MSS_dVar"] for r in rows if r["strategy"] == strategy])
        sec = np.array([r["elapsed"] for r in rows if r["strategy"] == strategy])
        print(f"{strategy:11s} MSS median {np.median(mss):.4f} "
              f"[{mss.min():.4f}, {mss.max():.4f}]   "
              f"elapsed median {np.median(sec):.1f}s", flush=True)


if __name__ == "__main__":
    main()
