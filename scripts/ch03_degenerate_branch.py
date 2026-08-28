"""degenerate_share.py — how often the dCor gradient lands at a degenerate point.

Section 3.2.3 states the share of gradient calls for which the unbiased statistics
satisfy `s_xx, s_xy, s_yy <= 0`, so that dCor² and its derivative are undefined and the
optimiser falls back to the surrogate direction `∂s_xy/∂β`. That share is the evidence
for specifying the degenerate branch at all rather than leaving it to the restarts, so
it needs a results file of its own.

It instruments `dcor_optimizer._dcor_and_grad` over the configurations Chapter 4
actually runs — the k = 1 grid of `tab:p1-grid` and the multi-direction configurations
of `tab:p1-joint`, sequential and joint — and records, per (configuration, seed), the
number of calls and how many were degenerate.

Writes `results_degenerate_share.csv`. About 6 minutes at ten seeds.
"""

# Thesis:   Chapter 3, §3.2.3
# Writes:   results/results_degenerate_share.csv
# Original: PP_Dcor/degenerate_share.py on the thesis branch.
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

_HERE = Path(__file__).resolve().parent
warnings.filterwarnings("ignore")

import dpp.supervised.dcor_optimizer as dco                             # noqa: E402
from dpp.supervised.data_generator import data_generator                # noqa: E402
from dpp.supervised.dcor_optimizer import optimize_dcor                 # noqa: E402
from dpp.supervised.pp_helpers import seq_pp                            # noqa: E402

SEEDS = (42, 7, 123, 2024, 5, 2026, 17, 99, 777, 31337)

#: k = 1 configurations, a spread of the tab:p1-grid design: the two symmetric
#: responses where dependence is hardest to detect, one monotone response, and the
#: pathological one, at the noise levels where the objective is weakest.
SINGLE = [("square", 0.0, 5), ("square", 2.0, 20), ("abs", 0.0, 5),
          ("sine", 0.0, 5), ("sine", 1.0, 20), ("cubic", 2.0, 20)]

#: The multi-direction configurations of tab:p1-joint.
MULTI = [(2, 5, "product", 0.0, 200), (2, 5, "sum_squares", 0.0, 200),
         (2, 10, "sum_squares", 0.0, 300), (2, 5, "product", 0.5, 200)]

_orig = dco._dcor_and_grad
_count = {"calls": 0, "degenerate": 0}


def _counting(beta, X, B, s_yy):
    _count["calls"] += 1
    val, grad = _orig(beta, X, B, s_yy)
    # The clamp returns exactly 0.0 for the index at a degenerate point; a genuine
    # projection with zero dependence to floating-point precision does not occur.
    if val <= 0.0:
        _count["degenerate"] += 1
    return val, grad


def _reset():
    _count.update(calls=0, degenerate=0)


def _row(label, kind, seed):
    calls = _count["calls"]
    deg = _count["degenerate"]
    return dict(config=label, kind=kind, seed=seed, calls=calls, degenerate=deg,
                share=(deg / calls if calls else float("nan")))


def main() -> None:
    dco._dcor_and_grad = _counting
    import dpp.supervised.joint_optimization as jo   # imports _dcor_and_grad by name
    jo._dcor_and_grad = _counting

    rows = []
    for seed in SEEDS:
        for nl, noise, p in SINGLE:
            X, Y, _ = data_generator(200, p, 1, nl, noise, seed)
            _reset()
            optimize_dcor(X, Y, init_method="random", optimizer="gradient_ascent",
                          n_restarts=3, seed=seed, max_iter=100)
            rows.append(_row(f"k1_{nl}_noise{noise}_p{p}", "single", seed))

        for k, p, nl, noise, n in MULTI:
            X, Y, _ = data_generator(n, p, k, nl, noise, seed)
            label = f"k{k}_p{p}_{nl}" + ("_noisy" if noise else "")
            _reset()
            seq_pp(X, Y, k, deflation="X_deflation", n_restarts=4, max_iter=100, seed=0)
            rows.append(_row(label, "sequential", seed))
            _reset()
            jo.joint_optimize(X, Y, k, lam=0.0, n_restarts=4, max_iter=100, seed=0)
            rows.append(_row(label, "joint", seed))
        print(f"  seed {seed} done", flush=True)

    record = write_csv(RESULTS / "results_degenerate_share.csv", rows,
                                  seeds=SEEDS,
                                  script="ch03_degenerate_branch.py")
    share = np.array([r["share"] for r in rows])
    print(f"\nwrote results_degenerate_share.csv ({len(rows)} rows)")
    print(f"share of gradient calls at a degenerate point: "
          f"median {np.median(share):.1%}, min {share.min():.1%}, "
          f"max {share.max():.1%}")
    for kind in ("single", "sequential", "joint"):
        s = np.array([r["share"] for r in rows if r["kind"] == kind])
        print(f"  {kind:11s} median {np.median(s):.1%}  "
              f"[{s.min():.1%}, {s.max():.1%}]")


if __name__ == "__main__":
    main()
