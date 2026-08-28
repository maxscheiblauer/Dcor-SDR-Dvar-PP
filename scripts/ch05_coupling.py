"""coupling_check.py — how much dependence the heteroscedastic coupling actually carries.

Section 5.2 of the thesis states that the coupling $z_j = |z_1|\\varepsilon_j$ leaves the
latent columns uncorrelated while giving them strong higher-order dependence, and it
quotes a squared Pearson correlation against a squared distance correlation to say how
strong. That pair of numbers had no generating script: the seventh pass of 2026-08-12
computed it by hand after finding the previous pair wrong under Rule 3 (the old $0.15$
came from ``dcor.distance_correlation``, which is biased *and* unsquared, so it was
being compared against a squared quantity). Under Rule 5c a number without a stamped
file is not quotable, which left it the one such number in Chapter 5. Written
2026-08-23 to close that.

What is measured, per seed: the two latent columns of the dependent construction at
$n = 2000$, $k = 2$ with bimodal factors, the configuration the independence study of
Section 5.5 uses. For those columns,

* ``pearson_sq``  the squared Pearson correlation, which the coupling drives to zero
  because $\\mathbb{E}[\\varepsilon_j] = 0$ makes $z_j$ conditionally mean-zero given
  $z_1$;
* ``dcor_sq``     the squared distance correlation, which does not vanish, because the
  spread of $z_j$ is a function of $|z_1|$.

Both are reported for the independent construction as well, so that the contrast is
against a measured baseline rather than against zero.

Estimator, and why this file crosses to the supervised side for it: ``dcor_u`` from
``PP_Dcor/dcor_optimizer.py`` is the U-statistic and returns dCor squared already
(Rule 3). ``Dvar-PP/evaluation.py`` has an ``inter_direction_dcor`` helper, but it calls
``dcor.distance_correlation``, the biased V-statistic on the unsquared scale, and its
values must not be mixed with a squared one.

Runtime: about a minute. The $O(n^2)$ distance matrices at $n = 2000$ are the cost.

Run:  python scripts/ch05_coupling.py     ->  results/results_coupling.csv
"""

# Thesis:   Chapter 5, §5.2
# Writes:   results/results_coupling.csv
# Original: Dvar-PP/coupling_check.py on the thesis branch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sys
from pathlib import Path

import numpy as np

#: Appended, not inserted: ``PP_Dcor`` has a ``data_generator`` of its own, and putting
#: it ahead of this directory shadows the one this script needs.

from dpp.unsupervised.data_generator import (_draw_factor_columns,          # noqa: E402
                            _make_dependent_factors)
from dpp.supervised.dcor_optimizer import dcor_u                          # noqa: E402

OUT = RESULTS / "results_coupling.csv"

SEEDS = (42, 7, 123, 2024, 5)

#: The configuration Section 5.2 quotes. n is larger than the 200 of the recovery
#: experiments because this measures a property of the construction, not of a search,
#: and the estimator is the thing being pinned down.
N = 2000
K = 2
DIST = "gaussian_mix"

COLUMNS = ["seed", "n", "k", "dist", "factors", "pearson_sq", "dcor_sq", "ratio"]


def one_run(seed: int, dependent: bool) -> dict:
    rng = np.random.default_rng(seed)
    Z = (_make_dependent_factors(rng, N, K, DIST) if dependent
         else _draw_factor_columns(rng, N, K, DIST))
    z1, z2 = Z[:, 0], Z[:, 1]

    pearson_sq = float(np.corrcoef(z1, z2)[0, 1] ** 2)
    dcor_sq = float(dcor_u(z1, z2))
    return {
        "seed": seed, "n": N, "k": K, "dist": DIST,
        "factors": "dependent" if dependent else "independent",
        "pearson_sq": pearson_sq,
        "dcor_sq": dcor_sq,
        "ratio": dcor_sq / pearson_sq if pearson_sq > 0 else float("inf"),
    }


def main() -> None:
    rows = [one_run(s, dep) for s in SEEDS for dep in (False, True)]
    for r in rows:
        print(f"  seed {r['seed']:5d}  {r['factors']:11s}  "
              f"Pearson^2 {r['pearson_sq']:.2e}   dCor^2 {r['dcor_sq']:.4f}",
              flush=True)

    print("\nmedian [min, max] over the seeds")
    for factors in ("independent", "dependent"):
        sub = [r for r in rows if r["factors"] == factors]
        ps = np.array([r["pearson_sq"] for r in sub])
        dc = np.array([r["dcor_sq"] for r in sub])
        print(f"  {factors:11s}  Pearson^2 {np.median(ps):.2e} "
              f"[{ps.min():.2e}, {ps.max():.2e}]   "
              f"dCor^2 {np.median(dc):.4f} [{dc.min():.4f}, {dc.max():.4f}]")
        if np.median(ps) > 0:
            print(f"               ratio of medians "
                  f"{np.median(dc) / np.median(ps):.0f}x")

    record = write_csv(OUT, rows, seeds=SEEDS, fieldnames=COLUMNS,
                                  script="ch05_coupling.py",
                                  n=N, k=K, dist=DIST, estimator="dcor_u (U-statistic, squared)")
    print(f"\nWrote {OUT.name}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
