"""Distance variance of unit-variance distributions (Table 2.1 and Chapter 1).

At fixed variance the distance variance still varies with the shape of the
distribution.  The ordering it produces is not distance from Gaussianity: it
tracks how far the mass sits from the centre.  This script produces the values
quoted in Chapter 1 and in Table 2.1.

Three things are reported alongside the index, all asked for in the supervisor
round of 2026-08-25:

* the sample **excess kurtosis** of the same draws, so that the claim that the
  distance variance moves opposite to peakedness can be checked rather than
  asserted (note C2-5);
* two **multimodal** rows, three and four equally weighted components, beside
  the two-component one that was already there (note C2-4);
* a **separation sweep** for the bimodal mixture: the component means move apart
  while the sample is standardised back to unit variance after every draw, so
  the sweep isolates shape from scale (note C2-4).

Everything is written to ``results/results_shape.csv``.

Run:  python scripts/ch05_shape_table.py
"""
from __future__ import annotations

from csvout import RESULTS, write_csv

import sys
from pathlib import Path

import numpy as np



from dpp.unsupervised.dvar_optimizer import dvar  # noqa: E402

# dvar() forms the dense n x n distance matrix, so n is bounded by memory
# rather than by time; averaging over independent draws recovers the precision.
N = 4_000
SEED = 20260808
REPS = 10            # repeat to report the spread across independent draws

#: Distance between the two component means, in units of the within-component
#: standard deviation.  4.0 reproduces the ``bimodal mixture`` row of the table.
SEPARATIONS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


def _standardise(z):
    return (z - z.mean()) / z.std(ddof=1)


def _excess_kurtosis(z):
    """Sample excess kurtosis of an already standardised vector."""
    return float(np.mean(z ** 4) - 3.0)


def _mixture(rng, n, sep, n_components):
    """Equally weighted unit-variance components with means sep apart."""
    offsets = np.arange(n_components, dtype=float)
    offsets -= offsets.mean()
    means = offsets * sep
    return rng.choice(means, size=n) + rng.standard_normal(n)


def draws(rng, name, n):
    """One sample of length n, before standardisation."""
    if name == "two-point":
        return rng.choice([-1.0, 1.0], size=n)
    if name == "bimodal mixture":                       # gaussian_mix in Ch. 5
        return _mixture(rng, n, sep=4.0, n_components=2)
    if name == "trimodal mixture":
        return _mixture(rng, n, sep=4.0, n_components=3)
    if name == "four-component mixture":
        return _mixture(rng, n, sep=4.0, n_components=4)
    if name == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n)
    if name == "Gaussian":
        return rng.standard_normal(n)
    if name == "centred exponential":                   # skewed in Ch. 5
        return rng.standard_exponential(n) - 1.0
    if name == "centred chi-squared(3)":                # chisq in Ch. 5
        return rng.chisquare(df=3, size=n) - 3.0
    if name == "Laplace":
        return rng.laplace(size=n)
    if name == "Student t3":
        return rng.standard_t(df=3, size=n)
    if name == "lognormal":
        return rng.lognormal(sigma=1.0, size=n)
    raise ValueError(name)


NAMES = [
    "two-point", "bimodal mixture", "trimodal mixture", "four-component mixture",
    "uniform", "centred chi-squared(3)", "Gaussian", "centred exponential",
    "Laplace", "Student t3", "lognormal",
]


def _measure(rng, sampler):
    """dVar and excess kurtosis of REPS standardised draws."""
    dv, ek = [], []
    for _ in range(REPS):
        z = _standardise(sampler(rng))
        dv.append(dvar(z))
        ek.append(_excess_kurtosis(z))
    return (float(np.mean(dv)), float(np.std(dv)),
            float(np.mean(ek)), float(np.std(ek)))


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    print(f"dVar at unit variance   (n = {N:,}, {REPS} independent draws)\n")
    print(f"{'distribution':26s} {'dVar':>7s} {'sd':>7s} {'ex.kurt':>9s}")
    print("-" * 52)
    named = []
    for name in NAMES:
        m, s, k, ks = _measure(rng, lambda r, nm=name: draws(r, nm, N))
        named.append((name, m, s, k, ks))
        rows.append({"family": "distribution", "distribution": name,
                     "separation": "", "dvar": round(m, 4),
                     "dvar_sd": round(s, 4), "excess_kurtosis": round(k, 4),
                     "excess_kurtosis_sd": round(ks, 4)})
    for name, m, s, k, _ in sorted(named, key=lambda r: -r[1]):
        print(f"{name:26s} {m:7.3f} {s:7.4f} {k:9.3f}")

    # Is the ordering by dVar the reverse of the ordering by excess kurtosis?
    by_dvar = [n for n, *_ in sorted(named, key=lambda r: -r[1])]
    by_kurt = [n for n, *_ in sorted(named, key=lambda r: r[3])]
    print(f"\ndVar order is the reverse of the excess-kurtosis order: "
          f"{by_dvar == by_kurt}")
    if by_dvar != by_kurt:
        print(f"  by dVar (desc): {by_dvar}")
        print(f"  by kurtosis (asc): {by_kurt}")

    print(f"\nbimodal mixture, modes moved apart, unit variance after every draw")
    print(f"{'separation':>10s} {'dVar':>7s} {'sd':>7s} {'ex.kurt':>9s}")
    print("-" * 36)
    for sep in SEPARATIONS:
        m, s, k, ks = _measure(
            rng, lambda r, d=sep: _mixture(r, N, sep=d, n_components=2))
        print(f"{sep:10.1f} {m:7.3f} {s:7.4f} {k:9.3f}")
        rows.append({"family": "bimodal_separation", "distribution":
                     "bimodal mixture", "separation": sep, "dvar": round(m, 4),
                     "dvar_sd": round(s, 4), "excess_kurtosis": round(k, 4),
                     "excess_kurtosis_sd": round(ks, 4)})

    write_csv(RESULTS / "results_shape.csv", rows, seeds=(SEED,))
    print("\nwrote results_shape.csv")


if __name__ == "__main__":
    main()
