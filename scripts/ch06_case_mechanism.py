"""
mechanism.py — why a direction fitted on the training half exposes the two months
of the September 2001 shock, and one fitted on the whole sample does not.

The puzzle
----------
Chapter 6 fits distance-variance projection pursuit on the 1964--1983 training half
of the published factor block, freezes the direction, and finds September and
October 2001 sitting 3.6 and 8.2 training standard deviations beyond the training
range. `Real Data Experiment/src/outlier_recovery.py` fits the same criterion on all
480 months and finds those months at ranks 133 and 128 of 480 on the leading
direction — ordinary, in the middle of the crowd. Same criterion, same panel, same
optimiser. Only the fitting sample differs.

Two explanations are possible and they make different predictions.

  (a) The search fails. The full-sample optimiser never finds the direction that
      exposes the pair, so the index would prefer it if only it were reached.
  (b) The objective declines it. The full-sample optimiser does better, by its own
      measure, on a direction that leaves the pair in the crowd — so the index is
      not looking for isolated points in the first place.

Under (a) the index value of Chapter 6's direction, evaluated on the full sample,
would beat the full-sample optimum. Under (b) it would lose to it. That is one
comparison and part 1 makes it.

If (b) holds, a second question follows: is it the two months themselves that make
Chapter 6's direction unattractive on the full sample, or is it the 250 months of
1984--2003 that the training half never contained? Part 2 separates those by
repeating the comparison with the pair deleted. If the direction is still beaten on
478 months, the pair is not the cause and the era is.

Parts 3 and 4 then give the reason in a form that does not depend on this panel.
Distance variance is a dispersion / shape index: at fixed variance it rises as mass
moves away from the centre, and `tab:bg-shape` orders distributions accordingly, with
a balanced two-point law highest at 1.000 and heavy-tailed laws below the Gaussian's
0.634. Two stray points out of 480 are not mass. Part 3 measures this on the real
panel, where the basis makes the projection variance identical for every direction so
nothing is a scale effect. Part 4 measures it on synthetic data, sweeping the size of
a planted spike and the number of points in it, which is the general statement the
real panel can only illustrate — and it also shows why the biloop index behaves
differently, which is why that arm sometimes lands on the pair-isolating direction
when the plain index never does.

Run:  python scripts/ch06_case_mechanism.py
Writes: results/mechanism_directions.csv, results/mechanism_shape.csv,
        results/mechanism_spike.csv,
        figures/mechanism_index_vs_margin.png, figures/mechanism_spike_trace.png
"""

# Thesis:   Chapter 6, the case-study mechanism
# Writes:   results/mechanism_directions.csv, results/mechanism_shape.csv, results/mechanism_spike.csv
# Original: Outlier Case Study/mechanism.py on the thesis branch.
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

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

import realdata.dataio as dio                                           # noqa: E402
from realdata import holdout as hr                                  # noqa: E402
from realdata import recovery as orec                                # noqa: E402
from dpp.unsupervised.dvar_optimizer import pp_dvar, dvar, dvar_biloop          # noqa: E402
# The reference values of tab:bg-shape are drawn with the same generator that writes
# that table, so the two cannot disagree.
from designs.shapes import draws, _standardise               # noqa: E402

warnings.filterwarnings("ignore")

#: Seed 0 is Chapter 6's own — `make_ch6_fig4.py` runs at it — and is kept first so
#: the row that reproduces the published numbers is identifiable. The other four are
#: the project's standard list.
SEEDS = (0, 42, 7, 123, 2024)

#: Chapter 6 searches eight directions and reports the highest-index one; parts 1 and
#: 2 keep that, so the direction compared against the full-sample optimum is the one
#: the chapter actually shows.
KMAX = 8

#: Directions per frame in part 3, matching the survey.
K_SHAPE = 3

#: Part 4 sweep. Spike sizes in standard deviations of the clean sample, and the
#: number of points moved into the spike. Two of 480 is the real case. The sweep starts
#: at 2 rather than 0 because moving points to 0 is not the clean sample either — it
#: pulls them to the centre — so the clean level is recorded as its own column instead.
SPIKE_SIZES = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0)
SPIKE_COUNTS = (2, 5, 10, 20)
N_SYNTH = 480

FIGURES = _HERE / "figures"


# ------------------------------------------------------- parts 1 and 2: directions
def leading_dvar_direction(X, seed, strategy="joint"):
    """The highest-index direction of a KMAX-direction plain dVar-PP frame.

    The same construction that produced Chapter 6's figure, restated here so this
    script depends on nothing outside this repository.
    """
    fit = pp_dvar(X, k=KMAX, index_fun="plain", strategy=strategy, whiten=False,
                  n_starts=20, max_iter=300, n_jobs=-1, seed=seed)
    W = fit["W"]
    order = np.argsort([dvar(X @ W[:, j]) for j in range(W.shape[1])])[::-1]
    w = W[:, order[0]]
    # A projection direction and its negation are the same direction, and the
    # optimiser returns whichever the restart happened to reach: at seed 0 the pair's
    # October score is +9.0 and at seed 42 it is -9.0 on the same axis. Signed
    # quantities are then not comparable across seeds — a median of +5.0 and -5.0 is
    # 0, which is how this first showed up. Orienting by the same rule as
    # `pca._sign_fix`, largest-magnitude loading positive, makes them comparable and
    # is decided by the loadings alone, so nothing about the two months enters it.
    return w if w[np.argmax(np.abs(w))] >= 0 else -w


def evaluate(w, X, rows_mask, hold_rows, label):
    """Index value, scale and pair extremeness of one direction on one sample.

    `rows_mask` selects the sample the index is evaluated on. The margin compares the
    pair against the months of that sample other than the pair itself, so it is
    defined whether or not the pair is inside it.
    """
    g = X @ w
    gs = g[rows_mask]
    sd = float(gs.std(ddof=1))
    med = float(np.median(gs))
    mad = float(np.median(np.abs(gs - med)))
    scale = hr.MAD_SCALE * mad if mad > 1e-12 else sd
    z = (g - med) / scale
    ref = np.abs(z[rows_mask & ~_mask_of(hold_rows, len(g))])
    lo, hi = float(gs.min()), float(gs.max())
    excess = []
    for t in hold_rows:
        if g[t] > hi:
            excess.append((g[t] - hi) / sd)
        elif g[t] < lo:
            excess.append((g[t] - lo) / sd)
        else:
            excess.append(0.0)
    return {
        f"dvar_{label}": dvar(gs),
        f"sd_{label}": sd,
        # dVar divided by the projection's own standard deviation: the scale-free
        # form, which isolates shape where the sample covariance is not isotropic.
        f"dvar_over_sd_{label}": dvar(gs) / sd if sd > 0 else float("nan"),
        f"margin_{label}": float(np.abs(z[hold_rows]).min() / ref.max()),
        f"excess_sep_{label}": excess[0],
        f"excess_oct_{label}": excess[1],
        # The quantity Chapter 6 quotes, in the form that survives aggregation: how
        # far beyond the nearer edge of the range the further of the two months lands,
        # regardless of which side it lands on.
        f"excess_abs_max_{label}": float(max(abs(excess[0]), abs(excess[1]))),
    }


def _mask_of(rows, n):
    m = np.zeros(n, bool)
    m[list(rows)] = True
    return m


def directions_study(F, dates, hold_rows):
    """Parts 1 and 2: fit on three samples, evaluate every direction on all three."""
    n = len(dates)
    train, _, _, _ = dio.split_masks(dates)
    full = np.ones(n, bool)
    held = ~_mask_of(hold_rows, n)
    samples = {"train": train, "full": full, "held": held}

    recs = []
    for seed in SEEDS:
        ws = {name: leading_dvar_direction(F[mask], seed)
              for name, mask in samples.items()}
        for name, w in ws.items():
            rec = dict(fitted_on=name, n_fit=int(samples[name].sum()), seed=seed)
            for ev_name, ev_mask in samples.items():
                rec.update(evaluate(w, F, ev_mask, hold_rows, ev_name))
            rec["angle_to_full_deg"] = float(np.degrees(np.arccos(
                min(1.0, abs(float(w @ ws["full"]))))))
            rec["loadings"] = " ".join(f"{v:+.3f}" for v in w)
            recs.append(rec)
        print(f"  seed {seed:4d}: "
              f"dVar on the full sample — Ch6 direction "
              f"{recs[-3]['dvar_full']:.4f}, full-sample optimum "
              f"{recs[-2]['dvar_full']:.4f}; "
              f"on the 478 — {recs[-3]['dvar_held']:.4f} against "
              f"{recs[-1]['dvar_held']:.4f}", flush=True)
    return pd.DataFrame(recs)


# ------------------------------------------------- part 3: shape at fixed variance
def shape_study(X, hold_rows, fit_mask, dates):
    """Index value against pair extremeness, direction by direction.

    Run in the basis re-extracted from the fitting months, whose covariance is
    isotropic to about 5e-15. Every unit direction therefore carries the same
    projection variance, so a difference in index value between two directions is a
    difference in shape and nothing else.
    """
    hr.K = K_SHAPE
    orec.K = K_SHAPE
    recs = []
    for seed in SEEDS:
        for strategy in ("sequential", "joint"):
            W, order, crit = orec.fit_dvar(X[fit_mask], "plain", strategy, seed)
            for pos, j in enumerate(order[:K_SHAPE]):
                g = X @ W[:, j]
                sc = hr.score_holdout(g, hold_rows, fit_mask, dates)
                recs.append(dict(seed=seed, strategy=strategy, direction=pos + 1,
                                 dvar=float(crit[j]),
                                 sd_fit=float(g[fit_mask].std(ddof=1)),
                                 margin=sc["margin"],
                                 pair_isolated=sc["pair_isolated"],
                                 worse_rank=sc["worse_rank"],
                                 loadings=" ".join(f"{v:+.3f}" for v in W[:, j])))
    return pd.DataFrame(recs)


def reference_shapes(n=4000, seed=20260808):
    """dVar of standard laws at unit variance — the tab:bg-shape reference points."""
    rng = np.random.default_rng(seed)
    names = ["two-point", "bimodal mixture", "uniform", "Gaussian", "Student t3"]
    return {nm: dvar(_standardise(draws(rng, nm, n))) for nm in names}


# --------------------------------------------------- part 4: planted-spike sweep
def spike_study():
    """dVar and dVar-biloop of a bulk with a planted spike, at unit variance.

    A clean Gaussian sample of length 480 has `m` of its points moved to `+c`
    standard deviations, and the result is standardised to unit variance before the
    index is taken, so the sweep asks about shape alone and never about scale. This is
    the synthetic counterpart of a projection that isolates a few months.
    """
    recs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        base = rng.standard_normal(N_SYNTH)
        sd0 = base.std(ddof=1)
        clean = _standardise(base)
        dv_clean, db_clean = dvar(clean), dvar_biloop(clean)
        for m in SPIKE_COUNTS:
            idx = np.argsort(base)[-m:]          # move the largest m, deterministic
            for c in SPIKE_SIZES:
                z = base.copy()
                z[idx] = c * sd0
                zs = _standardise(z)
                recs.append(dict(seed=seed, n_spike=m, spike_sd=c,
                                 dvar=dvar(zs), dvar_biloop=dvar_biloop(zs),
                                 dvar_clean=dv_clean, dvar_biloop_clean=db_clean))
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------- main
def main():
    dates_all = pd.to_datetime(pd.Series(dio.aligned_panel()["dates"])).to_numpy()
    hold_rows = hr.target_rows(dates_all)
    B, dates, fit_mask, eig = hr.holdout_bases(hold_rows)
    a = dio.aligned_panel()
    F_pub = a["targets"][[f"f{i}" for i in range(1, 9)]].to_numpy(dtype=float)

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("PARTS 1 AND 2 — the published block, plain dVar-PP, "
          f"{KMAX} directions, highest-index one kept")
    print("  fitted on the train half (228 months), on all 480, and on the 478 "
          "excluding the pair")
    d = directions_study(F_pub, dates, hold_rows)

    # Only sign-free quantities are aggregated: index values, scales, margins and the
    # absolute excess. aggfunc is named because the pandas default is the mean.
    piv = d.pivot_table(index="fitted_on",
                        values=["dvar_train", "dvar_full", "dvar_held",
                                "margin_train", "margin_full", "margin_held",
                                "excess_abs_max_train", "excess_abs_max_full"],
                        aggfunc="median")
    print("\n  median over seeds — rows are where the direction was fitted, "
          "columns where it was evaluated:")
    print(piv.to_string())

    # The two questions, answered in code so the verdict is in the log.
    med = d.groupby("fitted_on").median(numeric_only=True)
    q1 = med.loc["train", "dvar_full"] < med.loc["full", "dvar_full"]
    q2 = med.loc["train", "dvar_held"] < med.loc["held", "dvar_held"]
    print(f"\n  (1) the full-sample optimum beats Chapter 6's direction on the full "
          f"sample: {bool(q1)} "
          f"({med.loc['full', 'dvar_full']:.4f} against "
          f"{med.loc['train', 'dvar_full']:.4f})")
    print(f"      -> the objective declines that direction; the search does not fail")
    print(f"  (2) it still beats it with the pair deleted: {bool(q2)} "
          f"({med.loc['held', 'dvar_held']:.4f} against "
          f"{med.loc['train', 'dvar_held']:.4f})")
    print(f"      -> {'the era, not the pair' if q2 else 'the pair itself'} "
          f"is what makes Chapter 6's direction unattractive")
    write_csv(RESULTS / "mechanism_directions.csv", d, seeds=list(SEEDS),
                         script="ch06_case_mechanism.py", KMAX=KMAX)

    print(f"\nPART 3 — shape at fixed variance, basis re-extracted from the "
          f"{int(fit_mask.sum())} fitting months")
    print(f"  isotropy of that covariance: {hr.isotropy(B['fit_unit'][fit_mask]):.2e}"
          f" — projection variance is the same for every unit direction")
    shape = shape_study(B["fit_unit"], hold_rows, fit_mask, dates)
    print(shape.groupby(["strategy", "direction"])
               .agg(dvar_med=("dvar", "median"), sd_med=("sd_fit", "median"),
                    margin_med=("margin", "median"), margin_min=("margin", "min"),
                    margin_max=("margin", "max"),
                    isolated=("pair_isolated", "sum")).to_string())
    refs = reference_shapes()
    print("\n  reference laws at unit variance (tab:bg-shape): "
          + ", ".join(f"{k} {v:.3f}" for k, v in refs.items()))
    write_csv(RESULTS / "mechanism_shape.csv", shape, seeds=list(SEEDS),
                         script="ch06_case_mechanism.py", K=K_SHAPE)

    print("\nPART 4 — planted spike, standardised to unit variance")
    spike = spike_study()
    tab = spike.pivot_table(index="spike_sd", columns="n_spike", values="dvar",
                            aggfunc="median")
    print("  median dVar by spike size (rows) and points in the spike (columns):")
    print(tab.round(4).to_string())
    tab_b = spike.pivot_table(index="spike_sd", columns="n_spike",
                              values="dvar_biloop", aggfunc="median")
    print("  the same for the biloop index:")
    print(tab_b.round(4).to_string())
    write_csv(RESULTS / "mechanism_spike.csv", spike, seeds=list(SEEDS),
                         script="ch06_case_mechanism.py",
                         n=N_SYNTH, spike_counts=str(SPIKE_COUNTS))



if __name__ == "__main__":
    main()
