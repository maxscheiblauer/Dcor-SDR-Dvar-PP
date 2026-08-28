"""
Step 3 — k=1 experiments, sequential, compact grid.
"""

# Thesis:   Chapter 4, fig:p1-heatmaps and fig:p1-samplesize
# Writes:   results/results_1d_baselines.csv
# Original: PP_Dcor/step3_experiments.py on the thesis branch.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import RESULTS, pin_blas_threads, write_csv   # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import sys, time, itertools
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dpp.supervised.data_generator import data_generator
from dpp.supervised.dcor_optimizer import optimize_dcor
from dpp.supervised.evaluation import evaluate

from pathlib import Path

NONLINEARITIES = ['square', 'sine', 'abs', 'cubic', 'tanh']
NOISE_LEVELS   = [0.0, 0.5, 1.0, 2.0]
SAMPLE_SIZES   = [200, 500, 1000]   # populated below via sampling
DIMENSIONS     = [2, 5, 20]
N_RESTARTS     = 3
MAX_ITER       = 100

#: Five data seeds. Until 2026-08-11 this script ran the single seed 42, and every
#: number in tab:p1-grid was one replicate. Finding F2 of
#: thesis/REVISION_PLAN_2026-08-11.md is what that costs: a 36.81° at seed 42 against
#: 4.92°-8.76° at four others became a stated rule. Figures and the report now show the
#: median over these seeds; the CSV keeps every replicate.
SEEDS = (42, 7, 123, 2024, 5)

# For the main heatmap grid, use n=200 (fast); add n=500,1000 for scaling plot only
GRID_N = 200

FIELDS = ('dcor_val', 'angle', 'dcor_found', 'dcor_ols', 'angle_ols',
          'angle_sir', 'dcor_sir', 'angle_save', 'dcor_save', 'spread')


def run_one(nl, noise, n, p, seed):
    X, Y, W_true = data_generator(n, p, 1, nl, noise, seed)
    beta, dcor_val, info = optimize_dcor(X, Y, init_method='random',
                                         optimizer='gradient_ascent',
                                         n_restarts=N_RESTARTS, seed=seed,
                                         max_iter=MAX_ITER)
    m = evaluate(beta, W_true, X, Y)
    return {
        'dcor_val':   dcor_val,
        'angle':      m['angle_deg'],
        'beats_ols':  m['beats_ols'],
        'dcor_found': m['dcor_found'],
        'dcor_ols':   m['dcor_ols'],
        'angle_ols':  m['angle_ols'],
        'angle_sir':  m['angle_sir'],
        'dcor_sir':   m['dcor_sir'],
        'angle_save': m['angle_save'],
        'dcor_save':  m['dcor_save'],
        'spread':     info['val_spread'],
    }


per_seed: dict = {}          # (nl, noise, n, p) -> {seed: metrics}
t_start = time.time()
configs = list(itertools.product(NONLINEARITIES, NOISE_LEVELS, [GRID_N], DIMENSIONS))
scaling = [(nl, 0.5, n, 5) for nl in NONLINEARITIES for n in SAMPLE_SIZES]
for cfg in scaling:
    if cfg not in configs:
        configs.append(cfg)

print(f"{len(configs)} configurations x {len(SEEDS)} seeds = "
      f"{len(configs) * len(SEEDS)} runs "
      f"({len(NONLINEARITIES)} nonlinearities x {len(NOISE_LEVELS)} noise levels x "
      f"{len(DIMENSIONS)} dimensions at n={GRID_N}, plus the sample-size sweep at "
      f"p=5, noise=0.5)", flush=True)

for seed in SEEDS:
    for i, (nl, noise, n, p) in enumerate(configs):
        per_seed.setdefault((nl, noise, n, p), {})[seed] = \
            run_one(nl, noise, n, p, seed)
        if (i + 1) % 20 == 0:
            print(f"  seed {seed}: {i+1}/{len(configs)}  "
                  f"({time.time()-t_start:.0f}s)", flush=True)
    print(f"  seed {seed} done ({time.time()-t_start:.0f}s)", flush=True)

# ── Median over seeds ────────────────────────────────────────────────────────
# `results` keeps the shape the figure and report code below already expects, so a
# single number in a figure is now a median of five replicates rather than one draw.
# `spread_of_medians` carries the seed range, which is what makes a near-tie visible.
results = {}
for key, by_seed in per_seed.items():
    agg = {f: float(np.median([by_seed[s][f] for s in SEEDS])) for f in FIELDS}
    agg['beats_ols'] = bool(np.median([1.0 if by_seed[s]['beats_ols'] else 0.0
                                       for s in SEEDS]) >= 0.5)
    agg['angle_min'] = float(min(by_seed[s]['angle'] for s in SEEDS))
    agg['angle_max'] = float(max(by_seed[s]['angle'] for s in SEEDS))
    results[key] = agg

print(f"Done. Total: {time.time()-t_start:.1f}s", flush=True)

print(f"\nmedian over {len(SEEDS)} seeds, [min, max] on the angle")
print(f"{'nl':12s} {'noise':6s} {'n':6s} {'p':4s} | {'dCor^2':7s} "
      f"{'angle':>8s} {'[min, max]':>16s} {'>OLS':6s}")
print("-" * 76)
for (nl, noise, n, p), v in sorted(results.items()):
    print(f"{nl:12s} {noise:6.1f} {n:6d} {p:4d} | "
          f"{v['dcor_val']:7.4f} {v['angle']:8.2f} "
          f"  [{v['angle_min']:6.2f},{v['angle_max']:7.2f}] "
          f"{'Y' if v['beats_ols'] else 'N':6s}")
sys.stdout.flush()

# ── Archive every replicate to CSV (thesis cross-reference) ──────────────────
# One row per configuration *and seed*: the aggregate belongs in the table, the
# replicates belong in the record.
rows = []
for (nl, noise, n, p), by_seed in sorted(per_seed.items()):
    for seed in SEEDS:
        v = by_seed[seed]
        rows.append({
            'seed': seed, 'nonlinearity': nl, 'noise': noise, 'n': n, 'p': p,
            'angle_dcor': round(v['angle'], 4),
            'angle_ols': round(v['angle_ols'], 4),
            'angle_sir': round(v['angle_sir'], 4),
            'angle_save': round(v['angle_save'], 4),
            'dcor_found': round(v['dcor_found'], 6),
            'dcor_ols': round(v['dcor_ols'], 6),
            'dcor_sir': round(v['dcor_sir'], 6),
            'dcor_save': round(v['dcor_save'], 6),
            'spread': round(v['spread'], 6),
        })
_stamp = write_csv(RESULTS / 'results_1d_baselines.csv', rows, seeds=SEEDS,
                              script='ch04_single_direction_grid.py',
                              n_restarts=N_RESTARTS, max_iter=MAX_ITER)
print(f"Wrote: results_1d_baselines.csv ({len(rows)} rows)", flush=True)
