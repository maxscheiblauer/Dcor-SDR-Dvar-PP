# Projection pursuit with distance-based dependence measures: reproduction code

This repository regenerates every number in the thesis it accompanies. It contains the
methods, the simulators, the two real data sets, and one script per experiment. Each
script writes a CSV into `results/`, and the CSVs that the thesis was written from are
already there, so a rerun can be compared against them with a plain diff.

Nothing here draws a figure, and nothing here builds the thesis. Two files are not part
of the experiments and need filling in before publication: `LICENSE` (author name) and
`CITATION.cff` (author, title, repository URL, year).

## Layout

```
data/        the two data sets, plus the file the Chapter 6 data gate checks against
src/         the methods, the simulators, the evaluation code, the real-data pipeline
scripts/     one driver per experiment; every driver writes CSVs and prints tables
results/     the CSVs behind the thesis tables; a rerun overwrites them in place
```

`src/` holds two method packages and two support packages:

| package | contents |
|---|---|
| `dpp.supervised` | distance correlation `dcor_u` (U-statistic, on the **squared** scale), its closed-form Riemannian gradient, the sphere and Stiefel optimisers, sequential projection pursuit with three deflation strategies, the joint search with the λ redundancy penalty, SIR and SAVE, the Sheng & Yin solver, the response-model simulator |
| `dpp.unsupervised` | distance variance `dvar` and its gradient, the biloop transform, sequential and joint projection pursuit, whitening, the permutation test for the number of directions, the latent-factor simulator |
| `realdata` | the Ludvigson-Ng data pipeline (`dataio`, `pca`), the forecasting second stage (`regression`), the forecast metrics including Diebold-Mariano (`metrics`), the unified dimension-reduction interface (`sdr_registry`), INPCA (`inpca`), and the shared machinery of the Chapter 6 studies (`dcorlin`, `recovery`, `holdout`) |
| `designs` | two simulation designs shared by several drivers: the Sheng & Yin (2016) design and the shape-table draws |
| `csvout` | the CSV writer every driver uses, and `pin_blas_threads` |

## Running it

```
python -m venv .venv && .venv/Scripts/activate     # Windows; use bin/activate elsewhere
pip install -r requirements.txt
python scripts/ch03_gradient_check.py              # any driver, from the repository root
```

Drivers put `src/` on `sys.path` themselves, so no installation is needed; `pip install
-e .` also works. Run them from the repository root or any directory: paths are
resolved from each file's own location, never from the working directory.

There is deliberately no script that runs all of them. Each experiment is independent
and answers one question; run the one whose number you want to check.

**Threads.** Every driver calls `pin_blas_threads()` before importing numpy, which sets
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS`
to 1. This is not a performance setting. The thread count changes the order of the
floating-point reductions inside the distance-matrix products, which changes results in
the last bits, which decides the `s_xy <= 0` comparison that selects the
degenerate-point branch of the distance-correlation gradient. The published numbers are
single-threaded. Do not remove those calls, and do not move them below the numpy import,
where they have no effect.

## Seeds

Seeds are literals at the top of each driver, and the seed list is written into the
header of every CSV the driver produces. There is no environment variable and no
command-line switch: changing a seed means editing the file, which is a visible change.

| seed list | used by |
|---|---|
| `42, 7, 123, 2024, 5, 2026, 17, 99, 777, 31337` | the Chapter 3 studies, where a gradient identity should hold at every draw, so ten draws test it rather than one |
| `42, 7, 123, 2024, 5` | Chapters 4 and 5 |
| `20260808, 7, 123` | `ch03_derivative_free.py`, at three seeds because each takes ~35 minutes |
| `20260813` | base seed of the two Sheng & Yin design studies, which draw replicates rather than seeds |
| `20260808` | `ch05_shape_table.py`, one sequential generator threaded through all rows |
| `0` | Chapter 6, which has one panel, one split and one fixed factor block, so a seed varies only the optimiser's starting points |
| `42, 7, 123` | the two outlier-recovery studies and the case-study exhibit |
| `0, 42, 7, 123, 2024` | `ch06_case_mechanism.py` |

Three drivers key their random streams on **position in a configuration list**, not on
the seed value: `ch04_shengyin_2016_design.py`, `ch04_initialisation_ablation.py` and
`ch04_fair_comparison.py`, and `ch05_shape_table.py` threads a single generator through
every row in order. Editing, reordering or trimming those lists changes numbers that
have nothing to do with the row you edited. Run them whole.

## What each script produces

Chapter numbers refer to the thesis. Runtimes are wall-clock from one run on one
machine (see below) and are indicative only; they say which scripts are minutes and
which are hours, nothing more.

### Chapter 3: the optimisers

| script | writes | runtime |
|---|---|---|
| `ch03_gradient_check.py` | `results_gradcheck.csv` | 2 min |
| `ch03_sir_initialisation.py` | `results_sir_init.csv` | 2 min |
| `ch03_degenerate_branch.py` | `results_degenerate_share.csv` | 1 min |
| `ch03_derivative_free.py` | `results_derivative_free.csv` | **106 min** |

### Chapter 4: supervised projection pursuit, distance correlation

| script | writes | runtime |
|---|---|---|
| `ch04_single_direction_grid.py` | `results_1d_baselines.csv` | 2 min |
| `ch04_deflation_strategies.py` | `results_deflation.csv` | 4 min |
| `ch04_joint_and_penalty.py` | `results_joint.csv` | 8 min |
| `ch04_known_start.py` | `results_known_start.csv` | 1 min |
| `ch04_shengyin_on_our_grid.py` | `results_sheng_yin_ch4.csv` | 5 min |
| `ch04_shengyin_2016_design.py` | `results_sheng_yin_2016.csv` | **60 min** |
| `ch04_cost_scaling_n.py` | `results_cost_scaling_n.csv` | 14 min |
| `ch04_initialisation_ablation.py` | `results_init_ablation.csv` | 7 min |
| `ch04_master_grid.py` | `results_unified_grid.csv` | 18 min |
| `ch04_fair_comparison.py` | `results_fair_comparison.csv` | **57 min** |
| `ch04_autompg.py` | `autompg_ch4.csv` | 2 min |

### Chapter 5: unsupervised projection pursuit, distance variance

| script | writes | runtime |
|---|---|---|
| `ch05_single_direction_grid.py` | `results_1d.csv` | 1 min |
| `ch05_multi_direction.py` | `results_4.csv` | 2 min |
| `ch05_robustness.py` | `results_robust.csv`, `results_robust_agg.csv` | 4 min |
| `ch05_whitening.py` | `results_whitening.csv` | 1 min |
| `ch05_equal_variance.py` | `results_equalvar.csv` | 4 min |
| `ch05_coupling.py` | `results_coupling.csv` | 1 min |
| `ch05_multigroup.py` | `results_multigroup.csv` | 4 min |
| `ch05_minimisation.py` | `results_minimisation.csv` | 6 min |
| `ch05_shape_table.py` | `results_shape.csv` | 1 min |

### Chapter 6: excess bond returns

`ch06_data_gate.py` comes first: it rebuilds the macro panel from the raw data and
checks that the extracted factors reproduce Ludvigson & Ng's own published `Fhat_T` to
`|corr| = 1.00000` on all eight factors. It exits non-zero if that fails, which means
the data or the transform pipeline is wrong and no later number can be trusted.

| script | writes | runtime |
|---|---|---|
| `ch06_data_gate.py` | `data_gate.csv` | 1 min |
| `ch06_full_grid.py` | `grid_full.csv`, `grid_full_dm.csv` | 3 min |
| `ch06_best_d_dm_tests.py` | `ch6_bestd_levels.csv`, `ch6_bestd_dm.csv`, `ch6_dvar_anatomy.csv` | 3 min |
| `ch06_dcor_linear_vs_poly.py` | `exp1_dcor_poly_vs_lin.csv`, `exp2_leaked_dcorlin_dm.csv` | 1 min |
| `ch06_fixed_d_bottleneck.py` | `fixed_d_bottleneck.csv` | 2 min |
| `ch06_widen_bottleneck.py` | `widen_bottleneck.csv` | 2 min |
| `ch06_peer_dm_published.py` | `peer_dm_published.csv` | 1 min |
| `ch06_shengyin_on_panel.py` | `sheng_yin_ch6.csv` | 5 min |
| `ch06_outlier_recovery.py` | `outlier_recovery.csv`, `outlier_recovery_agg.csv` | 13 min |
| `ch06_holdout_recovery.py` | `holdout_recovery.csv`, `holdout_recovery_agg.csv` | 19 min |
| `ch06_case_motivating_example.py` | `motivating_example.csv` | 5 min |
| `ch06_case_mechanism.py` | `mechanism_directions.csv`, `mechanism_shape.csv`, `mechanism_spike.csv` | 4 min |

Everything together is about six hours, most of it in the three studies marked in bold.
Those three are slow by design: the derivative-free solvers are given twenty times the
gradient solver's iteration budget so that a poor result cannot be blamed on the budget,
and the Sheng & Yin sequential-quadratic-programming solver costs about 14 s per fit
against 1.6 s for the gradient search. Do not shorten them; an earlier run at equal
budget produced a false finding.

**The machine these runtimes come from**: Intel Core Ultra 7 155H (16 cores, 22
threads, 1.4 GHz base), 31.5 GB RAM, Windows 11, Python 3.12.10, BLAS pinned to one
thread per the note above, joblib worker counts as each script sets them. Wall-clock
will differ elsewhere, often by a lot on fewer cores. The *numbers* will not: worker count
changes how long a run takes, not what it returns, because each restart is independently
seeded and joblib preserves result order.

## CSV format

Each file opens with comment lines naming the script, the seed list, and the budgets the
run used, then the data:

```
# script: ch03_gradient_check.py
# seeds: 42, 7, 123, 2024, 5, 2026, 17, 99, 777, 31337
index,seed,n,p,best_h,err_at_best_h,err_at_1e5
dCor,42,200,5,3.162277660168379e-06,4.6342634267374365e-11,1.5335579637745428e-10
```

Read them with `pandas.read_csv(path, comment="#")`. To check a rerun, compare
everything below the header:

```
diff <(grep -v '^#' old.csv) <(grep -v '^#' new.csv)
```

On the environment in `requirements.txt` the statistical columns come back **character
for character identical**: the reproduction is exact, not approximate. A difference
there means the environment moved, the thread pinning was defeated, or a seed was
edited.

The exception is the columns that record wall-clock time: `elapsed`, `time`,
`seconds`, `sy_seconds` and the cost ratios derived from them. Those measure the
machine, not the method, and they do not reproduce: on a machine running two of these
studies at once they came out up to 368 times the recorded value. Three scripts report
a timing *comparison* as their result rather than as an aside:
`ch04_cost_scaling_n.py`, `ch04_shengyin_2016_design.py` and `ch04_fair_comparison.py`,
which measure the gradient search against the sequential-quadratic-programming solver.
Run those alone on an otherwise idle machine, or their ratios mean nothing. The
accuracy columns beside them are unaffected either way.

## Two conventions worth knowing before reading a number

**Distance correlation is on the squared scale.** `dcor_u` returns dCor², the
U-statistic (bias-corrected) estimator, and it is what every optimiser here maximises.
Do not compare its values against the biased V-statistic of the `dcor` package: the bias
acts like a fixed positive offset, so it is harmless when dependence is strong and
overwhelming when it is near zero. One documented case: 0.184 biased against 0.020
unbiased. The one place the biased estimator is used deliberately is
`dpp.unsupervised.evaluation.inter_direction_dcor`, whose docstring says so.

**Distance variance is a dispersion index, not a measure of non-Gaussianity.** It is
scale-equivariant, `dVar(aZ) = |a| dVar(Z)`, so it measures spread first. Held at unit
variance it orders distributions by how far their mass sits from the centre, and the
Gaussian sits in the *middle* of that ordering (0.635), with the two-point law above it
at 1.000 and Student t₃ below it at 0.465. Heavy tails move the index down, not up.
`ch05_shape_table.py` regenerates the evidence. `dvar` is the unsquared index; the
double-centred (biased) form is used deliberately, because the U-centred version can go
negative and the gradient of the square root is then undefined.

## Data provenance

`data/` is 748 KB and contains everything the real-data work needs.

| file | what it is |
|---|---|
| `LN2009_macro_panel_raw.csv` | the 131-series monthly macro panel of Ludvigson & Ng (2009), as exported from their replication archive; row 1 carries the per-series transformation codes |
| `RFS2009.xls` | their published factors f1..f8, the Cochrane-Piazzesi factor, and the one-year excess bond returns yr2..yr5 for maturities 2 to 5 |
| `Fhat64.mat` | their own full-sample factor estimates, used only by `ch06_data_gate.py` as the reference the pipeline must reproduce |
| `auto-mpg.data` | the AutoMPG data set from the UCI Machine Learning Repository, cached so the run needs no network; `ch04_autompg.py` re-downloads it from the UCI URL if the file is absent |

The macro panel is transformed, trimmed and reduced to factors by a line-for-line port
of the authors' own MATLAB (`Makedata.m`, `transx.m`, `standard.m`, `pc_T.m`) in
`src/realdata/dataio.py` and `src/realdata/pca.py`. Train/test split: predictors up to
1983-12 train, from 1984-01 test, forecast horizon 12 months, so a row is usable only
where a target twelve months ahead exists. The AutoMPG illustration has no split; it
reports out-of-fold R² from five-fold cross-validation, and the distance correlation
in-sample on all 392 complete rows.

The three Ludvigson-Ng files are redistributed here so that the reproduction is
self-contained. They are their authors' material, not ours, and the MIT licence on the
code does not extend to them.

## Credit for work that is not ours

| here | original |
|---|---|
| `src/dpp/supervised/sheng_yin.py` | The sequential-quadratic-programming solver of W. Sheng and X. Yin, *Direction estimation in single-index models via distance covariance*, Journal of Multivariate Analysis 122 (2013) 148-161, and *Sufficient dimension reduction via distance covariance*, Journal of Computational and Graphical Statistics 25(1) (2016) 91-104. Implemented in Python from the papers; it is not a translation of released code, and any discrepancy with their own implementation is ours. |
| `src/realdata/dataio.py`, `src/realdata/pca.py` | Port of the MATLAB replication files of S. C. Ludvigson and S. Ng, *Macro factors in bond risk premia*, The Review of Financial Studies 22(12) (2009) 5027-5067. |
| `src/realdata/inpca.py` | Implemented from F. Gunsilius and S. Schennach, *Independent nonlinear component analysis*, Journal of the American Statistical Association 118(542) (2023) 1305-1318. Entropic optimal transport via the POT package. |
| the λ penalty in `dpp/supervised/joint_optimization.py` | The independent-component objective of D. S. Matteson and R. S. Tsay, *Independent component analysis via distance covariance*, JASA 112(518) (2017) 623-637, used here as a regulariser on a supervised objective. |
| the biloop transform in `dpp/unsupervised/dvar_optimizer.py` | S. Leyder, J. Raymaekers and P. J. Rousseeuw, *Robust distance covariance*, International Statistical Review 94(1) (2026). |
| distance covariance itself | G. J. Székely, M. L. Rizzo and N. K. Bakirov, *Measuring and testing dependence by correlation of distances*, The Annals of Statistics 35(6) (2007) 2769-2794. |
| `dcor` package | Used for the V-statistic distance correlation in the one place the biased estimator is intended, and as an independent check on `dcor_u`. |
| `data/auto-mpg.data` | UCI Machine Learning Repository; the data set is due to R. Quinlan (1993). |

SIR and SAVE in `dpp/supervised/sdr_baselines.py` are standard inverse-regression
methods (Li 1991; Cook and Weisberg 1991), implemented here from the definitions.

## What is new here

Stated plainly, and bounded: closed-form Riemannian gradients for **both** indices, on
the sphere and on the Stiefel manifold, where the existing distance-covariance dimension
reduction literature avoids derivatives, using sequential quadratic programming (Sheng
and Yin) or a difference-of-convex reformulation (Wu and Chen 2021); the pairwise
dependence penalty used as a regulariser on a *supervised* objective, the penalty term
itself being Matteson and Tsay's; and the biloop transform examined as a projection
index, with its cost on clean data quantified. Everything else in `src/` is an
implementation of published method.
