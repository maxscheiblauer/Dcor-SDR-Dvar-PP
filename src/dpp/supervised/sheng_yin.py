"""
sheng_yin.py

Python port of the distance-covariance sufficient-dimension-reduction estimator
of Sheng & Yin (2013, 2016), as their own group distributes it.

Provenance of the reference code
--------------------------------
The authors publish no code with either paper. The MATLAB implementation used
here is the one bundled by Wu & Chen (2021, `wu2021mm`) in
https://github.com/runxiong-wu/MMRN, whose `readme.txt` states that the
`Technometric/` folder "contains the codes for the existing methods solving the
DCOV-based SDR model", i.e. Yin's own solver, kept there as the baseline their
MM algorithm is compared against (`Model_A.m`, `demo_circle.m` call it as
"SQP" / "Yin's"). The relevant files are:

    dcsol.m / dcsol2.m   the estimator: whiten, initialise, then fmincon
    dcreg.m              the objective, ``-10 * dCov(X N2 v, y)``
    tcon.m               the constraint, ``v'v - I_d = 0``
    DistCorrVec.m        the V-statistic distance covariance / correlation

What this port reproduces exactly
---------------------------------
* the objective: the **V-statistic (biased) distance covariance**, unsquared,
  scaled by 10 as in `dcreg.m` — not this project's U-statistic dCor²;
* the parameterisation: the search runs over the *whitened* predictors
  ``Z = (X - x_bar) Sigma^{-1/2}`` with the orthonormality constraint
  ``V'V = I_d`` imposed there, and the answer is mapped back as
  ``B = Sigma^{-1/2} V``, so ``B`` is **not** column-orthonormal in X-space;
* the constrained SQP solve, `fmincon(..., 'Algorithm', 'active-set')` with
  ``TolFun = TolX = 1e-4``, here `scipy.optimize.minimize(method='SLSQP')` with
  the same nonlinear equality constraint and tolerance, and likewise with a
  finite-difference gradient (their code supplies none either);
* the two-stage initialisation of `dcsol.m`: an inverse-regression seed, then
  500 orthonormalised random perturbations ``orth(v_i + 0.1 V_c G)`` of it,
  keeping whichever scores best under the same objective.

Where it deviates, and why
--------------------------
`dcsol.m` picks its inverse-regression seed as the best of SIR, DR (directional
regression) and dMAVE. DR and dMAVE are not implemented in this repository;
this port uses the best of SIR and SAVE by the same criterion — the objective
value at the seed — so the seed is chosen the same way from a smaller pool.
`find_initial.m`, which the MM paper uses for *both* solvers in its published
comparison, uses only SIR and DR, so a two-method pool is what their own timing
comparison runs on as well. The perturbation stage of `dcsol.m` then dominates
the choice of starting point in any case.

Reference
---------
Sheng, W. & Yin, X. (2013). Direction estimation in single-index models via
distance covariance. Journal of Multivariate Analysis 122, 148-161.
Sheng, W. & Yin, X. (2016). Sufficient dimension reduction via distance
covariance. Journal of Computational and Graphical Statistics 25(1), 91-104.
Wu, R. & Chen, X. (2021). MM algorithms for distance covariance based
sufficient dimension reduction and sufficient variable selection. CSDA 155.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .sdr_baselines import sir, save, _whiten


# ── the objective: V-statistic distance covariance (DistCorrVec.m) ────────────

def _dist(Z):
    """Euclidean distance matrix of the rows of Z (n, d)."""
    Z = np.atleast_2d(Z.T).T if Z.ndim == 1 else Z
    sq = np.einsum('ij,ij->i', Z, Z)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2)


def _double_center(D):
    """Double-centre a distance matrix (the biased / V-statistic centring)."""
    r = D.mean(axis=1)
    return D - r[:, None] - r[None, :] + D.mean()


def dcov_v(x, y):
    """V-statistic distance covariance, **unsquared**, of two samples.

    This is `DistCorrVec.m`'s first return value: ``sqrt(S1 + S2 - 2 S3)``.
    Written here as ``sqrt(mean(A_dc * B_dc))``, which is the identical
    quantity in double-centred form and agrees to float rounding.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = _double_center(_dist(x.reshape(len(x), -1)))
    B = _double_center(_dist(y.reshape(len(y), -1)))
    return float(np.sqrt(max(np.einsum('ij,ij->', A, B) / A.size, 0.0)))


def dcor_v(x, y):
    """V-statistic distance correlation (unsquared) — `DistCorrVec.m` output 2.

    Used only for the seed-selection step of `dcsol.m`, which scores its
    inverse-regression candidates by dCor rather than dCov.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = _double_center(_dist(x.reshape(len(x), -1)))
    B = _double_center(_dist(y.reshape(len(y), -1)))
    n2 = A.size
    vxy = np.einsum('ij,ij->', A, B) / n2
    vxx = np.einsum('ij,ij->', A, A) / n2
    vyy = np.einsum('ij,ij->', B, B) / n2
    if vxx <= 0 or vyy <= 0 or vxy <= 0:
        return 0.0
    return float(np.sqrt(vxy) / np.sqrt(np.sqrt(vxx * vyy)))


# ── the estimator (dcsol.m / dcsol2.m) ───────────────────────────────────────

def _orth(M):
    """Orthonormal basis of the column space, MATLAB `orth`."""
    U, s, _ = np.linalg.svd(np.asarray(M, float), full_matrices=False)
    tol = max(M.shape) * (s[0] if s.size else 0.0) * np.finfo(float).eps
    return U[:, s > tol]


def _orth_complement(V):
    """Orthonormal basis of the complement of span(V), MATLAB `orthcomp`."""
    p, d = V.shape
    Q, _ = np.linalg.qr(V, mode='complete')
    return Q[:, d:]


def sheng_yin_sdr(X, y, d=1, n_perturb=500, seed=0, tol=1e-4, max_iter=1000,
                  v0=None):
    """Sheng & Yin's dCov-based SDR estimate of the central subspace.

    Parameters
    ----------
    X, y       : (n, p) predictors and (n,) response.
    d          : working dimension of the central subspace.
    n_perturb  : random perturbations of the inverse-regression seed, as in
                 `dcsol.m` (500 there). 0 reproduces `dcsol2.m`, which starts
                 from the seed directly.
    seed       : RNG seed for the perturbations.
    tol        : SLSQP tolerance, matching `fmincon`'s TolFun/TolX of 1e-4.
    v0         : optional (p, d) starting point in *whitened* coordinates,
                 bypassing the initialisation (this is `dcsol1.m`).

    Returns
    -------
    B     : (p, d) directions in the original predictor scale, ``Sigma^{-1/2} V``.
            Columns are **not** unit-norm — that is what their code returns.
    info  : dict with the objective at the solution, the seed method, the
            elapsed seconds and the SLSQP exit status.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n, p = X.shape
    t0 = time.perf_counter()

    Z, inv_root, _ = _whiten(X)          # Z = (X - x_bar) Sigma^{-1/2}
    B_y = _double_center(_dist(y.reshape(n, 1)))

    def neg_obj(v_flat):
        V = v_flat.reshape(p, d)
        A = _double_center(_dist(Z @ V))
        v2 = np.einsum('ij,ij->', A, B_y) / A.size
        return -10.0 * np.sqrt(max(v2, 0.0))       # dcreg.m

    # ── initialisation (dcsol.m) ────────────────────────────────────────────
    if v0 is None:
        seed_method, v_seed, best = None, None, -np.inf
        for name, fn in (('sir', sir), ('save', save)):
            try:
                W = fn(X, y, k=d)
            except Exception:
                continue
            # sir/save return directions in the original scale; `dcsol.m` runs
            # them on the whitened sX, so map back: V = Sigma^{1/2} W.
            V = _orth(np.linalg.solve(inv_root, W))
            if V.shape[1] < d:
                continue
            val = dcor_v(X @ (inv_root @ V), y)     # p1..p3 in dcsol.m
            if val > best:
                best, seed_method, v_seed = val, name, V[:, :d]
        if v_seed is None:                          # degenerate fallback
            v_seed = _orth(np.random.default_rng(seed).standard_normal((p, d)))
            seed_method = 'random'

        v_start, start_val = v_seed, neg_obj(v_seed.ravel())
        if n_perturb > 0 and p > d:
            rng = np.random.default_rng(seed)
            Vc = _orth_complement(v_seed)
            for _ in range(n_perturb):
                cand = _orth(v_seed + 0.1 * (Vc @ rng.standard_normal((p - d, d))))
                if cand.shape[1] < d:
                    continue
                val = neg_obj(cand[:, :d].ravel())
                if val < start_val:
                    start_val, v_start = val, cand[:, :d]
    else:
        v_start, seed_method = np.asarray(v0, float).reshape(p, d), 'given'

    # ── constrained solve (fmincon active-set -> SLSQP) ─────────────────────
    iu = np.triu_indices(d)

    def con(v_flat):
        V = v_flat.reshape(p, d)
        return (V.T @ V - np.eye(d))[iu]            # tcon.m

    res = minimize(neg_obj, v_start.ravel(), method='SLSQP',
                   constraints=[{'type': 'eq', 'fun': con}],
                   options={'ftol': tol, 'maxiter': max_iter})

    V = res.x.reshape(p, d)
    V = _orth(V)[:, :d] if np.linalg.matrix_rank(V) == d else v_start
    B = inv_root @ V

    return B, {
        'objective': -neg_obj(V.ravel()) / 10.0,    # dCov at the solution
        'init': seed_method,
        'seconds': time.perf_counter() - t0,
        'status': int(res.status),
        'nfev': int(res.nfev),
    }
