"""
dcor_optimizer.py

Finds β* = argmax_{|β|=1} dCor(Y, X^T β) via Riemannian gradient ascent
or derivative-free methods (Nelder-Mead, COBYLA).

Gradient derivation (matches common.R):
  Let z = Xβ,  A = u_centered(|z_i - z_j|),  B = u_centered(|y_i - y_j|)  (cached).
  dCov_u(z,y) = (1/n(n-3)) Σ_{i≠j} A_ij B_ij  =: s_xy / n(n-3)
  dCov_u(z,z) = s_xx / n(n-3)
  dCov_u(y,y) = s_yy / n(n-3)  (constant — cached)
  dCor(z,y) = s_xy / sqrt(s_xx * s_yy)

  ∂s_xy/∂β = (2/n(n-3)) X^T rowsum( sign(diff) ⊙ B )
  ∂s_xx/∂β = (4/n(n-3)) X^T rowsum( sign(diff) ⊙ A )
        (factor 4 because differentiating both A copies)

  ∂dCor/∂β = dCor * ( ∂s_xy/β / s_xy  −  0.5 * ∂s_xx/β / s_xx )
  Riemannian tangent: g_tan = g − (β^T g) β
"""

import numpy as np
import dcor as _dcor
from scipy.optimize import minimize


# ── U-centering ───────────────────────────────────────────────────────────────

def _u_center(D):
    """U-center a distance matrix (unbiased dCov estimator)."""
    n = D.shape[0]
    row_m = D.sum(axis=1) / (n - 2)
    grand_m = D.sum() / ((n - 1) * (n - 2))
    A = D - row_m[:, None] - row_m[None, :] + grand_m
    np.fill_diagonal(A, 0.0)
    return A


def _make_B(y):
    """Precompute u-centered distance matrix B and s_yy for y."""
    D = np.abs(y[:, None] - y[None, :])
    B = _u_center(D)
    n = len(y)
    s_yy = (B * B).sum() / (n * (n - 3))
    return B, s_yy


def dcor_u(x, y):
    """U-statistic (unbiased-dCov) distance correlation between 1-D samples.

    This is the SAME estimator `_dcor_and_grad` optimizes — use it wherever
    dCor values are compared or reported, so restart selection, derivative-
    free optimizers and evaluation all speak one estimator (the `dcor`
    package value is the biased V-statistic and differs at small n).
    Clamped to 0 when s_xx, s_yy or s_xy is non-positive (the U-statistic
    can be slightly negative under independence).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.size
    A = _u_center(np.abs(x[:, None] - x[None, :]))
    B = _u_center(np.abs(y[:, None] - y[None, :]))
    denom = n * (n - 3)
    s_xx = (A * A).sum() / denom
    s_yy = (B * B).sum() / denom
    s_xy = (A * B).sum() / denom
    if s_xx <= 0 or s_yy <= 0 or s_xy <= 0:
        return 0.0
    return float(s_xy / np.sqrt(s_xx * s_yy))


# ── Analytical gradient ───────────────────────────────────────────────────────

# Reusable (n,n) work buffers keyed by n — the gradient is called hundreds of
# times per restart and re-allocating ~8 n×n temporaries per call dominates
# the empirical O(n^2.8) scaling (Task 7). Same-process reuse only; joblib
# process workers each hold their own cache.
_GRAD_BUFS = {}


def _get_grad_bufs(n):
    bufs = _GRAD_BUFS.get(n)
    if bufs is None:
        bufs = _GRAD_BUFS[n] = (np.empty((n, n)), np.empty((n, n)))
    return bufs


def _dcor_and_grad(beta, X, B, s_yy):
    """
    Return (dCor value, Euclidean gradient w.r.t. β).
    Caller is responsible for projecting gradient to tangent space.

    Allocation-lean implementation: two cached buffers, in-place
    U-centering, einsum reductions — no math change vs the reference
    (_u_center + dense products); values agree to float rounding.
    """
    z = X @ beta
    n = len(z)
    buf1, buf2 = _get_grad_bufs(n)

    diff = np.subtract(z[:, None], z[None, :], out=buf1)   # (n,n)
    A = np.abs(diff, out=buf2)                             # |z_i - z_j|
    # U-center A in place (same formula as _u_center)
    row_m = A.sum(axis=1) / (n - 2)
    grand_m = A.sum() / ((n - 1) * (n - 2))
    A -= row_m[:, None]
    A -= row_m[None, :]
    A += grand_m
    np.fill_diagonal(A, 0.0)
    denom = n * (n - 3)

    s_xx = np.einsum('ij,ij->', A, A) / denom
    s_xy = np.einsum('ij,ij->', A, B) / denom

    if s_xx <= 0 or s_yy <= 0 or s_xy <= 0:
        dc = max(0.0, s_xy / np.sqrt(max(s_xx, 1e-30) * s_yy)) if s_yy > 0 else 0.0
        # Degenerate region (usually s_xy <= 0 at an uninformative β).
        # Returning a zero gradient here permanently stalls the restart at
        # its initialisation; instead follow the ascent direction of s_xy
        # itself so the iterate can climb into the s_xy > 0 region where
        # the true objective is defined. (Adam rescales, so the raw
        # magnitude of this surrogate gradient is irrelevant.)
        if s_yy > 0 and s_xx > 0:
            S = np.sign(diff, out=diff)
            grad_sxy = (2.0 / denom) * (X.T @ np.einsum('ij,ij->i', S, B))
            return dc, grad_sxy
        return dc, np.zeros_like(beta)

    dc = s_xy / np.sqrt(s_xx * s_yy)

    S = np.sign(diff, out=diff)  # antisymmetric (n,n); diff no longer needed
    # ∂s_xy: 2/denom * X^T (S ⊙ B) 1_n
    row_T = np.einsum('ij,ij->i', S, B)
    grad_sxy = (2.0 / denom) * (X.T @ row_T)

    # ∂s_xx: 4/denom * X^T (S ⊙ A) 1_n
    row_U = np.einsum('ij,ij->i', S, A)
    grad_sxx = (4.0 / denom) * (X.T @ row_U)

    grad = dc * (grad_sxy / s_xy - 0.5 * grad_sxx / s_xx)
    return dc, grad


def check_gradient(n=200, p=5, nonlinearity='square', seed=0, eps=1e-6):
    """Central-difference verification of _dcor_and_grad.

    dCor(Xβ, Y) is defined for any β (scale-invariant in β), so the plain
    Euclidean gradient can be checked entrywise without sphere retraction.
    Returns max relative error vs central differences.
    """
    from .data_generator import data_generator
    k = 2 if nonlinearity in ('product', 'sum_squares', 'sine_product') else 1
    X, Y, W_true = data_generator(n, p, k, nonlinearity, 0.0, seed)
    Y = np.asarray(Y).reshape(-1)
    B, s_yy = _make_B(Y)
    rng = np.random.default_rng(seed + 1)
    # Perturbed true direction: at a fully random β, s_xy is often ≤ 0 and
    # the degenerate clamp returns an exactly-zero gradient — FD would then
    # vacuously compare 0 to 0. Start where the gradient is live.
    beta = W_true[:, 0] + 0.3 * rng.standard_normal(p)
    beta /= np.linalg.norm(beta)
    _, g = _dcor_and_grad(beta, X, B, s_yy)
    g_fd = np.zeros(p)
    for i in range(p):
        bp = beta.copy(); bp[i] += eps
        bm = beta.copy(); bm[i] -= eps
        vp, _ = _dcor_and_grad(bp, X, B, s_yy)
        vm, _ = _dcor_and_grad(bm, X, B, s_yy)
        g_fd[i] = (vp - vm) / (2.0 * eps)
    return float(np.abs(g - g_fd).max() / (np.abs(g_fd).max() + 1e-30))


# ── Initialization ────────────────────────────────────────────────────────────

def _init_beta(X, Y, method, rng):
    p = X.shape[1]
    if method == 'random':
        v = rng.standard_normal(p)
    elif method == 'pca':
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        v = Vt[0]
    elif method == 'ols':
        v, *_ = np.linalg.lstsq(X, Y, rcond=None)
    elif method in ('sir', 'save'):
        # Informed seeds from the inverse-regression baselines: cheap,
        # often within the dCor basin for monotone / symmetric links.
        # Used as INITIALISATION only — dCor remains the objective.
        from .sdr_baselines import sir as _sir, save as _save
        fn = _sir if method == 'sir' else _save
        try:
            v = fn(X, np.asarray(Y).reshape(-1), k=1)[:, 0]
        except Exception:
            v = rng.standard_normal(p)
    else:
        raise ValueError(f"Unknown init_method: {method}")
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else rng.standard_normal(p) / np.sqrt(p)


def _probe_inits(X, Y, B, s_yy, n_probes, m_top, rng):
    """Value-probe initialisation: score n_probes random unit directions by
    the U-statistic dCor VALUE (no gradient runs) and return the m_top best
    as starting points. O(n² · n_probes) — cheap relative to full Adam runs;
    designed for narrow-basin objectives (sine) where random restarts
    almost never land inside the basin."""
    n, p = X.shape
    denom = n * (n - 3)
    best = []
    for _ in range(n_probes):
        w = rng.standard_normal(p)
        w /= np.linalg.norm(w)
        z = X @ w
        A = _u_center(np.abs(z[:, None] - z[None, :]))
        s_xx = (A * A).sum() / denom
        s_xy = (A * B).sum() / denom
        val = s_xy / np.sqrt(s_xx * s_yy) if (s_xx > 0 and s_xy > 0) else 0.0
        best.append((val, w))
    best.sort(key=lambda t: t[0], reverse=True)
    return [w for _, w in best[:m_top]]


def aggregate_directions(betas, vals, m_top=None):
    """Subspace-average the top-m restart directions (sign-invariant):
    leading eigenvector of Σ_i β_i β_iᵀ over the m_top best restarts.
    Stabilises recovery on flat landscapes where near-tied restarts
    scatter around the truth."""
    order = np.argsort(vals)[::-1]
    if m_top is not None:
        order = order[:m_top]
    M = np.zeros((betas[0].size, betas[0].size))
    for i in order:
        M += np.outer(betas[i], betas[i])
    evals, evecs = np.linalg.eigh(M)
    return evecs[:, -1]


# ── Single-start optimizers ───────────────────────────────────────────────────

def _riemannian_adam(beta0, X, B, s_yy,
                     max_iter=300, lr=0.05, lr_min=0.005, tol=1e-6):
    """Riemannian Adam on S^{p-1} with cosine LR annealing."""
    beta = beta0.copy()
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    prev_val = -np.inf
    best_beta, best_val = beta.copy(), -np.inf

    for t in range(1, max_iter + 1):
        lr_t = lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * (t-1) / max_iter))
        val, g_euc = _dcor_and_grad(beta, X, B, s_yy)
        if val > best_val:
            best_val, best_beta = val, beta.copy()
        # Project to tangent space
        g_tan = g_euc - np.dot(beta, g_euc) * beta

        # Adam moments
        m = b1 * m + (1 - b1) * g_tan
        v = b2 * v + (1 - b2) * g_tan**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        step = m_hat / (np.sqrt(v_hat) + eps_adam)

        # Gradient clipping
        step_nrm = np.linalg.norm(step)
        if step_nrm > 1.0:
            step = step / step_nrm

        beta_new = beta + lr_t * step
        nrm = np.linalg.norm(beta_new)
        if nrm < 1e-12:
            break
        beta = beta_new / nrm

        if abs(val - prev_val) < tol and t > 20:
            break
        prev_val = val

    final_val, _ = _dcor_and_grad(beta, X, B, s_yy)
    if final_val > best_val:
        best_val, best_beta = final_val, beta
    # best-seen iterate, not the final one — a late uphill-then-settle step
    # can otherwise return a slightly sub-optimal β
    return best_beta, best_val


def _nelder_mead(beta0, X, Y, max_iter=500):
    """Nelder-Mead on the sphere (project after each step).

    Objective and returned value use the U-statistic `dcor_u` so all
    optimizer choices select and report the same estimator.
    """
    def neg_dcor(b):
        nrm = np.linalg.norm(b)
        if nrm < 1e-12:
            return 0.0
        b_unit = b / nrm
        return -dcor_u(X @ b_unit, Y)

    res = minimize(neg_dcor, beta0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
    beta = res.x / np.linalg.norm(res.x)
    val = dcor_u(X @ beta, Y)
    return beta, val


def _cobyla(beta0, X, Y, max_iter=500):
    """COBYLA with unit-norm constraint. Uses `dcor_u` (see _nelder_mead)."""
    def neg_dcor(b):
        nrm = np.linalg.norm(b)
        if nrm < 1e-12:
            return 0.0
        b_unit = b / nrm
        return -dcor_u(X @ b_unit, Y)

    res = minimize(neg_dcor, beta0, method='COBYLA',
                   constraints={'type': 'ineq', 'fun': lambda b: 1e-4 - abs(np.dot(b, b) - 1)},
                   options={'maxiter': max_iter, 'rhobeg': 0.1})
    beta = res.x / np.linalg.norm(res.x)
    val = dcor_u(X @ beta, Y)
    return beta, val


# ── Public API ────────────────────────────────────────────────────────────────

def optimize_dcor(X, Y, init_method='random', optimizer='gradient_ascent',
                  max_iter=300, tol=1e-6, n_restarts=1, seed=0,
                  n_probes=0, informed_inits=(), aggregate_top_m=None):
    """
    Find β* = argmax_{|β|=1} dCor(Y, X β).  U-statistic dCor is used for
    optimisation, restart selection AND the returned value (one estimator
    everywhere — see `dcor_u`).

    Parameters
    ----------
    X            : (n, p)
    Y            : (n,)
    init_method  : 'random' | 'pca' | 'ols' | 'sir' | 'save'
    optimizer    : 'gradient_ascent' | 'nelder_mead' | 'cobyla'
    max_iter     : int
    tol          : float
    n_restarts   : int
    seed         : int
    n_probes     : int — if > 0, score this many random directions by dCor
                   VALUE first and use the best as extra starting points
                   (narrow-basin rescue, e.g. sine). 0 = off (default).
    informed_inits : iterable of init_method names (e.g. ('sir','save','ols'))
                   each contributing one extra seeded restart.
    aggregate_top_m : int or None — if set, also subspace-average the top-m
                   restart directions (`aggregate_directions`) and return
                   that direction (best single restart kept in info).

    Returns
    -------
    beta_hat     : (p,) unit vector
    dcor_val     : float — U-statistic dCor at solution
    info         : dict  — convergence metadata
    """
    rng = np.random.default_rng(seed)
    B, s_yy = _make_B(Y)   # cached for gradient_ascent

    # Build the initialisation list. Defaults reproduce the old behaviour
    # exactly: restart 0 = init_method, the rest random.
    inits = [_init_beta(X, Y, init_method, rng)]
    for m in informed_inits:
        if len(inits) < n_restarts:
            inits.append(_init_beta(X, Y, m, rng))
    if n_probes > 0 and len(inits) < n_restarts:
        inits.extend(_probe_inits(X, Y, B, s_yy, n_probes,
                                  n_restarts - len(inits), rng))
    while len(inits) < n_restarts:
        inits.append(_init_beta(X, Y, 'random', rng))

    best_beta, best_val = None, -np.inf
    restart_vals, restart_betas = [], []

    for beta0 in inits[:n_restarts]:
        if optimizer == 'gradient_ascent':
            beta, val = _riemannian_adam(beta0, X, B, s_yy, max_iter=max_iter, tol=tol)
        elif optimizer == 'nelder_mead':
            beta, val = _nelder_mead(beta0, X, Y, max_iter=max_iter)
        elif optimizer == 'cobyla':
            beta, val = _cobyla(beta0, X, Y, max_iter=max_iter)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        restart_vals.append(val)
        restart_betas.append(beta.copy())
        if val > best_val:
            best_val, best_beta = val, beta.copy()

    info = {
        'n_restarts': n_restarts,
        'restart_vals': restart_vals,
        'val_spread': max(restart_vals) - min(restart_vals),
    }

    if aggregate_top_m:
        beta_agg = aggregate_directions(restart_betas, restart_vals,
                                        m_top=aggregate_top_m)
        val_agg = dcor_u(X @ beta_agg, np.asarray(Y).reshape(-1))
        info['best_single_val'] = best_val
        info['best_single_beta'] = best_beta
        info['aggregated'] = True
        best_beta, best_val = beta_agg, val_agg

    return best_beta, best_val, info
