"""
pp_helpers.py — importable projection pursuit utilities (no auto-execution).
"""
import numpy as np
from .dcor_optimizer import optimize_dcor, dcor_u
from .evaluation import principal_angles


def deflate_Y(Y, X, beta):
    z = X @ beta
    z_nrm = np.linalg.norm(z)
    if z_nrm < 1e-12:
        return Y.copy()
    z_unit = z / z_nrm
    return Y - np.dot(Y, z_unit) * z_unit


def deflate_X(X, found_dirs):
    Q, _ = np.linalg.qr(found_dirs)
    return X - X @ Q @ Q.T


def sigma_inv_root(X, ridge=1e-10):
    """Symmetric inverse square root of the sample covariance of X."""
    S = np.cov(X, rowvar=False)
    S = np.atleast_2d(S)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, ridge * max(w.max(), 1.0))
    return V @ np.diag(w ** -0.5) @ V.T


"""Sigma-orthogonal deflation is not a separate deflation function.

Plain :func:`deflate_X` removes the span of the directions found so far in the
Euclidean geometry, which forces ``beta_i' beta_j = 0``. That leaves
``Cov(X beta_i, X beta_j) = beta_i' Sigma beta_j`` free, and it vanishes only
for isotropic predictors. Deflating in the Sigma inner product instead forces
``beta_i' Sigma beta_j = 0`` — uncorrelated projections, the middle condition of
the deflation hierarchy rather than the leftmost one.

The implementation is whitening. With ``W = Sigma^{-1/2}`` the whole search runs
on ``Z = X W``, where ordinary Euclidean deflation applies, and the answer is
mapped back as ``beta = W v``; then ``beta_i' Sigma beta_j = v_i' v_j = 0``.
:func:`seq_pp` does this when ``deflation='X_sigma'``. It requires Sigma to be
of full rank, which is why :func:`sigma_inv_root` floors the eigenvalues.
"""


def seq_pp(X, Y, k, deflation='X_deflation', n_restarts=5, max_iter=150,
           seed=0, verbose=False, init_method='random', informed_inits=()):
    """Extract k directions sequentially.

    ``deflation`` is one of ``'X_deflation'`` (orthogonal weights),
    ``'X_sigma'`` (uncorrelated projections, via whitening) or ``'Y_residual'``
    (linear residualisation of the response).

    ``init_method`` and ``informed_inits`` are handed to :func:`optimize_dcor`
    unchanged at every direction, so an inverse-regression seed is recomputed on
    the deflated predictors rather than reused from the undeflated ones. The
    defaults leave every existing caller with the random starts it had before.
    """
    n, p = X.shape
    W_hat = np.zeros((p, k))          # directions in the original X scale
    V_hat = np.zeros((p, k))          # the same directions in search coordinates
    per_dir_info = []
    inv_root = None
    if deflation == 'X_sigma':
        inv_root = sigma_inv_root(X)
        X_work = X @ inv_root          # search runs in whitened coordinates
    else:
        X_work = X.copy()
    Y_work = Y.copy()

    for j in range(k):
        beta, dcor_val, opt_info = optimize_dcor(
            X_work, Y_work, init_method=init_method,
            optimizer='gradient_ascent',
            n_restarts=n_restarts, max_iter=max_iter, seed=seed + j,
            informed_inits=informed_inits)

        V_hat[:, j] = beta
        if inv_root is None:
            W_hat[:, j] = beta
        else:
            b = inv_root @ beta        # X_work beta = X (Sigma^-1/2 beta)
            W_hat[:, j] = b / np.linalg.norm(b)
        z = X @ W_hat[:, j]
        dc_orig = dcor_u(z, Y)
        per_dir_info.append({'dcor_vs_Y_work': dcor_val,
                             'dcor_vs_Y_orig': dc_orig,
                             'spread': opt_info['val_spread']})
        if verbose:
            print(f"    dir {j+1}: dCor(Y_work)={dcor_val:.4f}  "
                  f"dCor(Y_orig)={dc_orig:.4f}", flush=True)

        if j < k - 1:
            if deflation == 'Y_residual':
                Y_work = deflate_Y(Y_work, X, W_hat[:, j])
            elif deflation in ('X_deflation', 'X_sigma'):
                X_work = deflate_X(X_work, V_hat[:, :j+1])

    return W_hat, per_dir_info


def seq_pp_auto(X, Y, k_max=None, alpha=0.05, n_perm=100,
                perm_restarts=2, perm_max_iter=60,
                deflation='X_deflation', n_restarts=5, max_iter=150,
                seed=0, verbose=False):
    """Sequential PP with a permutation-test stopping rule — selects k.

    At each step the best-restart dCor objective on (X_work, Y_work) is
    compared against a permutation null: Y_work rows are permuted n_perm
    times and the objective RE-OPTIMISED each time (small budget:
    perm_restarts restarts, perm_max_iter iterations), so the null
    accounts for the maximisation over β. Without re-optimisation the
    null would be anti-conservative (observed value is a max, fixed-β
    null is not).

    p-value = (1 + #{null >= observed}) / (n_perm + 1).
    Stop before adding the first direction with p > alpha.

    Returns (W_hat[:, :k_sel], per_dir_info, k_sel); per_dir_info has one
    entry per TESTED direction (the last, rejected one included) with its
    p-value. Explicit-k `seq_pp` is unchanged and remains the default path.
    """
    n, p = X.shape
    if k_max is None:
        k_max = p
    rng = np.random.default_rng(seed + 987654321)
    W_hat = np.zeros((p, k_max))
    per_dir_info = []
    X_work = X.copy()
    Y_work = Y.copy()
    k_sel = 0

    for j in range(k_max):
        beta, dcor_val, opt_info = optimize_dcor(
            X_work, Y_work, init_method='random',
            optimizer='gradient_ascent',
            n_restarts=n_restarts, max_iter=max_iter, seed=seed + j)

        null_vals = np.empty(n_perm)
        for b in range(n_perm):
            Y_perm = Y_work[rng.permutation(n)]
            _, null_vals[b], _ = optimize_dcor(
                X_work, Y_perm, init_method='random',
                optimizer='gradient_ascent',
                n_restarts=perm_restarts, max_iter=perm_max_iter,
                seed=seed + 1000 * (j + 1) + b)
        pval = (1.0 + float((null_vals >= dcor_val).sum())) / (n_perm + 1.0)

        info = {'dcor_vs_Y_work': dcor_val,
                'null_q95': float(np.quantile(null_vals, 1 - alpha)),
                'null_max': float(null_vals.max()),
                'p_value': pval,
                'accepted': pval <= alpha,
                'spread': opt_info['val_spread']}
        per_dir_info.append(info)
        if verbose:
            print(f"    dir {j+1}: dCor={dcor_val:.4f}  "
                  f"null q95={info['null_q95']:.4f}  p={pval:.4f}  "
                  f"{'ACCEPT' if info['accepted'] else 'STOP'}", flush=True)
        if not info['accepted']:
            break

        W_hat[:, j] = beta
        k_sel += 1
        if deflation == 'Y_residual':
            Y_work = deflate_Y(Y_work, X, beta)
        elif deflation == 'X_deflation':
            X_work = deflate_X(X_work, W_hat[:, :k_sel])

    return W_hat[:, :k_sel], per_dir_info, k_sel


def eval_subspace(W_hat, W_true, X, Y, label=''):
    angles = principal_angles(W_hat, W_true)
    k = W_true.shape[1]
    inter = []
    for i in range(k):
        for j in range(i+1, k):
            inter.append(dcor_u(X @ W_hat[:, i], X @ W_hat[:, j]))
    per_dir = [dcor_u(X @ W_hat[:, j], Y) for j in range(k)]
    gram = W_hat.T @ W_hat
    orth_err = np.linalg.norm(gram - np.eye(k), 'fro')

    if label:
        print(f"  [{label}]")
        print(f"    Principal angles (deg): {np.round(angles, 2)}")
        print(f"    Mean principal angle:   {angles.mean():.2f}°")
        print(f"    Per-dir dCor(z_j,Y):   {np.round(per_dir, 4)}")
        if inter:
            print(f"    Redundancy (inter-dCor): {np.round(inter, 4)}")
        print(f"    ||W^T W - I||_F:       {orth_err:.4f}", flush=True)

    return {'angles': angles, 'per_dir': per_dir, 'inter': inter,
            'orth_err': orth_err, 'mean_angle': float(angles.mean())}
