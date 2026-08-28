"""
evaluation.py — MSS, orthogonality defect, baselines, summary table.

Mean Squared Sine of principal angles between subspaces (port from R):
    MSS(W_true, W_hat) = mean(1 - σ²)  where σ are singular values of W_true^T W_hat.
0 = perfect overlap, 1 = orthogonal subspaces. Lower is better.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import FastICA, PCA
import dcor

from .dvar_optimizer import make_index, whiten_svd


def mss_principal(W_true: np.ndarray, W_hat: np.ndarray) -> float:
    M = W_true.T @ W_hat
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.mean(1.0 - s**2))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Acute angle in degrees between two directions, sign-invariant.

    For k = 1 this is the principal angle between the two spanned lines, so it is
    the natural companion of :func:`mss_principal` (which is its mean squared sine).
    Accepts either a flat vector or a p x 1 column.

    Promoted here on 2026-08-11 from two ad-hoc copies — ``orchestrate.py``'s local
    ``ang`` and ``make_ch5_figures.py``'s ``_axis_angle`` — so that the grid
    generator, the figures and the smoke test all report the same quantity.
    """
    u = np.asarray(a, dtype=float).ravel()
    v = np.asarray(b, dtype=float).ravel()
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0.0:
        return float("nan")
    c = np.clip(abs(float(u @ v)) / denom, 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def random_mss_baseline(W_true: np.ndarray, n_rand: int = 20,
                        seed: int = 0) -> np.ndarray:
    """MSS of `n_rand` random k-frames against W_true — the no-information floor.

    The counterpart of :func:`random_dvar_baseline`: that one says what objective
    value an uninformed direction attains, this one says what recovery it attains.
    For k = 1 the expectation is 1 - 1/p, so a method is only doing work if it
    lands well below this.
    """
    rng = np.random.default_rng(seed)
    p, k = W_true.shape
    out = np.zeros(n_rand)
    for i in range(n_rand):
        A = rng.standard_normal((p, k))
        Q, _ = np.linalg.qr(A)
        out[i] = mss_principal(W_true, Q)
    return out


def orthogonality_defect(W: np.ndarray) -> float:
    k = W.shape[1]
    return float(np.linalg.norm(W.T @ W - np.eye(k), ord="fro"))


def pca_directions(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=k).fit(Xc)
    return pca.components_.T  # p x k


def fastica_directions(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Return p x k orthonormal frame whose columns span the FastICA components.

    sklearn's FastICA returns a mixing matrix A and unmixing W; we want a
    p-dim orthonormal frame for subspace comparison. Use the columns of A
    after orthonormalisation.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        ica = FastICA(n_components=k, random_state=seed, whiten="unit-variance",
                      max_iter=500, tol=1e-4)
        ica.fit(Xc)
    except Exception:                                # pragma: no cover - fall back to PCA
        return pca_directions(X, k)
    # mixing_ has shape (p, k) and represents directions in observed space
    A = np.asarray(ica.mixing_)
    if A.shape[1] < k:
        return pca_directions(X, k)
    Q, _ = np.linalg.qr(A[:, :k])
    return Q


def random_dvar_baseline(X: np.ndarray, n_rand: int = 30, whitened: bool = False,
                         index_fun: str = "plain", seed: int = 7) -> np.ndarray:
    """dVar of `n_rand` random unit directions; reference floor for the objective.

    Default whitened=False since the project found whitening harmful in the
    factor-model regime (see Report §1).
    """
    rng = np.random.default_rng(seed)
    X_use = whiten_svd(X)[0] if whitened else X
    p = X_use.shape[1]
    I_fun = make_index(index_fun)
    out = np.zeros(n_rand)
    for i in range(n_rand):
        w = rng.standard_normal(p); w /= np.linalg.norm(w)
        out[i] = I_fun(X_use @ w)
    return out


def _per_dir_dvar(X: np.ndarray, W: np.ndarray, index_fun: str = "plain"):
    I_fun = make_index(index_fun)
    return np.array([I_fun(X @ W[:, j]) for j in range(W.shape[1])])


def evaluate(W_hat: np.ndarray, X: np.ndarray, W_true: np.ndarray,
             index_fun: str = "plain", whitened_eval: bool = False,
             n_rand: int = 30) -> dict:
    """One-shot summary against PCA, FastICA and random baselines.

    Default whitened_eval=False since the project found whitening harmful
    (Report §1). Set True only for diagnostic comparisons against the
    Step 1 motivating example.
    """
    k = W_true.shape[1]
    W_pca = pca_directions(X, k)
    W_ica = fastica_directions(X, k)

    # dVar values are evaluated in the same coordinate system used during PP:
    # whitened space if whitened_eval, raw otherwise. Random baseline matches.
    if whitened_eval:
        Xw, W_white = whiten_svd(X)
        # Map true / pca / ica directions into whitened space
        # If X is whitened by W_white (n x p · p x p), then for any direction d
        # in raw coords, the corresponding direction in whitened coords is
        # solved from Xw·u = X·d → u = (Vt^T diag(1/s) sqrt(n-1))^-1 d.
        # Equivalently: u ∝ pinv(W_white) @ d, then renormalise.
        Wh_inv = np.linalg.pinv(W_white)
        def to_white(W):
            U = Wh_inv @ W
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-30
            return U
        # W_hat is reported in raw coords by pp_dvar; map for dVar eval
        dvar_est  = _per_dir_dvar(Xw, to_white(W_hat),  index_fun)
        dvar_true = _per_dir_dvar(Xw, to_white(W_true), index_fun)
        dvar_pca  = _per_dir_dvar(Xw, to_white(W_pca),  index_fun)
        dvar_ica  = _per_dir_dvar(Xw, to_white(W_ica),  index_fun)
    else:
        dvar_est  = _per_dir_dvar(X, W_hat,  index_fun)
        dvar_true = _per_dir_dvar(X, W_true, index_fun)
        dvar_pca  = _per_dir_dvar(X, W_pca,  index_fun)
        dvar_ica  = _per_dir_dvar(X, W_ica,  index_fun)

    rand_dv = random_dvar_baseline(X, n_rand=n_rand, whitened=whitened_eval,
                                   index_fun=index_fun)

    return dict(
        MSS_dVar    = mss_principal(W_true, W_hat),
        MSS_PCA     = mss_principal(W_true, W_pca),
        MSS_FastICA = mss_principal(W_true, W_ica),
        ortho_defect= orthogonality_defect(W_hat),
        dvar_est    = dvar_est,
        dvar_true   = dvar_true,
        dvar_pca    = dvar_pca,
        dvar_ica    = dvar_ica,
        dvar_rand_mean = float(rand_dv.mean()),
        dvar_rand_max  = float(rand_dv.max()),
    )


def inter_direction_dcor(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Pairwise distance correlation between projections — catches the
    'found same direction twice' failure mode (two columns whose
    projections are highly dCor-dependent).

    ESTIMATOR WARNING.  This is the *biased* V-statistic distance correlation of
    the ``dcor`` package, on the **unsquared** scale.  Two consequences:

    * The bias behaves like a fixed positive offset, so the entries are inflated
      where the true dependence is near zero.  A value of 0.18 here can
      correspond to a bias-corrected 0.02.
    * The supervised side of this package reports the U-statistic ``dCor^2``
      (``dpp.supervised.dcor_optimizer.dcor_u``).  Numbers from the two are on
      different scales and must not be placed in one table or compared.

    The biased estimator is kept because it is what produced the published
    inter-direction columns.  For a bias-corrected, squared value, call
    ``dpp.supervised.dcor_optimizer.dcor_u(Z[:, i], Z[:, j])`` instead.
    """
    k = W.shape[1]
    M = np.zeros((k, k))
    Z = X @ W
    for i in range(k):
        for j in range(k):
            if i == j:
                M[i, j] = 1.0
            elif i < j:
                M[i, j] = M[j, i] = float(dcor.distance_correlation(Z[:, i], Z[:, j]))
    return M
