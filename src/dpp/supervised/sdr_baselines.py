"""
sdr_baselines.py

Classical inverse-regression estimators of the central subspace, used as
sufficient-dimension-reduction (SDR) baselines against the distance-correlation
projection pursuit of this pillar.

Two methods are implemented from scratch (no external SDR dependency), both in
the standard whiten -> slice -> eigendecompose form:

  * SIR  — Sliced Inverse Regression (Li, 1991). A *first-moment* method: it
           looks at how the slice mean E[X | Y in slice] moves with Y. It is
           blind to any dependence for which the inverse-regression mean is
           flat, in particular symmetric nonlinearities (Y = Z^2, |Z|), exactly
           the OLS-blind cases — SIR inherits OLS's blindness here.

  * SAVE — Sliced Average Variance Estimation (Cook & Weisberg, 1991). A
           *second-moment* method: it looks at how the within-slice covariance
           Cov[X | Y in slice] departs from the overall covariance. Because it
           reads a second moment it recovers symmetric structure that SIR (and
           OLS) miss, at the price of needing more samples per slice and the
           usual linearity / constant-covariance conditions on X.

Both return unit-norm direction estimates (p, k) whose column span estimates the
central subspace. For k = 1 the leading column is the single-index direction.

References
----------
Li, K.-C. (1991). Sliced inverse regression for dimension reduction. JASA.
Cook, R. D. & Weisberg, S. (1991). Discussion of Li (1991). JASA.
"""
import numpy as np


# ── whitening ─────────────────────────────────────────────────────────────────

def _whiten(X):
    """
    Standardise the predictors: return Z = (X - x_bar) Sigma^{-1/2}, the inverse
    root Sigma^{-1/2}, and the column mean. Sigma^{-1/2} is formed from the
    symmetric eigendecomposition of the sample covariance, with small eigenvalues
    floored for numerical safety.
    """
    n = X.shape[0]
    x_bar = X.mean(axis=0)
    Xc = X - x_bar
    Sigma = (Xc.T @ Xc) / (n - 1)
    w, U = np.linalg.eigh(Sigma)
    w = np.clip(w, 1e-12, None)
    inv_root = U @ np.diag(1.0 / np.sqrt(w)) @ U.T
    Z = Xc @ inv_root
    return Z, inv_root, x_bar


def _slice_indices(Y, n_slices):
    """Partition sample indices into n_slices contiguous slices of the sorted Y."""
    n = len(Y)
    order = np.argsort(Y, kind="mergesort")
    return [order[s] for s in np.array_split(np.arange(n), n_slices)]


def _back_transform(eigvecs, inv_root):
    """Map whitened-space directions back to predictor space and unit-normalise."""
    B = inv_root @ eigvecs                 # (p, k)
    norms = np.linalg.norm(B, axis=0)
    norms[norms < 1e-12] = 1.0
    return B / norms


# ── SIR ───────────────────────────────────────────────────────────────────────

def sir(X, Y, k=1, n_slices=5):
    """
    Sliced Inverse Regression. Estimate the k leading central-subspace directions
    from the between-slice scatter of the whitened slice means.

    M = sum_h p_h m_h m_h^T,   m_h = mean of Z over slice h,   p_h = n_h / n.
    The top-k eigenvectors of M, back-transformed by Sigma^{-1/2}, estimate the
    directions.
    """
    Z, inv_root, _ = _whiten(np.asarray(X, float))
    p = Z.shape[1]
    n = Z.shape[0]
    M = np.zeros((p, p))
    for idx in _slice_indices(np.asarray(Y, float), n_slices):
        if len(idx) == 0:
            continue
        m_h = Z[idx].mean(axis=0)
        M += (len(idx) / n) * np.outer(m_h, m_h)
    w, V = np.linalg.eigh(M)
    top = V[:, np.argsort(w)[::-1][:k]]
    return _back_transform(top, inv_root)


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save(X, Y, k=1, n_slices=5):
    """
    Sliced Average Variance Estimation. Estimate the k leading central-subspace
    directions from the average squared departure of the within-slice covariance
    from the identity (in whitened coordinates).

    M = sum_h p_h (I - V_h)^2,   V_h = within-slice covariance of Z.
    The top-k eigenvectors of M, back-transformed by Sigma^{-1/2}, estimate the
    directions. Unlike SIR this reads a second moment, so it is sensitive to
    symmetric structure.
    """
    Z, inv_root, _ = _whiten(np.asarray(X, float))
    p = Z.shape[1]
    n = Z.shape[0]
    I = np.eye(p)
    M = np.zeros((p, p))
    for idx in _slice_indices(np.asarray(Y, float), n_slices):
        n_h = len(idx)
        if n_h < 2:
            continue
        V_h = np.cov(Z[idx], rowvar=False)
        diff = I - V_h
        M += (n_h / n) * (diff @ diff)
    w, V = np.linalg.eigh(M)
    top = V[:, np.argsort(w)[::-1][:k]]
    return _back_transform(top, inv_root)
