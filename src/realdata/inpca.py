"""
inpca.py — Independent Nonlinear Component Analysis (Gunsilius & Schennach 2023,
JASA 118(542)), a from-scratch implementation.  This is the method the whole
strand replicates and that the legacy steps never actually implemented (they
swapped in dVar/dCor *instead* of it).  PLAN_rebuild.md §5.

Algorithm (paper's Theorem 1 + §2.3), with the modernised Brenier-map estimator
(PLAN §5): the paper's bespoke finite-difference Monge-Ampere PDE solver is
replaced by the consistent entropic-OT + barycentric-projection estimator
(Sinkhorn; Pooladian & Niles-Weed 2021), with a smooth kernel-ridge
out-of-sample extension so the map is evaluable and finite-differentiable at
test rows.

    1. Sinkhorn plan between (standardised) input factors and a fixed reference
       standard-Gaussian sample of equal size.
    2. Barycentric projection -> (y_i, x_i) pairs (train only).
    3. Kernel-ridge T_hat: y -> x fit on those pairs (RBF, CV alpha/gamma).
    4. Numerical Jacobian J(y)=dT/dy by symmetric finite differences
       (== the paper's own Theorem-3 estimator).
    5. Symmetrise J (T_hat is not an exact gradient field — disclosed relaxation).
    6. J_bar = mean_i -ln J(y_i)  (matrix log via eigendecomposition);
       eigendecompose -> entropy-ranked nonlinear principal directions.
    7. Hybrid: nonlinear-reshape the top-k block, leave the rest linear.

INPCA is UNSUPERVISED (like PCA): it sees only the factor block, never the
return target — the supervision, if any, is the Layer-2 regression.
"""
from __future__ import annotations
import numpy as np
import ot
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


# ----------------------------------------------------------- Brenier map
def sinkhorn_brenier_map(Yb, reg=0.05, seed=0, ref=None):
    """Entropic-OT barycentric estimate of the Brenier map from the empirical
    distribution of `Yb` (n x d, standardised) to a standard d-Gaussian.

    Returns (pairs_x, ref) where pairs_x[i] is the barycentric image of Yb[i]
    and `ref` is the fixed Gaussian reference sample (reused so the same target
    cloud defines the map every call)."""
    n, d = Yb.shape
    if ref is None:
        rng = np.random.default_rng(seed)
        ref = rng.standard_normal((n, d))
    M = ot.dist(Yb, ref, metric="sqeuclidean")
    M = M / (np.median(M) + 1e-12)                     # scale for a stable reg
    a = np.full(n, 1.0 / n); b = np.full(n, 1.0 / n)
    P = ot.sinkhorn(a, b, M, reg, numItermax=5000, stopThr=1e-9)
    row = P.sum(1, keepdims=True); row[row < 1e-300] = 1e-300
    pairs_x = (P @ ref) / row                          # barycentric projection
    return pairs_x, ref


def fit_transport(Yb, pairs_x, seed=0):
    """Smooth OOS extension T_hat: y -> x by kernel ridge (RBF), CV over
    (alpha, gamma) on the train pairs.  Evaluable and differentiable anywhere."""
    grid = {"alpha": [1e-3, 1e-2, 1e-1, 1.0],
            "gamma": [0.05, 0.1, 0.25, 0.5, 1.0]}
    krr = GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=5,
                       scoring="neg_mean_squared_error")
    krr.fit(Yb, pairs_x)
    return krr.best_estimator_


def jacobian_fd(T_hat, Y, h=1e-3):
    """Symmetric finite-difference Jacobians of T_hat at each row of Y (n x d).
    Returns (n, d, d) with J[i] = dT/dy at Y[i] — the paper's Theorem-3 form."""
    n, d = Y.shape
    J = np.zeros((n, d, d))
    for b in range(d):
        e = np.zeros(d); e[b] = h
        Tp = T_hat.predict(Y + e)
        Tm = T_hat.predict(Y - e)
        J[:, :, b] = (Tp - Tm) / (2.0 * h)
    return J


def entropy_directions(J):
    """J_bar = mean_i -ln(sym(J_i)); return (evals desc, U) with U columns the
    entropy-ranked nonlinear principal directions (Theorem 1)."""
    n, d, _ = J.shape
    Jbar = np.zeros((d, d))
    for i in range(n):
        S = 0.5 * (J[i] + J[i].T)
        w, Q = np.linalg.eigh(S)
        w = np.clip(w, 1e-8, None)                      # PD relaxation
        logS = (Q * np.log(w)) @ Q.T
        Jbar += -logS
    Jbar /= n
    Jbar = 0.5 * (Jbar + Jbar.T)
    w, U = np.linalg.eigh(Jbar)
    order = np.argsort(w)[::-1]                          # largest entropy first
    return w[order], U[:, order]


# ----------------------------------------------------------- estimator
class INPCA:
    """Hybrid INPCA reducer with the sklearn-style fit/transform contract used
    by the SDR registry.  `k_nl` PCA factors are nonlinear-reshaped (entropy
    ordered); the remaining factors are passed through linearly.  For Gaussian
    input the Brenier map is linear and this collapses to PCA (Corollary 1(ii)).
    """
    def __init__(self, k_nl=3, reg=0.05, seed=0):
        self.k_nl = k_nl
        self.reg = reg
        self.seed = seed

    def fit(self, F_train):
        k = min(self.k_nl, F_train.shape[1])
        self.k = k
        B = F_train[:, :k]
        self.mu_ = B.mean(0)
        self.sd_ = B.std(0, ddof=1); self.sd_[self.sd_ < 1e-12] = 1.0
        Bs = (B - self.mu_) / self.sd_
        pairs_x, ref = sinkhorn_brenier_map(Bs, reg=self.reg, seed=self.seed)
        self.T_ = fit_transport(Bs, pairs_x, seed=self.seed)
        J = jacobian_fd(self.T_, Bs)
        self.evals_, self.U_ = entropy_directions(J)
        # freeze train score scale so reshaped columns are unit-variance
        S = self.T_.predict(Bs) @ self.U_
        self.score_sd_ = S.std(0, ddof=1); self.score_sd_[self.score_sd_ < 1e-12] = 1.0
        return self

    def transform(self, F):
        B = (F[:, : self.k] - self.mu_) / self.sd_
        S = (self.T_.predict(B) @ self.U_) / self.score_sd_   # nonlinear scores
        if F.shape[1] > self.k:
            return np.column_stack([S, F[:, self.k:]])         # + linear tail
        return S
