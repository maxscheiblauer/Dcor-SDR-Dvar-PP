"""
pca.py — two PCA paths on the transformed macro panel.

`pc_T` / `full_sample_factors`  : exact Ludvigson-Ng replica (T x T SVD,
    full-sample standardise) — the Stage-0 validation gate reproduces `Fhat_T`.

`TrainPCA`                      : the leak-free fix (PLAN_rebuild.md §3).  Mean,
    std and eigenvectors are fit on TRAIN rows only and frozen; test factors are
    out-of-sample projections, never refit.  This is `PCA_ours`, the new baseline
    that stands next to (never replaces) the published look-ahead factors.
"""
from __future__ import annotations
import numpy as np


def standardize(Y, ddof=1):
    """Column mean-0 / var-1 with MATLAB's std (ddof=1), matching `standard.m`."""
    mu = Y.mean(0)
    sd = Y.std(0, ddof=ddof)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Y - mu) / sd, mu, sd


def pc_T(Ystd, nfac=8):
    """Bai-Ng static factors, F'F/T = I (Ludvigson-Ng `pc_T.m`).
    yy = Y Y' (T x T); SVD; fhat = U[:, :nfac] * sqrt(T)."""
    T = Ystd.shape[0]
    yy = Ystd @ Ystd.T
    U, s, _ = np.linalg.svd(yy)
    fhat = U[:, :nfac] * np.sqrt(T)
    lam = Ystd.T @ fhat / T
    return fhat, lam, s


def full_sample_factors(Y, nfac=8):
    """Full-sample standardise then pc_T — the exact LN replica for the gate."""
    Ystd, _, _ = standardize(Y)
    fhat, lam, s = pc_T(Ystd, nfac)
    return fhat, lam, s


def _sign_fix(V):
    """Deterministic sign: make each eigenvector's largest-|entry| positive."""
    for j in range(V.shape[1]):
        k = np.argmax(np.abs(V[:, j]))
        if V[k, j] < 0:
            V[:, j] *= -1.0
    return V


class TrainPCA:
    """Train-only PCA (leak-free).  Fit on train rows; project any rows.

    Factors are the projections onto the top-k eigenvectors of the train
    feature covariance, each rescaled to unit variance on the train block so
    the columns are directly comparable to the published unit-variance factors.
    OLS / dCor / dVar are invariant to that per-column scaling, so it changes no
    downstream result — it only makes the factors readable.
    """
    def __init__(self, nfac=8):
        self.nfac = nfac

    def fit(self, Y_train):
        Xs, self.mu_, self.sd_ = standardize(Y_train)
        C = Xs.T @ Xs                         # (p x p), proportional to cov
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1][: self.nfac]
        self.V_ = _sign_fix(V[:, order].copy())      # (p x nfac)
        F_train = Xs @ self.V_
        self.fac_sd_ = F_train.std(0, ddof=1)
        self.fac_sd_ = np.where(self.fac_sd_ < 1e-12, 1.0, self.fac_sd_)
        self.eigvals_ = w[order]
        return self

    def transform(self, Y):
        Xs = (Y - self.mu_) / self.sd_
        return (Xs @ self.V_) / self.fac_sd_
