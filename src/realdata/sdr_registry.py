"""
sdr_registry.py — one unified interface for every Layer-1 direction-finder
(PLAN_rebuild.md §4/§6).  Each finder exposes `project(X, d) -> (n, d)` mapping
the 8-dim PCA_ours factor block to `d` ordered factor scores; `None` means the
method does not define that `d` (e.g. joint dCor only at d in {2,3}).

Unsupervised finders (PCA, DvarSDR, INPCA) ignore the target and are fit ONCE.
Supervised finders (PLS, SIR, SAVE, DcorSDR) are fit per maturity on (X, y).

Adapters wrap the already-validated thesis optimisers verbatim:
    DvarSDR  -> Dvar-PP/dvar_optimizer.py::pp_dvar   (4-way strategy x index grid)
    DcorSDR  -> PP_Dcor/dcor_optimizer.py::optimize_dcor, pp_helpers::seq_pp,
                joint_optimization::joint_optimize
    SIR/SAVE -> PP_Dcor/sdr_baselines.py::sir, save
    INPCA    -> src/inpca.py::INPCA (new)
"""
from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from dpp.unsupervised.dvar_optimizer import pp_dvar, make_index
from dpp.supervised.dcor_optimizer import optimize_dcor
from dpp.supervised.pp_helpers import seq_pp
from dpp.supervised.joint_optimization import joint_optimize
from dpp.supervised.sdr_baselines import sir, save
from realdata.inpca import INPCA

KMAX = 8
N_RESTARTS = 10
MAX_ITER = 300


# ------------------------------------------------------------- finder types
class Identity:
    """PCA: the factor block itself (already the reduction)."""
    def project(self, X, d):
        return X[:, :d]


class Loading:
    """Fixed nested loading matrix W (p x KMAX); scores = X @ W[:, :d]."""
    def __init__(self, W):
        self.W = W
    def project(self, X, d):
        return X @ self.W[:, :d]


class DcorSeq:
    """Sequential dCor-SDR: d=1 uses the OLS-init single-direction optimum,
    d>=2 uses the deflation sequence (matches legacy step9)."""
    def __init__(self, W_seq, beta1):
        self.W_seq = W_seq; self.beta1 = beta1
    def project(self, X, d):
        W = self.beta1[:, None] if d == 1 else self.W_seq[:, :d]
        return X @ W


class DcorJoint:
    """Joint Stiefel dCor-SDR (penalised); only the fitted d's are defined."""
    def __init__(self, Bs):
        self.Bs = Bs
    def project(self, X, d):
        B = self.Bs.get(d)
        return None if B is None else X @ B


class PLSFinder:
    def __init__(self, pls):
        self.pls = pls
    def project(self, X, d):
        return self.pls.transform(X)[:, :d]


class InpcaFinder:
    def __init__(self, model):
        self.m = model
    def project(self, X, d):
        return self.m.transform(X)[:, :d]


# ------------------------------------------------------------- builders
def _order_by_dvar(Xtr, W, index_fun):
    I = make_index(index_fun)
    score = np.array([I(Xtr @ W[:, j]) for j in range(W.shape[1])])
    return W[:, np.argsort(score)[::-1]]


def build_unsupervised(Xtr, seed=0):
    """PCA, the 4-way DvarSDR grid, and INPCA k*={2,3,4}.  Fit once."""
    finders = {"PCA": Identity()}
    for strat in ("sequential", "joint"):
        for idx in ("plain", "robust"):
            fit = pp_dvar(Xtr, k=KMAX, index_fun=idx, strategy=strat,
                          whiten=False,
                          n_starts=(20 if idx == "plain" else 8),
                          max_iter=(300 if idx == "plain" else 80),
                          n_jobs=-1, seed=seed)
            W = _order_by_dvar(Xtr, fit["W"], idx)
            finders[f"DvarSDR({strat[:3]},{idx})"] = Loading(W)
    for k in (2, 3, 4):
        finders[f"INPCA(k{k})"] = InpcaFinder(INPCA(k_nl=k, seed=seed).fit(Xtr))
    return finders


def build_supervised(Xtr, ytr, seed=0):
    """PLS, SIR, SAVE, and the DcorSDR sub-variants.  Fit per maturity."""
    finders = {}
    pls = PLSRegression(n_components=KMAX, scale=False).fit(Xtr, ytr)
    finders["PLS"] = PLSFinder(pls)
    finders["SIR"] = Loading(sir(Xtr, ytr, k=KMAX, n_slices=10))
    finders["SAVE"] = Loading(save(Xtr, ytr, k=KMAX, n_slices=6))

    Wx, _ = seq_pp(Xtr, ytr, k=KMAX, deflation="X_deflation",
                   n_restarts=N_RESTARTS, max_iter=150, seed=seed)
    b1, _, _ = optimize_dcor(Xtr, ytr, init_method="ols",
                             optimizer="gradient_ascent",
                             n_restarts=N_RESTARTS, max_iter=MAX_ITER, seed=seed)
    finders["DcorSDR(seqX)"] = DcorSeq(Wx, b1)
    Wy, _ = seq_pp(Xtr, ytr, k=KMAX, deflation="Y_residual",
                   n_restarts=N_RESTARTS, max_iter=150, seed=seed)
    finders["DcorSDR(seqY)"] = DcorSeq(Wy, b1)
    for lam in (0.0, 0.5):
        Bs = {}
        for d in (2, 3):
            B, _, _ = joint_optimize(Xtr, ytr, k=d, lam=lam,
                                     n_restarts=4, max_iter=80, seed=seed)
            Bs[d] = B
        finders[f"DcorSDR(joint{lam})"] = DcorJoint(Bs)
    return finders
