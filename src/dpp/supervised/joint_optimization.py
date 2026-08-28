"""
joint_optimization.py

Joint optimization over the Stiefel manifold St(p,k).

Objective (all values are the U-statistic, bias-corrected dCor^2 returned by
`dcor_optimizer.dcor_u` -- one scale everywhere):
    max_{B in St(p,k)}  sum_j dCor^2(Y, X beta_j)
                        -  lam * sum_{i<j} dCor^2(X beta_i, X beta_j)

  - Signal term: each column's projection maximizes dependence with Y.
  - Penalty: penalizes inter-direction dCor^2 to discourage redundancy.  The term
    is the independent-component objective of Matteson & Tsay (2017), used here as
    a regulariser on a supervised objective; lam=0.0 (the default) leaves the
    objective unpenalised.  The manifold constraint itself is enforced by the QR
    retraction, not by a penalty.
  - Optimizer: Riemannian Adam with QR retraction.
"""
import numpy as np

from .dcor_optimizer import _make_B, _dcor_and_grad


# ── Penalty gradient ──────────────────────────────────────────────────────────

def _penalty_grad_col(beta_i, beta_j, X):
    """Gradient of dCor^2(Xβ_i, Xβ_j) w.r.t. β_i using analytical formula."""
    z_j = X @ beta_j
    B_j, s_jj = _make_B(z_j)
    dc, g = _dcor_and_grad(beta_i, X, B_j, s_jj)
    return dc, g


# ── Joint optimizer ───────────────────────────────────────────────────────────

def joint_optimize(X, Y, k, lam=0.0, n_restarts=5, max_iter=150,
                   lr=0.05, lr_min=0.005, seed=0, verbose=False):
    """
    Riemannian Adam on St(p,k) = {B in R^{p x k} : B^T B = I_k}.
    Retraction: QR decomposition of (B + α * tangent_gradient).
    """
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    B_Y, s_yy = _make_B(Y)   # cached

    def _obj_grad(B):
        val = 0.0
        grad = np.zeros_like(B)
        for j in range(k):
            dc_j, g_j = _dcor_and_grad(B[:, j], X, B_Y, s_yy)
            val += dc_j
            grad[:, j] += g_j
        if lam > 0:
            for i in range(k):
                for jj in range(i+1, k):
                    dc_ij, g_i = _penalty_grad_col(B[:, i], B[:, jj], X)
                    _,     g_j = _penalty_grad_col(B[:, jj], B[:, i], X)
                    val -= lam * dc_ij
                    grad[:, i] -= lam * g_i
                    grad[:, jj] -= lam * g_j
        return val, grad

    def _run(B0):
        B = B0.copy()
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        M = np.zeros_like(B); V = np.zeros_like(B)
        prev = -np.inf
        for t in range(1, max_iter+1):
            lr_t = lr_min + 0.5*(lr-lr_min)*(1 + np.cos(np.pi*(t-1)/max_iter))
            val, g = _obj_grad(B)
            g_tan = g - B @ (B.T @ g)          # project to tangent space
            M = b1*M + (1-b1)*g_tan
            V = b2*V + (1-b2)*g_tan**2
            Mh = M/(1-b1**t); Vh = V/(1-b2**t)
            step = Mh/(np.sqrt(Vh)+eps_a)
            sn = np.linalg.norm(step)
            if sn > 1: step = step/sn
            B_new = B + lr_t*step
            B, _ = np.linalg.qr(B_new)         # retraction
            if abs(val-prev) < 1e-7 and t > 20: break
            prev = val
        fv, _ = _obj_grad(B)
        return B, fv

    best_B, best_val = None, -np.inf
    restart_vals = []
    for r in range(n_restarts):
        B0, _ = np.linalg.qr(rng.standard_normal((p, k)))
        Br, vr = _run(B0)
        restart_vals.append(vr)
        if vr > best_val:
            best_val, best_B = vr, Br.copy()
        if verbose:
            print(f"    restart {r+1}: val={vr:.4f}", flush=True)

    return best_B, best_val, {
        'restart_vals': restart_vals,
        'spread': max(restart_vals)-min(restart_vals)
    }
