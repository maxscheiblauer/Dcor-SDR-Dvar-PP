"""
dvar_optimizer.py — dVar machinery and projection-pursuit optimisers.

Functions
---------
dvar_sq, dvar
    Direct O(n^2) implementations of the squared / unsigned distance variance.
dvar_sq_and_grad_w
    Returns (dvar_sq(Xw), gradient of dvar_sq(Xw) w.r.t. w) using the
    analytical formula derived from A = HDH and ∂D_ij/∂z_k = sign(z_i-z_j)(δ_ik-δ_jk).
biloop
    Leyder–Raymaekers–Rousseeuw (2024) bounded redescender, c=4.
optimise_one_direction
    Adam on the unit sphere with tangent projection, multi-start.
pp_dvar_sequential, pp_dvar_joint
    k-direction projection pursuit. Sequential deflates X via X ← X - (Xw)w^T;
    joint optimises a Stiefel-manifold matrix via QR retraction.
pp_dvar
    Top-level: optional whitening, dispatch to sequential or joint.
"""
from __future__ import annotations
import numpy as np
from joblib import Parallel, delayed


# ----------------------------------------------------------------------- dVar
def _double_centre(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    rm = D.mean(axis=1, keepdims=True)
    cm = D.mean(axis=0, keepdims=True)
    gm = D.mean()
    return D - rm - cm + gm


def dvar_sq(z: np.ndarray) -> float:
    z = np.asarray(z).reshape(-1)
    D = np.abs(z[:, None] - z[None, :])
    A = _double_centre(D)
    return float((A * A).mean())  # = sum/n^2


def dvar(z: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, dvar_sq(z))))


# Reusable (n,n) work buffers keyed by n — re-allocating the D/S/A
# temporaries every call dominates the empirical O(n^2.8) scaling (Task 7).
# Same-process reuse only; joblib process workers each hold their own cache.
_GRAD_BUFS: dict = {}


def _get_grad_bufs(n: int):
    bufs = _GRAD_BUFS.get(n)
    if bufs is None:
        bufs = _GRAD_BUFS[n] = (np.empty((n, n)), np.empty((n, n)))
    return bufs


def dvar_sq_and_grad_w(X: np.ndarray, w: np.ndarray):
    """Analytical gradient of dvar^2(Xw) w.r.t. w.

    Derivation: with z = Xw, D_ij = |z_i - z_j|, A = HDH (H = I - 11'/n),
        dVar^2 = ||A||_F^2 / n^2
        ∂(dVar^2)/∂D_ij = (2/n^2) A_ij
        ∂D_ij/∂z_k    = sign(z_i - z_j) (δ_ik - δ_jk)
    Combining and using A symmetric, S antisymmetric (S_ij = sign(z_i - z_j)):
        ∂(dVar^2)/∂z = (4/n^2) (A ⊙ S) 1
        ∂(dVar^2)/∂w = X^T · ∂(dVar^2)/∂z

    Cost: O(n^2) for v plus O(np) for X^T v — same order as one dVar evaluation
    and ~p× cheaper than central finite differences, which need p forward passes.

    Allocation-lean implementation (Task 7): two cached buffers, in-place
    double-centering, einsum reductions — no math change; values agree with
    the reference to float rounding.
    """
    n = X.shape[0]
    z = X @ w
    buf1, buf2 = _get_grad_bufs(n)
    diff = np.subtract(z[:, None], z[None, :], out=buf1)
    S = np.sign(diff, out=buf2)                     # antisymmetric, zero diag
    A = np.abs(diff, out=buf1)                      # D = |z_i - z_j|
    # double-centre in place (same formula as _double_centre)
    rm = A.mean(axis=1, keepdims=True)
    cm = A.mean(axis=0, keepdims=True)
    gm = A.mean()
    A -= rm
    A -= cm
    A += gm
    dvsq = float(np.einsum('ij,ij->', A, A)) / (n * n)
    v = np.einsum('ij,ij->i', A, S)                 # row sums of A ⊙ S
    grad_w = (4.0 / (n * n)) * (X.T @ v)
    return dvsq, grad_w


# ------------------------------------------------------- multivariate dVar
def dvar_sq_mv(Y: np.ndarray) -> float:
    """Squared multivariate dVar of the rows of Y (n x d).

    Same double-centred convention as dvar_sq, but with Euclidean ROW
    distances D_ij = ||Y_i - Y_j||_2 (exactly what dvar_biloop does for its
    n x 2 case). For d = 1 this coincides with dvar_sq.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    diff = Y[:, None, :] - Y[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=-1))
    A = _double_centre(D)
    return float((A * A).mean())


def dvar_mv(Y: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, dvar_sq_mv(Y))))


def dvar_sq_and_grad_B(X: np.ndarray, B: np.ndarray):
    """Analytical gradient of dvar_sq_mv(X @ B) w.r.t. B (p x k).

    Derivation: with Y = XB, D_ij = ||Y_i - Y_j||_2, A = HDH:
        ∂(dVar²)/∂D_ij = (2/n²) A_ij                     (HAH = A)
        ∂D_ij/∂Y_r     = (δ_ir - δ_jr)(Y_i - Y_j)/D_ij   (D_ij > 0)
        grad_Y[r]      = (4/n²) Σ_j A_rj (Y_r - Y_j)/D_rj
        grad_B         = X^T grad_Y
    For k = 1, (Y_r - Y_j)/D_rj = sign(z_r - z_j), recovering
    dvar_sq_and_grad_w exactly. Zero distances (diagonal, exact ties)
    contribute 0 — same convention as sign(0) = 0 in the 1-D case.
    """
    Y = X @ B
    n = X.shape[0]
    diff = Y[:, None, :] - Y[None, :, :]                # n x n x k
    D = np.sqrt((diff * diff).sum(axis=-1))
    A = _double_centre(D)
    dvsq = float((A * A).mean())
    E = np.divide(A, D, out=np.zeros_like(A), where=D > 0)
    grad_Y = (4.0 / (n * n)) * (E.sum(axis=1)[:, None] * Y - E @ Y)
    return dvsq, X.T @ grad_Y


# -------------------------------------------------------------------- biloop
def biloop(x: np.ndarray, c: float = 4.0) -> np.ndarray:
    """Leyder, Raymaekers, Rousseeuw (2024) bounded redescender.

    Maps each scalar onto a 2D double-loop curve after median/MAD
    standardisation. Extreme outliers map back near the origin (redescending).
    Returns an n x 2 array; downstream we evaluate dvar(biloop(z, c)) treating
    the 2D output as a multivariate observation.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    med = np.median(x)
    mad_ = np.median(np.abs(x - med))               # MAD with constant=1
    if mad_ < np.finfo(float).eps:
        mad_ = x.std() + 1e-12
    xs = (x - med) / mad_
    tt = np.tanh(xs / c)
    two_pi_tt = 2.0 * np.pi * tt
    pos = xs >= 0
    u = np.where(pos,
                 c * (1 + np.cos(two_pi_tt + np.pi)),
                 -c * (1 + np.cos(two_pi_tt - np.pi)))
    v = np.sin(two_pi_tt)
    return np.column_stack([u, v])


def dvar_biloop(z: np.ndarray, c: float = 4.0) -> float:
    """dVar of the 2D biloop transform of z. Direct numpy implementation —
    ~40x faster than going through dcor and avoids the multivariate-dvar
    convention question (we use the standard double-centred Euclidean-distance
    formula on rows of Y).

        dVar^2(Y) = ||double-centre(D)||_F^2 / n^2 ,  D_ij = ||Y_i - Y_j||_2

    biloop maps each scalar onto a 2D point, so Y is n x 2.
    """
    Y = biloop(z, c=c)
    diff = Y[:, None, :] - Y[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=-1))
    A = _double_centre(D)
    return float(np.sqrt(max(0.0, (A * A).mean())))


# ----------------------------------------- index function (objective)
def make_index(index_fun: str = "plain", c: float = 4.0):
    if index_fun == "plain":
        return dvar
    elif index_fun == "robust":
        return lambda z: dvar_biloop(z, c=c)
    raise ValueError(f"index_fun must be 'plain' or 'robust', got {index_fun!r}.")


# -------------------------------------------------- Adam on the sphere
def _adam_step_sphere(w, g, state, lr, beta1, beta2, eps, grad_clip):
    # tangent projection — the critical detail (R header note)
    g = g - (w @ g) * w
    gn = np.linalg.norm(g)
    if gn > grad_clip and gn > 0:
        g = g * (grad_clip / gn)
    state["t"] += 1
    state["m"] = beta1 * state["m"] + (1 - beta1) * g
    state["v"] = beta2 * state["v"] + (1 - beta2) * g * g
    m_hat = state["m"] / (1 - beta1 ** state["t"])
    v_hat = state["v"] / (1 - beta2 ** state["t"])
    w_new = w + lr * m_hat / (np.sqrt(v_hat) + eps)
    return w_new / np.linalg.norm(w_new)


def _numerical_grad_w(X, w, I_fun, eps=1e-4):
    """Central differences for the robust (non-analytical) index."""
    z = X @ w
    g = np.zeros_like(w)
    for i in range(w.size):
        zp = z + eps * X[:, i]
        zm = z - eps * X[:, i]
        g[i] = (I_fun(zp) - I_fun(zm)) / (2.0 * eps)
    return g


# ----------------------------------------- single-direction multi-start
def _run_one_start(X, w_init, index_fun, max_iter, lr, beta1, beta2,
                   eps, grad_clip, tol, finite_diff_eps, sense="max"):
    """One Adam-on-sphere run from a given initialisation.

    ``sense="min"`` descends the same index instead of ascending it. The index
    itself, its gradient and the retraction are untouched: only the sign of the
    step and the direction of the improvement test change, so a minimising run
    is the maximising run on -I. This exists for the study of Section 5.5:
    distance variance places the Gaussian in the middle of its ordering, so the
    shapes that sit *below* the Gaussian baseline are the ones a maximiser is by
    construction searching away from, and only a minimiser can find them.
    """
    if sense not in ("max", "min"):
        raise ValueError(f"sense must be 'max' or 'min', got {sense!r}.")
    sgn = 1.0 if sense == "max" else -1.0
    use_analytical = (index_fun == "plain")
    I_fun = make_index(index_fun)
    w = w_init / np.linalg.norm(w_init)
    state = dict(m=np.zeros_like(w), v=np.zeros_like(w), t=0)

    if use_analytical:
        obj = float(np.sqrt(max(0.0, dvar_sq_and_grad_w(X, w)[0])))
    else:
        obj = float(I_fun(X @ w))

    best_w, best_obj, prev = w.copy(), obj, obj
    trace = np.zeros(max_iter)
    for it in range(max_iter):
        if use_analytical:
            dvsq, g_w = dvar_sq_and_grad_w(X, w)
            # gradient of sqrt(dvsq); avoid division by zero
            scale = 1.0 / (2.0 * np.sqrt(max(dvsq, 1e-30)))
            g = scale * g_w
        else:
            g = _numerical_grad_w(X, w, I_fun, finite_diff_eps)

        w = _adam_step_sphere(w, sgn * g, state, lr, beta1, beta2, eps, grad_clip)
        obj = float(I_fun(X @ w))
        trace[it] = obj
        if sgn * obj > sgn * best_obj:
            best_obj = obj
            best_w = w.copy()
        if it > 10 and abs(obj - prev) < tol:
            trace = trace[: it + 1]
            break
        prev = obj
    return dict(w=best_w, obj=best_obj, trace=trace)


def aggregate_directions(ws, objs, m_top=None):
    """Sign-invariant subspace average of the top-m restart directions:
    leading eigenvector of Σ_i w_i w_iᵀ. Stabilises recovery when several
    near-tied restarts scatter around the same direction."""
    order = np.argsort(np.asarray(objs))[::-1]
    if m_top is not None:
        order = order[:m_top]
    M = np.zeros((ws[0].size, ws[0].size))
    for i in order:
        M += np.outer(ws[i], ws[i])
    _, evecs = np.linalg.eigh(M)
    return evecs[:, -1]


def optimise_one_direction(
    X: np.ndarray,
    index_fun: str = "plain",
    n_starts: int = 20,
    max_iter: int = 300,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    grad_clip: float = 1.0,
    tol: float = 1e-6,
    finite_diff_eps: float = 1e-4,
    init_dirs: np.ndarray | None = None,
    n_jobs: int = -1,
    seed: int = 0,
    aggregate_top_m: int | None = None,
    sense: str = "max",
) -> dict:
    p = X.shape[1]
    rng = np.random.default_rng(seed)

    inits: list[np.ndarray] = []
    if init_dirs is not None:
        init_dirs = np.atleast_2d(init_dirs)
        if init_dirs.shape[0] != p and init_dirs.shape[1] == p:
            init_dirs = init_dirs.T
        for j in range(init_dirs.shape[1]):
            w0 = init_dirs[:, j]
            n0 = np.linalg.norm(w0)
            if n0 > 0:
                inits.append(w0 / n0)
    while len(inits) < n_starts:
        w0 = rng.standard_normal(p)
        inits.append(w0 / np.linalg.norm(w0))

    runs = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_run_one_start)(X, w0, index_fun, max_iter, lr, beta1, beta2,
                                eps, grad_clip, tol, finite_diff_eps, sense)
        for w0 in inits[:n_starts]
    )
    objs = np.array([r["obj"] for r in runs])
    best = int(np.argmax(objs) if sense == "max" else np.argmin(objs))
    out = dict(w=runs[best]["w"], obj=float(objs[best]),
               spread=(float(objs.min()), float(objs.max())),
               traces=[r["trace"] for r in runs],
               best_idx=best)
    if aggregate_top_m:
        if sense == "min":
            raise ValueError("aggregate_top_m ranks restarts by descending "
                             "objective and is not defined for sense='min'.")
        w_agg = aggregate_directions([r["w"] for r in runs], objs,
                                     m_top=aggregate_top_m)
        I_fun = make_index(index_fun)
        out["best_single_w"] = out["w"]
        out["best_single_obj"] = out["obj"]
        out["aggregated"] = True
        out["w"] = w_agg
        out["obj"] = float(I_fun(X @ w_agg))
    return out


# ----------------------------------------- sequential deflation
def deflate(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """X ← X - (X w) w^T   (rank-1 deflation along w in feature space)."""
    return X - np.outer(X @ w, w)


def pp_dvar_sequential(X, k, index_fun="plain", init_W=None, **opt_kw):
    """Sequential dVar-PP by rank-1 deflation.

    Returns, besides the frame and the per-direction objectives, ``spread``: the
    (min, max) objective across restarts for each direction, as
    ``optimise_one_direction`` reports it. It is carried out of the loop because
    Chapter 5 makes a claim about the restart spread and, until 2026-08-11, this
    function discarded the only place it was measured — so the claim rested on a
    column that no code path could fill. ``pp_dvar_joint`` has always returned it.
    """
    p = X.shape[1]
    W = np.zeros((p, k))
    objs = np.zeros(k)
    traces: list = []
    spreads: list = []
    X_cur = X.copy()
    for j in range(k):
        init_dirs = None
        if init_W is not None and j < init_W.shape[1]:
            w0 = init_W[:, j].copy()
            for jj in range(j):                      # orthogonalise warm-start
                w0 = w0 - (W[:, jj] @ w0) * W[:, jj]
            if np.linalg.norm(w0) > 1e-8:
                init_dirs = w0[:, None]
        res = optimise_one_direction(X_cur, index_fun=index_fun,
                                     init_dirs=init_dirs, **opt_kw)
        W[:, j] = res["w"]
        objs[j] = res["obj"]
        traces.append(res["traces"])
        spreads.append(tuple(res["spread"]))
        X_cur = deflate(X_cur, res["w"])
    return dict(W=W, obj=objs, traces=traces, strategy="sequential",
                index_fun=index_fun, spread=spreads)


def pp_dvar_sequential_auto(X, k_max=None, alpha=0.05, n_perm=50,
                            null_starts=4, null_max_iter=150,
                            statistic="normalized",
                            index_fun="plain", init_W=None, seed=0,
                            n_jobs=-1, **opt_kw):
    """Sequential dVar-PP with a Gaussian-null stopping rule — selects k.

    Null-model note: dVar(Xw) is a function of the empirical distribution
    of the projected sample only, so permuting rows of X (or entries of
    the projection) leaves the objective EXACTLY unchanged — a permutation
    null is a no-op for dVar. The hypothesis "no structure left in X_cur"
    is instead "X_cur is Gaussian with the observed mean/covariance":
    we draw n_perm parametric-bootstrap samples
        X_null ~ N(mean(X_cur), Cov(X_cur))
    re-optimise max_w dVar(X_null w) on each (the null must include the
    maximisation over w, else it is anti-conservative), and stop before
    adding the first direction whose observed statistic has
    p-value > alpha, with p = (1 + #{null >= obs}) / (n_perm + 1).

    statistic="normalized" (default) tests T = dVar(Xw*)/std(Xw*) at the
    found direction — scale-free, so it isolates SHAPE non-Gaussianity.
    The raw max-dVar statistic ("raw") has almost no power in the factor
    regime: the covariance-matched null retains the sigma_signal variance
    lift, and raw dVar is dominated by scale (measured: S1 raw p≈0.39,
    normalized p≈0.03 on the same data).

    Returns dict(W, obj, p_values, k_selected, ...). Explicit-k
    `pp_dvar_sequential` is unchanged and remains the default path.
    """
    n, p = X.shape
    if k_max is None:
        k_max = p
    rng = np.random.default_rng(seed + 123456789)
    W = np.zeros((p, k_max))
    objs = np.zeros(k_max)
    pvals = []
    infos = []
    X_cur = X.copy()
    k_sel = 0

    for j in range(k_max):
        init_dirs = None
        if init_W is not None and j < init_W.shape[1]:
            w0 = init_W[:, j].copy()
            for jj in range(j):
                w0 = w0 - (W[:, jj] @ w0) * W[:, jj]
            if np.linalg.norm(w0) > 1e-8:
                init_dirs = w0[:, None]
        res = optimise_one_direction(X_cur, index_fun=index_fun,
                                     init_dirs=init_dirs, n_jobs=n_jobs,
                                     seed=seed + j, **opt_kw)

        def _stat(obj, X_, w_):
            if statistic == "normalized":
                s = (X_ @ w_).std(ddof=1)
                return obj / s if s > 0 else 0.0
            return obj

        T_obs = _stat(res["obj"], X_cur, res["w"])
        mu = X_cur.mean(axis=0)
        C = np.cov(X_cur, rowvar=False)
        null_vals = np.empty(n_perm)
        for b in range(n_perm):
            X_null = rng.multivariate_normal(mu, C, size=n,
                                             method="eigh")
            r_null = optimise_one_direction(X_null, index_fun=index_fun,
                                            n_starts=null_starts,
                                            max_iter=null_max_iter,
                                            n_jobs=n_jobs,
                                            seed=seed + 1000 * (j + 1) + b)
            null_vals[b] = _stat(r_null["obj"], X_null, r_null["w"])
        pval = (1.0 + float((null_vals >= T_obs).sum())) / (n_perm + 1.0)
        info = dict(obj=res["obj"], statistic=T_obs, p_value=pval,
                    null_q=float(np.quantile(null_vals, 1 - alpha)),
                    null_max=float(null_vals.max()),
                    accepted=pval <= alpha)
        pvals.append(pval)
        infos.append(info)
        if not info["accepted"]:
            break

        W[:, j] = res["w"]
        objs[j] = res["obj"]
        k_sel += 1
        X_cur = deflate(X_cur, res["w"])

    return dict(W=W[:, :k_sel], obj=objs[:k_sel], p_values=np.array(pvals),
                k_selected=k_sel, step_info=infos,
                strategy="sequential_auto", index_fun=index_fun, alpha=alpha)


# ----------------------------------------- joint Stiefel via QR retraction
def _stiefel_grad(X, B, index_fun, finite_diff_eps):
    """Sum_j ∂dvar(X B_j)/∂B_j as a (p x k) matrix (column-stacked)."""
    use_analytical = (index_fun == "plain")
    I_fun = make_index(index_fun)
    p, k = B.shape
    G = np.zeros_like(B)
    for j in range(k):
        if use_analytical:
            dvsq, gw = dvar_sq_and_grad_w(X, B[:, j])
            scale = 1.0 / (2.0 * np.sqrt(max(dvsq, 1e-30)))
            G[:, j] = scale * gw
        else:
            G[:, j] = _numerical_grad_w(X, B[:, j], I_fun, finite_diff_eps)
    return G


def _project_stiefel_tangent(B, G):
    """Riemannian gradient on St(p, k): G - B sym(B^T G).
    Tangent vectors satisfy B^T η + η^T B = 0."""
    BtG = B.T @ G
    return G - B @ ((BtG + BtG.T) / 2.0)


def _qr_retract(B, eta, alpha):
    """Retract Y = B + α η back onto St(p, k) via thin QR with sign-fix."""
    Q, R = np.linalg.qr(B + alpha * eta)
    # Stabilise sign so the retraction is continuous near the identity
    sgn = np.sign(np.diag(R))
    sgn[sgn == 0] = 1.0
    return Q * sgn


def _joint_one_start(X, k, init_seed, index_fun, max_iter, lr, beta1, beta2,
                     eps, grad_clip, tol, finite_diff_eps,
                     joint_index="marginal", B_init=None):
    rng = np.random.default_rng(init_seed)
    p = X.shape[1]
    if B_init is not None:
        B, _ = np.linalg.qr(np.asarray(B_init, dtype=float))
    else:
        B, _ = np.linalg.qr(rng.standard_normal((p, k)))
    state = dict(m=np.zeros_like(B), v=np.zeros_like(B), t=0)
    I_fun = make_index(index_fun)

    if joint_index == "multivariate":
        def _obj(B_):
            return dvar_mv(X @ B_)
    else:
        def _obj(B_):
            return sum(I_fun(X @ B_[:, j]) for j in range(k))

    best_B = B.copy()
    best_obj = _obj(B)
    prev = best_obj
    trace = np.zeros(max_iter)
    for it in range(max_iter):
        if joint_index == "multivariate":
            dvsq, G = dvar_sq_and_grad_B(X, B)
            G = G / (2.0 * np.sqrt(max(dvsq, 1e-30)))   # chain rule for sqrt
        else:
            G = _stiefel_grad(X, B, index_fun, finite_diff_eps)
        eta = _project_stiefel_tangent(B, G)
        gn = np.linalg.norm(eta)
        if gn > grad_clip and gn > 0:
            eta = eta * (grad_clip / gn)
        # Adam on the tangent
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * eta
        state["v"] = beta2 * state["v"] + (1 - beta2) * eta * eta
        m_hat = state["m"] / (1 - beta1 ** state["t"])
        v_hat = state["v"] / (1 - beta2 ** state["t"])
        step = m_hat / (np.sqrt(v_hat) + eps)
        # Re-project the Adam-modulated direction back to the tangent space:
        step = _project_stiefel_tangent(B, step)
        B = _qr_retract(B, step, lr)
        obj = _obj(B)
        trace[it] = obj
        if obj > best_obj:
            best_obj = obj
            best_B = B.copy()
        if it > 10 and abs(obj - prev) < tol:
            trace = trace[: it + 1]
            break
        prev = obj
    return dict(B=best_B, obj=best_obj, trace=trace)


def pp_dvar_joint(X, k, index_fun="plain", n_starts=20, max_iter=300,
                  lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8,
                  grad_clip=1.0, tol=1e-6, finite_diff_eps=1e-4,
                  n_jobs=-1, seed=0, joint_index="marginal", init_B=None,
                  **_extra):
    """Joint Stiefel PP.

    joint_index="marginal"      (default) objective Σ_j dVar(X B_j) —
                                cannot see dependence BETWEEN projections.
    joint_index="multivariate"  objective dVar_mv(X B) on Euclidean row
                                distances of the n x k projection — sees
                                joint non-Gaussianity (dependent factors).
                                Requires index_fun="plain" (analytical grad).
    """
    if joint_index not in ("marginal", "multivariate"):
        raise ValueError(f"joint_index must be 'marginal' or 'multivariate', "
                         f"got {joint_index!r}.")
    if joint_index == "multivariate" and index_fun != "plain":
        raise ValueError("joint_index='multivariate' requires index_fun='plain'.")
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_starts).tolist()
    # start 0 uses init_B (if given, e.g. a PCA frame); the rest stay random
    inits = [init_B] + [None] * (n_starts - 1) if init_B is not None \
        else [None] * n_starts
    runs = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_joint_one_start)(X, k, s, index_fun, max_iter, lr, beta1,
                                  beta2, eps, grad_clip, tol, finite_diff_eps,
                                  joint_index, B0)
        for s, B0 in zip(seeds, inits)
    )
    objs = np.array([r["obj"] for r in runs])
    best = int(np.argmax(objs))
    B = runs[best]["B"]
    I_fun = make_index(index_fun)
    per_dir = np.array([I_fun(X @ B[:, j]) for j in range(k)])
    return dict(W=B, obj=per_dir, traces=[r["trace"] for r in runs],
                strategy="joint", index_fun=index_fun,
                joint_index=joint_index, obj_joint=float(objs[best]),
                spread=(float(objs.min()), float(objs.max())))


# -------------------------------------------------------------- whitening
def whiten_svd(X: np.ndarray):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = X.shape[0]
    s_reg = np.maximum(s, 1e-6 * s.max())
    W_white = Vt.T @ np.diag(1.0 / s_reg) * np.sqrt(n - 1)  # p x p
    Xw = Xc @ W_white
    return Xw, W_white


# -------------------------------------------------------------- top-level
def pp_dvar(X, k=2, index_fun="plain", strategy="sequential", whiten=False,
            n_starts=20, max_iter=300, lr=0.05, n_jobs=-1, seed=0,
            init_W=None, joint_index="marginal", **opt_kw):
    """Top-level entry point.

    Default is whiten=False (raw dVar-PP). Whitening was investigated and
    abandoned: in the factor-model regime where sigma_signal lifts the
    signal subspace's variance above the Gaussian noise dimensions, the
    whitening step erases that lift and the optimiser chases spurious
    finite-sample non-Gaussianity in noise dimensions. See report.
    """
    if opt_kw.get("sense", "max") == "min" and strategy != "sequential":
        # pp_dvar_joint swallows unknown keywords through **_extra, so without
        # this the request would be silently ignored and the run would maximise.
        raise ValueError("sense='min' is implemented for the sequential "
                         "single-direction path only.")

    if whiten:
        X_work, W_white = whiten_svd(X)
    else:
        X_work, W_white = X, None

    if strategy == "sequential":
        res = pp_dvar_sequential(X_work, k, index_fun=index_fun,
                                 n_starts=n_starts, max_iter=max_iter, lr=lr,
                                 n_jobs=n_jobs, seed=seed, init_W=init_W,
                                 **opt_kw)
    elif strategy == "joint":
        res = pp_dvar_joint(X_work, k, index_fun=index_fun,
                            n_starts=n_starts, max_iter=max_iter, lr=lr,
                            n_jobs=n_jobs, seed=seed,
                            joint_index=joint_index, init_B=init_W, **opt_kw)
    else:
        raise ValueError(f"strategy must be 'sequential' or 'joint', got {strategy!r}.")

    if whiten:
        # Map directions back: w_obs ∝ W_white @ w_white  (and re-orthonormalise as a frame)
        W_back = W_white @ res["W"]
        Q, _ = np.linalg.qr(W_back)
        res["W_whitened"] = res["W"]
        res["W"] = Q
    res["preprocessing"] = dict(whiten=whiten, index_fun=index_fun,
                                strategy=strategy)
    return res
