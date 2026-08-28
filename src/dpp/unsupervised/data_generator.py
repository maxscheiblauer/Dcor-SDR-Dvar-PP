"""
data_generator.py — factor-model simulator for dVar-PP experiments.

X = Z @ W_true.T * sigma_signal + E
    Z       : n x k latent factors, each column standardised to unit variance
    W_true  : p x k random orthonormal (QR of Gaussian)
    E       : n x p iid N(0, 1)

dependent_factors=True replaces independent latent factors with a coupled
construction (Z2 = Z1 + small_eps and similar), so that marginal
non-Gaussianity is preserved but the factors share higher-order structure.
This is used in Step 4 to probe whether sequential deflation finds
*independent* non-Gaussian directions or only *orthogonal* ones.

Outlier contamination follows the R generator: a fraction `outlier_frac` of
rows is replaced with random unit-direction spikes of magnitude
outlier_magnitude * sqrt(p).
"""
from __future__ import annotations
import numpy as np


DISTRIBUTIONS = ("gaussian_mix", "t3", "uniform", "skewed", "chisq",
                 "gaussian", "ring_mix")

#: Group count for ``dist="ring_mix"``; overridden per call by
#: ``generate_data(..., n_groups=...)``.
RING_GROUPS_DEFAULT = 4


def ring_centres(n_groups: int, radius: float = 2.0) -> np.ndarray:
    """The ``n_groups`` cluster centres of ``dist="ring_mix"``, one row each."""
    ang = 2.0 * np.pi * np.arange(n_groups) / n_groups
    return np.column_stack([np.cos(ang), np.sin(ang)]) * radius


def _draw_factor_columns(rng: np.random.Generator, n: int, k: int, dist: str,
                         n_groups: int = RING_GROUPS_DEFAULT,
                         aux: dict | None = None) -> np.ndarray:
    if dist == "gaussian_mix":
        s = rng.choice([-1.0, 1.0], size=(n, k))
        z = s * 2.0 + rng.standard_normal((n, k))
    elif dist == "gaussian":
        # The control the grid never had: dVar(N(0,1)) = 0.634 is exactly the
        # noise baseline, so the index has no shape gradient to climb and only
        # the variance lift sigma_signal^2 + 1 distinguishes the direction.
        z = rng.standard_normal((n, k))
    elif dist == "ring_mix":
        # A mixture of ``n_groups`` equally weighted components whose centres sit
        # on a circle in the k = 2 latent plane.  This is *not* what
        # ``gaussian_mix`` gives at k = 2: there the two coordinates are
        # independent two-point mixtures, so the four groups sit at the corners
        # of an axis-aligned square and each latent axis splits the same pair
        # twice over.  On a ring the centres do not factorise across coordinates
        # (for n_groups != 4 they cannot), so different directions in the plane
        # separate different pairs of groups, which is the question this
        # construction exists to ask.
        if k != 2:
            raise ValueError("dist='ring_mix' is defined for k = 2 only, got "
                             f"k = {k}.")
        centres = ring_centres(n_groups)
        g = rng.integers(0, n_groups, size=n)
        z = centres[g] + rng.standard_normal((n, 2))
        if aux is not None:
            aux["group"] = g
            aux["centres"] = centres
    elif dist == "t3":
        z = rng.standard_t(df=3, size=(n, k))
    elif dist == "uniform":
        z = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n, k))
    elif dist == "skewed":
        z = rng.standard_exponential(size=(n, k)) - 1.0
    elif dist == "chisq":
        z = rng.chisquare(df=3, size=(n, k)) - 3.0
    else:
        raise ValueError(f"Unknown distribution: {dist!r}; choose from {DISTRIBUTIONS}.")
    # Standardise each column to unit variance so sigma_signal acts as SNR.
    z = z - z.mean(axis=0, keepdims=True)
    z = z / z.std(axis=0, ddof=1, keepdims=True)
    return z


def random_orthonormal(rng: np.random.Generator, p: int, k: int) -> np.ndarray:
    assert k <= p
    Q, _ = np.linalg.qr(rng.standard_normal((p, k)))
    return Q


def _make_dependent_factors(rng: np.random.Generator, n: int, k: int, dist: str) -> np.ndarray:
    """Higher-order-dependent factors with non-Gaussian marginals.

    Construction (heteroscedastic coupling):
        z1 ~ dist                                 (k=1: just z1)
        z_j = |z1| * eps_j   (j >= 2),  eps_j ~ dist iid

    This gives Cov(z1, z_j) = 0 (since E[eps_j]=0 makes z_j conditionally
    mean-zero given z1) but z_j's spread is driven by |z1|, so the joint
    distribution has strong higher-order dependence. Each column is
    standardised to unit variance afterwards. Crucially the column span
    of Z stays rank-k (z_j is not a linear function of z1), unlike the
    naive z_j = z1 + small_eps construction which collapses rank.
    """
    if k == 1:
        return _draw_factor_columns(rng, n, 1, dist)
    base = _draw_factor_columns(rng, n, 1, dist)[:, 0]
    cols = [base]
    abs_base = np.abs(base)
    for _ in range(k - 1):
        eps = _draw_factor_columns(rng, n, 1, dist)[:, 0]
        cols.append(abs_base * eps)
    Z = np.column_stack(cols)
    Z = Z - Z.mean(axis=0, keepdims=True)
    Z = Z / Z.std(axis=0, ddof=1, keepdims=True)
    return Z


def generate_data(
    n: int = 200,
    p: int = 50,
    k: int = 2,
    sigma_signal: float = 2.0,
    dist: str = "gaussian_mix",
    outlier_frac: float = 0.0,
    outlier_magnitude: float = 10.0,
    dependent_factors: bool = False,
    seed: int | None = None,
    n_groups: int = RING_GROUPS_DEFAULT,
) -> dict:
    rng = np.random.default_rng(seed)
    aux: dict = {}
    Z = (_make_dependent_factors(rng, n, k, dist) if dependent_factors
         else _draw_factor_columns(rng, n, k, dist, n_groups=n_groups, aux=aux))
    W_true = random_orthonormal(rng, p, k)
    E = rng.standard_normal((n, p))
    X = Z @ W_true.T * sigma_signal + E

    out_idx = np.array([], dtype=int)
    if outlier_frac > 0.0:
        n_out = int(round(outlier_frac * n))
        if n_out > 0:
            out_idx = rng.choice(n, size=n_out, replace=False)
            d = rng.standard_normal((n_out, p))
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
            sgn = np.sign(rng.standard_normal(n_out))
            X[out_idx, :] = d * outlier_magnitude * np.sqrt(p) * sgn[:, None]

    return dict(X=X, Z=Z, W_true=W_true,
                sigma_signal=sigma_signal, dist=dist,
                outlier_frac=outlier_frac, outlier_idx=out_idx,
                dependent_factors=dependent_factors,
                group=aux.get("group"), centres=aux.get("centres"))
