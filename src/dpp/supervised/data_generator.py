"""
data_generator.py

Generate synthetic data for dCor projection pursuit experiments.
True signal lives in the first k standard basis directions (e_1,...,e_k).
Remaining p-k dimensions are pure noise.
"""

import numpy as np


NONLINEARITIES = ['square', 'sine', 'abs', 'cubic', 'tanh', 'product',
                  'sum_squares', 'sine_product']


def data_generator(n, p, k, nonlinearity='square', noise_sd=0.0, seed=42,
                   rho=0.0):
    """
    Parameters
    ----------
    n            : int   — sample size
    p            : int   — ambient dimension
    k            : int   — number of informative directions
    nonlinearity : str   — response function key
    noise_sd     : float — additive noise std on Y
    seed         : int
    rho          : float — predictor correlation. 0 (the default) gives the
                   isotropic X ~ N(0, I_p) every experiment before 2026-08-13
                   used; rho > 0 gives the AR(1) covariance
                   Sigma_ij = rho^|i-j| of Sheng & Yin (2016) and Wu & Chen
                   (2021). The Gaussian draw is unchanged and only rotated by
                   the Cholesky factor, so rho = 0 reproduces earlier runs bit
                   for bit.

    Returns
    -------
    X              : (n, p) array   — predictors
    Y              : (n,) array     — response
    true_directions: (p, k) array   — first k standard basis vectors

    Note that with rho > 0 the informative directions are still e_1, ..., e_k,
    but their projections are correlated with each other and with the noise
    coordinates. That is the case in which weight orthogonality and
    uncorrelated projections part company: B^T B = I no longer implies that the
    columns of XB are uncorrelated.
    """
    rng = np.random.default_rng(seed)

    # X: standard normal, all p dimensions
    X = rng.standard_normal((n, p))
    if rho:
        idx = np.arange(p)
        sigma = rho ** np.abs(idx[:, None] - idx[None, :])
        X = X @ np.linalg.cholesky(sigma).T

    # Informative projections: just the first k columns of X
    Z = X[:, :k]  # (n, k)

    # Response function
    if nonlinearity == 'square':
        # Y = Z1^2;  symmetric → Pearson r → 0
        Y = Z[:, 0] ** 2
    elif nonlinearity == 'sine':
        # Y = sin(2π Z1)
        Y = np.sin(2 * np.pi * Z[:, 0])
    elif nonlinearity == 'abs':
        Y = np.abs(Z[:, 0])
    elif nonlinearity == 'cubic':
        Y = Z[:, 0] ** 3
    elif nonlinearity == 'tanh':
        Y = np.tanh(2 * Z[:, 0])
    elif nonlinearity == 'product':
        # k >= 2: product of first two projections
        if k < 2:
            raise ValueError("'product' requires k >= 2")
        Y = Z[:, 0] * Z[:, 1]
    elif nonlinearity == 'sum_squares':
        Y = np.sum(Z ** 2, axis=1)
    elif nonlinearity == 'sine_product':
        if k < 2:
            raise ValueError("'sine_product' requires k >= 2")
        Y = np.sin(np.pi * Z[:, 0]) * np.cos(np.pi * Z[:, 1])
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}. "
                         f"Choose from {NONLINEARITIES}")

    # Additive noise
    if noise_sd > 0:
        Y = Y + rng.standard_normal(n) * noise_sd

    # True directions: first k standard basis vectors as columns
    true_directions = np.eye(p, k)  # (p, k)

    return X, Y, true_directions
