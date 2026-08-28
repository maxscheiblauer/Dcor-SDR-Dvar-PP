"""
evaluation.py

Metrics for assessing β_hat quality against ground truth.
"""

import numpy as np

from .sdr_baselines import sir, save
# U-statistic dCor — same estimator the optimizer selects on (Task 5:
# one estimator everywhere; the dcor package V-statistic differs at small n)
from .dcor_optimizer import dcor_u


def _subspace_angle_1d(beta_hat, true_directions):
    """
    Angle (degrees) between β_hat and the column space of true_directions.
    For k=1 this is just arccos(|β_hat^T true_dir|).
    For k>1 we use the smallest principal angle.
    """
    # Project β_hat onto subspace spanned by true_directions
    Q, _ = np.linalg.qr(true_directions)       # (p, k) orthonormal
    proj = Q @ (Q.T @ beta_hat)
    proj_nrm = np.linalg.norm(proj)
    if proj_nrm < 1e-12:
        return 90.0
    cos_theta = min(1.0, proj_nrm / np.linalg.norm(beta_hat))
    return float(np.degrees(np.arccos(cos_theta)))


def principal_angles(A, B):
    """
    Principal angles (degrees) between column spaces of A and B.
    Uses SVD of Q_A^T Q_B.
    """
    Q_A, _ = np.linalg.qr(A)
    Q_B, _ = np.linalg.qr(B)
    sv = np.linalg.svd(Q_A.T @ Q_B, compute_uv=False)
    sv = np.clip(sv, 0, 1)
    return np.degrees(np.arccos(sv))


def evaluate(beta_hat, true_directions, X, Y, n_slices=5):
    """
    Parameters
    ----------
    beta_hat        : (p,) unit vector — estimated direction
    true_directions : (p, k) — ground truth subspace
    X               : (n, p)
    Y               : (n,)

    n_slices        : int — number of slices for the SIR/SAVE baselines

    Returns
    -------
    dict with keys:
        angle_deg   : angle to true subspace (degrees; 0 = perfect)
        dcor_found  : dCor(Y, X β_hat)
        pearson_r   : |Pearson r(Y, X β_hat)|
        dcor_ols    : dCor(Y, X β_ols) — OLS baseline
        pearson_ols : |Pearson r(Y, X β_ols)| — OLS baseline
        angle_sir   : recovery angle (deg) of the SIR direction — SDR baseline
        dcor_sir    : dCor(Y, X β_sir)
        angle_save  : recovery angle (deg) of the SAVE direction — SDR baseline
        dcor_save   : dCor(Y, X β_save)
    """
    angle = _subspace_angle_1d(beta_hat, true_directions)
    proj = X @ beta_hat

    dc_found = dcor_u(proj, Y)
    r_found = abs(np.corrcoef(proj, Y)[0, 1])

    # OLS baseline
    beta_ols, *_ = np.linalg.lstsq(X, Y, rcond=None)
    nrm = np.linalg.norm(beta_ols)
    if nrm > 1e-12:
        beta_ols = beta_ols / nrm
    proj_ols = X @ beta_ols
    dc_ols = dcor_u(proj_ols, Y)
    r_ols = abs(np.corrcoef(proj_ols, Y)[0, 1])
    angle_ols = _subspace_angle_1d(beta_ols, true_directions)

    # Inverse-regression SDR baselines (first-moment SIR, second-moment SAVE)
    beta_sir = sir(X, Y, k=1, n_slices=n_slices)[:, 0]
    beta_save = save(X, Y, k=1, n_slices=n_slices)[:, 0]
    angle_sir = _subspace_angle_1d(beta_sir, true_directions)
    angle_save = _subspace_angle_1d(beta_save, true_directions)
    dc_sir = dcor_u(X @ beta_sir, Y)
    dc_save = dcor_u(X @ beta_save, Y)

    return {
        'angle_deg':   angle,
        'dcor_found':  dc_found,
        'pearson_r':   r_found,
        'dcor_ols':    dc_ols,
        'pearson_ols': r_ols,
        'angle_ols':   angle_ols,
        'beats_ols':   dc_found > dc_ols,
        'angle_sir':   angle_sir,
        'dcor_sir':    dc_sir,
        'angle_save':  angle_save,
        'dcor_save':   dc_save,
    }


def print_evaluation(metrics):
    print(f"  Angle to true dir  : {metrics['angle_deg']:.2f}°")
    print(f"  dCor (found / OLS) : {metrics['dcor_found']:.4f} / {metrics['dcor_ols']:.4f}"
          f"  {'✓ beats OLS' if metrics['beats_ols'] else '✗ OLS wins'}")
    print(f"  |r|  (found / OLS) : {metrics['pearson_r']:.4f} / {metrics['pearson_ols']:.4f}")
    print(f"  angle (SIR / SAVE) : {metrics['angle_sir']:.2f}° / {metrics['angle_save']:.2f}°")
