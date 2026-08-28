"""Unsupervised projection pursuit: directions maximising distance variance.

    max_{W in St(p,k)}  sum_j dVar(X w_j)

dVar is a dispersion / shape index, not a measure of non-Gaussianity: it is
scale-equivariant, and at unit variance it orders distributions by how far their
mass sits from the centre.  The Gaussian sits in the middle of that ordering
(0.635), with the two-point law at 1.000 above it and Student t_3 at 0.465 below.

`dvar` is the unsquared index, `dvar_sq` its square.  The double-centred (biased)
form is used deliberately: the U-centred version can go negative and the gradient
of the square root is then undefined.
"""
from .data_generator import generate_data, random_orthonormal, ring_centres
from .dvar_optimizer import (dvar, dvar_sq, dvar_mv, dvar_sq_mv,
                             dvar_sq_and_grad_w, dvar_sq_and_grad_B,
                             biloop, dvar_biloop, make_index,
                             optimise_one_direction, deflate,
                             pp_dvar, pp_dvar_sequential, pp_dvar_sequential_auto,
                             pp_dvar_joint, whiten_svd, aggregate_directions)
from .evaluation import (evaluate, mss_principal, angle_deg, pca_directions,
                         fastica_directions, orthogonality_defect,
                         inter_direction_dcor)

__all__ = [
    "generate_data", "random_orthonormal", "ring_centres",
    "dvar", "dvar_sq", "dvar_mv", "dvar_sq_mv",
    "dvar_sq_and_grad_w", "dvar_sq_and_grad_B",
    "biloop", "dvar_biloop", "make_index",
    "optimise_one_direction", "deflate",
    "pp_dvar", "pp_dvar_sequential", "pp_dvar_sequential_auto", "pp_dvar_joint",
    "whiten_svd", "aggregate_directions",
    "evaluate", "mss_principal", "angle_deg", "pca_directions",
    "fastica_directions", "orthogonality_defect", "inter_direction_dcor",
]
