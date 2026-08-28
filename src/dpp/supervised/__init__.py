"""Supervised projection pursuit: directions maximising distance correlation with a
response.

    max_{beta: ||beta||=1}  dCor^2(X beta, Y)

`dcor_u` is the U-statistic (bias-corrected) estimator of dCor^2 and is the
objective every optimiser in this subpackage maximises.  All values returned are
on the **squared** scale.
"""
from .data_generator import data_generator
from .dcor_optimizer import (dcor_u, check_gradient, optimize_dcor,
                             aggregate_directions)
from .pp_helpers import (seq_pp, seq_pp_auto, deflate_X, deflate_Y,
                         sigma_inv_root, eval_subspace)
from .joint_optimization import joint_optimize
from .sdr_baselines import sir, save
from .sheng_yin import sheng_yin_sdr, dcov_v, dcor_v
from .evaluation import evaluate, principal_angles

__all__ = [
    "data_generator",
    "dcor_u", "check_gradient", "optimize_dcor", "aggregate_directions",
    "seq_pp", "seq_pp_auto", "deflate_X", "deflate_Y", "sigma_inv_root",
    "eval_subspace",
    "joint_optimize",
    "sir", "save",
    "sheng_yin_sdr", "dcov_v", "dcor_v",
    "evaluate", "principal_angles",
]
