"""dpp — projection pursuit with distance-based dependence measures.

Two subpackages, one for each index:

    dpp.supervised     dCor^2(X beta, Y) maximised over the sphere or St(p,k).
                       Closed-form Riemannian gradient; optional redundancy
                       penalty on the joint search; SIR/SAVE baselines and the
                       Sheng & Yin (2011, 2016) SQP solver for comparison.

    dpp.unsupervised   dVar(X w) maximised over the sphere or St(p,k), by
                       sequential deflation or a joint QR retraction.  Includes
                       the biloop transform (bounded influence function).

Both subpackages ship the simulator the methods were developed against:
`dpp.supervised.data_generator` (single-index / multi-index response models) and
`dpp.unsupervised.generate_data` (latent factor model with a choice of latent
distribution).
"""
from . import supervised, unsupervised

__all__ = ["supervised", "unsupervised"]
__version__ = "1.0.0"
