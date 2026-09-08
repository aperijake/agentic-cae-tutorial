"""Isotropic linear elastic material.

Kept in its own module so that the element formulations do not have to import
the mesh-file machinery just to get an elasticity matrix. That separation is
what lets the Cook's membrane path run on numpy and scipy alone.
"""

import numpy as np


def elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """Isotropic elasticity matrix, Voigt order [xx yy zz xy yz xz].

    Uses engineering shear strain, so the shear rows carry mu rather than
    2*mu. Note that lambda blows up as nu approaches 0.5 -- that divergence is
    the mechanism behind volumetric locking, not a numerical accident.
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2.0 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    return D
