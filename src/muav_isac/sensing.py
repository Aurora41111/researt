"""Sensing metric: the Cramer-Rao bound on target angle (paper Eq.5).

Eq.5:
    CRB(phi) = sigma^2 / ( 2 |beta|^2 * Tr( A^H A i i^H ) )
             = sigma^2 / ( 2 |beta|^2 * i^H A^H A i )

where  A  is the partial derivative of the sensing signal matrix
C(phi) = sqrt(P_LoS) a a^H + sqrt(P_NLoS) kappa N   w.r.t. the elevation angle
phi, and  beta = sigma_k / (2 d)  (paper's literal form for the round-trip gain).

The paper overloads the symbol "A_{k,n}" for both C and dC/dphi; we keep them
distinct: ``C`` is the matrix, ``A`` is its angle derivative.

``A`` is computed by central finite differences (the closed form involves the
derivative of the steering vector and of sqrt(P_LoS), and is error-prone). The
NLoS matrix is held fixed across the +/- perturbation, which is exactly what
makes the finite difference equal dC/dphi.
"""

from __future__ import annotations

import numpy as np

from . import geometry as geom
from .channel import sensing_matrix_at_phi
from .config import Params


def beta(p: Params, d: float) -> float:
    """Round-trip sensing gain beta_{k,n} = sigma_k / (2 d) (paper Eq.4 text).

    ``sigma_k`` is the radar cross section in m^2 (linear). This literal form is
    kept configurable here so it can be retuned without touching call sites.
    """
    return float(p.sigma_k) / (2.0 * float(d))


def A_matrix(
    p: Params,
    uav_xy: np.ndarray,
    target_xy: np.ndarray,
    H: float,
    nlos_mat: np.ndarray,
    h: float = 1.0e-6,
) -> np.ndarray:
    """Derivative of the sensing signal matrix C w.r.t. elevation angle [rad^-1].

    Central finite difference on C(phi) holding distance and the NLoS draw fixed.
    """
    d = float(geom.distance3d(uav_xy, target_xy, H))
    phi0 = float(geom.elevation_rad(H, d))
    c_plus = sensing_matrix_at_phi(p, phi0 + h, nlos_mat)
    c_minus = sensing_matrix_at_phi(p, phi0 - h, nlos_mat)
    return (c_plus - c_minus) / (2.0 * h)


def fisher(A: np.ndarray, i_beam: np.ndarray) -> float:
    """Fisher-information term  i^H A^H A i  (real, non-negative)."""
    aha = A.conj().T @ A
    return float(np.real(i_beam.conj().T @ aha @ i_beam))


def crb(p: Params, A: np.ndarray, i_beam: np.ndarray, beta_val: float) -> float:
    """Cramer-Rao bound on estimating phi (paper Eq.5), in [rad^2]."""
    f = fisher(A, i_beam)
    if f <= 0.0:
        return float("inf")
    return float(p.sigma2) / (2.0 * abs(beta_val) ** 2 * f)
