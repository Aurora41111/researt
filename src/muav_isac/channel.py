"""Channel realizations for communication (Eq.2) and sensing (Eq.4).

Communication channel (paper Eq.2), an M x 1 complex vector:

    h_v[n] = sqrt(P_LoS  * alpha0 * d^-2) * s(phi)
           + sqrt(P_NLoS * alpha0 * d^-2) * g

where  s(phi)  is the ULA steering vector and  g ~ CN(0, I_M).

Sensing "signal matrix" (paper Eq.4), an M x M complex matrix (the echo channel
is beta * C(phi)):

    C(phi) = sqrt(P_LoS)            * a(phi) a(phi)^H
           + sqrt(P_NLoS) * kappa   * N           ,   N random CN(0, I)-like

The NLoS draws are produced by a seeded ``np.random.Generator`` so that every
realization is reproducible. The NLoS matrix ``N`` for a given target is kept
fixed when the elevation angle is perturbed (needed for the CRB derivative in
``sensing.py``).
"""

from __future__ import annotations

import numpy as np

from . import geometry as geom
from .config import Params


def steering_vector(phi_rad: np.ndarray, M: int, d_ant: float, lam: float) -> np.ndarray:
    """ULA steering vector a(phi), shape ``phi_rad.shape + (M,)``.

    a(phi)[m] = exp(-j 2 pi (d_ant/lam) sin(phi) m),  m = 0 .. M-1.
    """
    m = np.arange(M)
    phase = np.asarray(-1j * 2.0 * np.pi * (d_ant / lam) * np.sin(phi_rad))
    return np.exp(phase[..., None] * m)


def comm_channel(
    p: Params, uav_xy: np.ndarray, ground_xy: np.ndarray, H: float, g_nlos: np.ndarray
) -> np.ndarray:
    """Communication channel vector h_v[n] (paper Eq.2), shape (M,).

    ``uav_xy``, ``ground_xy`` are (2,) arrays; ``H`` is the UAV altitude;
    ``g_nlos`` is a fixed CN(0, I_M) draw for this (u, v) link (small-scale
    fading held constant over the flight -> deterministic objective).
    """
    d = float(geom.distance3d(uav_xy, ground_xy, H))
    el_deg = float(geom.elevation_deg(H, d))
    el_rad = float(geom.elevation_rad(H, d))
    plos = float(geom.p_los(p.C, p.D, el_deg))

    a = steering_vector(np.array(el_rad), p.M, p.d_ant, p.lam).ravel()  # (M,)

    coeff = np.sqrt(p.alpha0) / d                                       # sqrt(alpha0 d^-2)
    return np.sqrt(plos) * coeff * a + np.sqrt(1.0 - plos) * coeff * g_nlos


def sensing_channel_matrix(
    p: Params,
    uav_xy: np.ndarray,
    target_xy: np.ndarray,
    H: float,
    nlos_mat: np.ndarray,
) -> np.ndarray:
    """Sensing signal matrix C(phi) (paper Eq.4, without the beta factor), (M, M).

    ``nlos_mat`` is the fixed random draw for this (u, k) link. Pass the same
    matrix to ``sensing_matrix_at_phi`` when forming the angle derivative.
    """
    d = float(geom.distance3d(uav_xy, target_xy, H))
    el_deg = float(geom.elevation_deg(H, d))
    el_rad = float(geom.elevation_rad(H, d))
    plos = float(geom.p_los(p.C, p.D, el_deg))

    a = steering_vector(np.array(el_rad), p.M, p.d_ant, p.lam).ravel()
    los = np.outer(a, a.conj())                                        # a a^H, (M, M)
    return np.sqrt(plos) * los + np.sqrt(1.0 - plos) * p.kappa * nlos_mat


def sensing_matrix_at_phi(
    p: Params, phi_rad: float, nlos_mat: np.ndarray
) -> np.ndarray:
    """C(phi) evaluated at a prescribed elevation angle, keeping ``nlos_mat`` fixed.

    Used by ``sensing.A_matrix`` for the CRB derivative: distance/draw held fixed,
    only the angle varies, exactly as required by the Fisher-information term.
    """
    el_deg = np.degrees(phi_rad)
    plos = float(geom.p_los(p.C, p.D, el_deg))
    a = steering_vector(np.array(phi_rad), p.M, p.d_ant, p.lam).ravel()
    los = np.outer(a, a.conj())
    return np.sqrt(plos) * los + np.sqrt(1.0 - plos) * p.kappa * nlos_mat


# --------------------------------------------------------------------------
# CN(0, I) helpers
# --------------------------------------------------------------------------
def cn0(rng: np.random.Generator, n: int) -> np.ndarray:
    """Length-n complex Gaussian vector, CN(0, I) (unit variance per dim)."""
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)


def cn0_matrix(rng: np.random.Generator, M: int) -> np.ndarray:
    """M x M complex Gaussian matrix with i.i.d. CN(0, 1/sqrt(M))-like entries.

    Normalized so that E[||N||_F^2] = M (consistent with a unit-covariance
    response), matching the paper's "zero mean and unit covariance" description.
    """
    n = (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))) / np.sqrt(2.0)
    return n / np.sqrt(M)
