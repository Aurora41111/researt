"""Achievable rate primitives (paper Eq.3) in both beam-vector and SDR form.

For user v served by UAV u:

    signal      = |h_v^u^H g_v^u|^2                       # beam-vector form
                = Tr(H_v^u G_v^u)                         # SDR form, H=h h^H, G=g g^H

    Theta_v     = sum of |h_v^{u'}^H beam|^2 over every OTHER transmitted beam
                  (comm beams of all UAVs except the serving one, and all sensing
                  beams), plus sigma^2.  The paper's P1 uses this *sum-of-powers*
                  form (linear in the beam covariances), which is what makes the
                  SDR convex; we keep it consistently for optimization and for
                  reporting.

    R_v[n]      = tau * log2( 1 + signal / Theta_v )
"""

from __future__ import annotations

import numpy as np


def received_power_vec(h: np.ndarray, beam: np.ndarray) -> float:
    """|h^H beam|^2 for M-vectors."""
    proj = np.vdot(h, beam)  # h^H beam
    return float(np.abs(proj) ** 2)


def received_power_sdr(H: np.ndarray, G: np.ndarray) -> float:
    """Tr(H G) for the M x M covariances H = h h^H, G = g g^H."""
    return float(np.real(np.trace(H @ G)))


def interference_vec(h_cross: np.ndarray, beams: list[tuple[int, np.ndarray]]) -> float:
    """Sum of |h_cross[u]^H beam|^2 over a list of ``(uav_idx, beam)`` pairs.

    ``h_cross`` has shape (U, M): the channel from every UAV to the user of
    interest. ``beams`` excludes the user's own serving beam.
    """
    total = 0.0
    for uu, b in beams:
        total += float(np.abs(np.vdot(h_cross[uu], b)) ** 2)
    return total


def interference_sdr(H_cross: np.ndarray, beams_sdr: list[tuple[int, np.ndarray]]) -> float:
    """SDR counterpart: sum of Tr(H_cross[u] B)."""
    total = 0.0
    for uu, B in beams_sdr:
        total += float(np.real(np.trace(H_cross[uu] @ B)))
    return total


def rate(signal: float, interference: float, sigma2: float) -> float:
    """log2(1 + signal / (interference + sigma2)) per slot (without tau)."""
    return float(np.log2(1.0 + signal / (interference + sigma2)))


def cov(v: np.ndarray) -> np.ndarray:
    """Outer product v v^H (rank-one covariance)."""
    return np.outer(v, v.conj())
