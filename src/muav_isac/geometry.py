"""Geometric primitives and the probabilistic LoS model (paper Eq.1).

Notation (paper Sec. II):
    o_u[n] = (x_u[n], y_u[n], H_u[n])   3D UAV position at slot n
    l_v    = (x_v, y_v, 0)              ground user position (z = 0)
    l_k    = (x_k, y_k, 0)              ground target position (z = 0)

Eq.1:
    d        = sqrt( ||o_xy - l_xy||^2 + H^2 )           # 3D distance
    phi      = (180/pi) * arcsin( H / d )                 # elevation [deg]
    P_LoS    = 1 / ( 1 + C * exp(-D * (phi - C)) )
    P_NLoS   = 1 - P_LoS
"""

from __future__ import annotations

import numpy as np


def horizontal_distance(uav_xy: np.ndarray, ground_xy: np.ndarray) -> np.ndarray:
    """Ground-plane (2D) distance between a UAV projection and a ground node."""
    ux, uy = uav_xy[..., 0], uav_xy[..., 1]
    gx, gy = ground_xy[..., 0], ground_xy[..., 1]
    return np.hypot(ux - gx, uy - gy)


def distance3d(uav_xy: np.ndarray, ground_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """3D distance between UAV (at altitude ``H``) and a ground node.

    ``uav_xy`` and ``ground_xy`` are (..., 2) arrays; ``H`` broadcasts.
    """
    dh = horizontal_distance(uav_xy, ground_xy)
    return np.sqrt(dh**2 + H**2)


def elevation_rad(H: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Elevation angle [rad] of the UAV as seen from the ground node."""
    return np.arcsin(np.clip(H / d, -1.0, 1.0))


def elevation_deg(H: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Elevation angle [deg] (used inside the LoS-probability formula)."""
    return np.degrees(elevation_rad(H, d))


def p_los(C: float, D: float, elev_deg: np.ndarray) -> np.ndarray:
    """Probability of a LoS link (paper Eq.1). ``elev_deg`` is in degrees."""
    return 1.0 / (1.0 + C * np.exp(-D * (elev_deg - C)))


def p_nlos(C: float, D: float, elev_deg: np.ndarray) -> np.ndarray:
    """Probability of an NLoS link."""
    return 1.0 - p_los(C, D, elev_deg)
