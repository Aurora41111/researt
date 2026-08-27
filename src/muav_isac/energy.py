"""UAV energy consumption: transmit energy (Eq.6) and propulsion energy (Eq.7).

Eq.6 (communication + sensing transmit energy per slot):
    E_cs[n] = tau * ( sum_v ||g_v[n]||^2 + sum_k ||i_k[n]||^2 )
            = tau * ( sum_v Tr(G_v[n]) + sum_k Tr(I_k[n]) )    in SDR form

Eq.7 (propulsion / flying energy). The paper's printed formula is garbled and
its label of C0/C1 against the terms contradicts the standard Zeng & Zhang
model. We follow the physics-consistent reading (also matching the paper's
*prose* "C0 = induced, C1 = blade profile"):

    P(V, a_c) = C1 (1 + 3 V^2 / U_tip^2)                          # blade profile
              + C0 ( sqrt(1 + V^4/(4 a0^4)) - V^2/(2 a0^2) )^0.5    # induced
              + 0.5 * psi0 * rho * r_tilde * G * V^3               # parasite drag
              + C2 * |a_c|                                         # vertical climb

    E_fly[n] = tau * P(V, a_c),   V = horizontal speed, a_c = |dH|/tau.

At hover (V = 0): profile = C1, induced = C0, giving the standard ~887 W.
"""

from __future__ import annotations

import numpy as np

from .config import Params


def propulsion_power(p: Params, V: float, ac: float) -> float:
    """Instantaneous propulsion power [W] for horizontal speed ``V`` and climb ``ac``."""
    V = max(float(V), 0.0)
    profile = p.C1 * (1.0 + 3.0 * V**2 / p.U_tip**2)
    induced_inner = np.sqrt(1.0 + V**4 / (4.0 * p.a0**4)) - V**2 / (2.0 * p.a0**2)
    induced_inner = max(induced_inner, 0.0)          # guard tiny negative round-off
    induced = p.C0 * np.sqrt(induced_inner)
    parasite = 0.5 * p.psi0 * p.rho * p.r_tilde * p.G * V**3
    climb = p.C2 * abs(ac)
    return float(profile + induced + parasite + climb)


def E_fly(p: Params, V: float, ac: float) -> float:
    """Flying energy of one slot (paper Eq.7)."""
    return p.tau * propulsion_power(p, V, ac)


def E_cs(p: Params, total_tx_power: float) -> float:
    """Communication + sensing transmit energy of one slot (paper Eq.6).

    ``total_tx_power`` = sum_v ||g_v||^2 + sum_k ||i_k||^2  (W).
    """
    return p.tau * float(total_tx_power)


def trajectory_flying_energy(
    p: Params, traj_xyz: np.ndarray
) -> np.ndarray:
    """Per-slot flying energy for every UAV.

    ``traj_xyz`` has shape (U, N, 3) with columns [x, y, H]. Returns (U, N-1):
    the energy over each consecutive slot transition. The horizontal speed uses
    the 3D segment length (the paper bounds total displacement), and the climb
    rate uses the altitude change.
    """
    seg = np.diff(traj_xyz, axis=1)                  # (U, N-1, 3)
    horiz = np.hypot(seg[..., 0], seg[..., 1])       # horizontal displacement
    dh = np.abs(seg[..., 2])                         # |altitude change|
    V = horiz / p.tau                                # horizontal speed
    ac = dh / p.tau                                  # climb rate
    return p.tau * (
        p.C1 * (1.0 + 3.0 * V**2 / p.U_tip**2)
        + p.C0
        * np.sqrt(
            np.clip(
                np.sqrt(1.0 + V**4 / (4.0 * p.a0**4)) - V**2 / (2.0 * p.a0**2), 0.0, None
            )
        )
        + 0.5 * p.psi0 * p.rho * p.r_tilde * p.G * V**3
        + p.C2 * ac
    )
