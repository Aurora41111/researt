"""All simulation parameters for the Multi-UAV ISAC reproduction.

``Params`` collects the values explicitly given in the paper's Table I and the
values we had to *infer* (the paper omits carrier frequency, bandwidth, noise
power, number of slots, etc.). Every inferred field is tagged with a comment
so it is easy to retune; see ``README.md`` for the rationale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # ------------------------------------------------------------------
    # Topology (Table I)
    # ------------------------------------------------------------------
    U: int = 3                       # number of UAVs
    M: int = 3                       # antennas per UAV (uniform linear array)
    Vu_range: tuple[int, int] = (3, 5)   # users served per UAV, sampled in [lo, hi]
    Ku_range: tuple[int, int] = (2, 4)   # targets sensed per UAV, sampled in [lo, hi]

    # ------------------------------------------------------------------
    # Area / motion (Table I + inferred)
    # ------------------------------------------------------------------
    area_m: float = 500.0            # square area side [m]  (paper: 500 x 500)
    ah_range: tuple[float, float] = (10.0, 20.0)   # horizontal speed [m/s]
    theta_range: tuple[float, float] = (-5.0 * math.pi / 12, 5.0 * math.pi / 12)
    H_range: tuple[float, float] = (150.0, 200.0)  # altitude [m]
    dmin: float = 50.0               # min inter-UAV separation [m] (inferred)

    # ------------------------------------------------------------------
    # Channel (Table I + inferred)
    # ------------------------------------------------------------------
    g0_dB: float = -70.0             # reference path gain at d0 = 1 m [dB]   (Table I: g0)
    C: float = 11.95                 # LoS-probability environment constant  (Table I)
    D: float = 0.136                 # LoS-probability environment constant  (Table I)
    fc: float = 2.0e9                # carrier frequency [Hz]                (inferred)
    d_ant_over_lambda: float = 0.5   # antenna spacing / wavelength          (inferred, λ/2)
    kappa: float = 0.1               # sensing NLoS factor                    (inferred)
    sigma2_dBm: float = -80.0        # AWGN power [dBm]                       (inferred)
    # Common wideband ISAC baseline: -174 dBm/Hz + 10*log10(B~250 MHz) + 10 dB NF
    # ≈ -80 dBm. The paper omits bandwidth/noise; Fig.1(c)'s script defaults to a
    # more conservative -75 dBm calibration to separate the 60/70/80 dBm power
    # curves, while Fig.1(b)'s tight CRB endpoint is better behaved at -80 dBm.

    # ------------------------------------------------------------------
    # Sensing (Table I)
    # ------------------------------------------------------------------
    sigma_k_dBsm: float = -17.0      # radar cross section [dBsm]             (Table I)

    # ------------------------------------------------------------------
    # Power / energy (inferred)
    # ------------------------------------------------------------------
    Pmax_W: float = 1.0e4            # per-slot per-UAV power budget [W] = 70 dBm
    # Paper Fig.1(c) sweeps Pmax at {60, 70, 80} dBm; 70 dBm (the middle value) is
    # the default regime. Pmax is omitted from the paper's Table I, so this is the
    # inferred default. (Fig.1c overrides per curve.)
    Eth_factor: float = 1.2          # E_th = factor * (max-TX + full-speed-fly) energy

    # ------------------------------------------------------------------
    # Time discretization (inferred; paper only says T split into N slots)
    # ------------------------------------------------------------------
    N: int = 60                      # number of time slots
    T: float = 60.0                  # flight period [s]   -> tau = T/N = 1 s

    # ------------------------------------------------------------------
    # DDPG (Table I)
    # ------------------------------------------------------------------
    replay_buf: int = 1600
    batch_size: int = 32

    # ------------------------------------------------------------------
    # UAV propulsion constants (Table I)
    # ------------------------------------------------------------------
    U_tip: float = 120.0             # rotor blade tip speed [m/s]
    psi0: float = 0.6                # fuselage drag ratio
    C0: float = 798.6                # induced power in hover [W]
    C1: float = 88.6                 # blade-profile power in hover [W]
    C2: float = 11.5                 # vertical climb coefficient
    r_tilde: float = 0.005           # rotor solidity
    rho: float = 1.226               # air density [kg/m^3]
    G: float = 0.503                 # rotor disc area [m^2]
    a0: float = 4.3                  # mean induced velocity in hover [m/s]

    # ------------------------------------------------------------------
    # CRB sensing threshold (swept in Fig.1b; inferred)
    # ------------------------------------------------------------------
    Gamma: float = 1.0e-5            # CRB threshold [rad^2]

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = 0

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def c(self) -> float:
        """Speed of light [m/s]."""
        return 2.998e8

    @property
    def lam(self) -> float:
        """Carrier wavelength [m]."""
        return self.c / self.fc

    @property
    def d_ant(self) -> float:
        """Antenna element spacing [m]."""
        return self.d_ant_over_lambda * self.lam

    @property
    def alpha0(self) -> float:
        """Reference path-gain at d0 = 1 m (linear)."""
        return 10.0 ** (self.g0_dB / 10.0)

    @property
    def sigma2(self) -> float:
        """AWGN power [W] from dBm."""
        return 1.0e-3 * 10.0 ** (self.sigma2_dBm / 10.0)

    @property
    def sigma_k(self) -> float:
        """Radar cross section [m^2] from dBsm."""
        return 10.0 ** (self.sigma_k_dBsm / 10.0)

    @property
    def tau(self) -> float:
        """Time-slot duration [s]."""
        return self.T / self.N

    @property
    def amax(self) -> float:
        """Max horizontal speed [m/s] (upper bound of ah_range)."""
        return self.ah_range[1]


def default_params() -> Params:
    """Default parameter set matching Table I + documented inferences."""
    return Params()
