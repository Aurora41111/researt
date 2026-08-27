"""Energy unit tests: transmit energy (Eq.6) and Zeng propulsion model (Eq.7)."""

import numpy as np

from muav_isac import energy as en
from muav_isac.config import default_params


def test_propulsion_hover_matches_induced_plus_profile():
    # V = 0: profile = C1, induced = C0, parasite = climb = 0  -> C0 + C1
    p = default_params()
    P0 = en.propulsion_power(p, V=0.0, ac=0.0)
    assert np.isclose(P0, p.C0 + p.C1)
    assert 850.0 < P0 < 920.0                       # ~887 W for the given constants


def test_propulsion_positive_and_forward_below_hover():
    p = default_params()
    P_hover = en.propulsion_power(p, 0.0, 0.0)
    P_cruise = en.propulsion_power(p, 15.0, 0.0)
    assert P_cruise > 0.0
    # forward flight is more efficient than hover in the Zeng model
    assert P_cruise < P_hover


def test_E_cs_is_tau_times_total_power():
    p = default_params()
    assert np.isclose(en.E_cs(p, 0.3), p.tau * 0.3)


def test_trajectory_flying_energy_shape_and_positive():
    p = default_params()
    U, N = p.U, p.N
    rng = np.random.default_rng(5)
    traj = rng.uniform(0, p.area_m, size=(U, N, 2))
    H = np.full((U, N, 1), 175.0)
    traj = np.concatenate([traj, H], axis=2)
    E = en.trajectory_flying_energy(p, traj)
    assert E.shape == (U, N - 1)
    assert np.all(E > 0.0)
