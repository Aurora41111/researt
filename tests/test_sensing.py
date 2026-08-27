"""Sensing unit tests: beta, the A = dC/dphi derivative, and the CRB (Eq.5)."""

import numpy as np

from muav_isac import channel as ch
from muav_isac import geometry as geom
from muav_isac import sensing as sens
from muav_isac.config import default_params


def _analytic_A(p, phi_rad, nlos_mat):
    """Closed-form dC/dphi_rad for cross-checking the numerical derivative."""
    phi_deg = np.degrees(phi_rad)
    P = float(geom.p_los(p.C, p.D, phi_deg))
    u = p.C * np.exp(-p.D * (phi_deg - p.C))
    dP_ddeg = p.D * u / (1.0 + u) ** 2
    dP_drad = dP_ddeg * (180.0 / np.pi)            # chain rule: phi_deg = (180/pi) phi_rad

    a = ch.steering_vector(np.array(phi_rad), p.M, p.d_ant, p.lam).ravel()
    m = np.arange(p.M)
    da = -1j * 2.0 * np.pi * (p.d_ant / p.lam) * np.cos(phi_rad) * m * a

    sqrtP, sqrtN = np.sqrt(P), np.sqrt(1.0 - P)
    aa = np.outer(a, a.conj())
    d_los = (dP_drad / (2.0 * sqrtP)) * aa + sqrtP * (
        np.outer(da, a.conj()) + np.outer(a, da.conj())
    )
    d_nlos = (-dP_drad / (2.0 * sqrtN)) * p.kappa * nlos_mat
    return d_los + d_nlos


def test_A_numerical_matches_analytic():
    p = default_params()
    rng = np.random.default_rng(3)
    uav = np.array([180.0, 180.0])
    tgt = np.array([200.0, 240.0])
    nlos_mat = ch.cn0_matrix(rng, p.M)
    A_num = sens.A_matrix(p, uav, tgt, 175.0, nlos_mat, h=1e-6)

    d = float(geom.distance3d(uav, tgt, 175.0))
    phi0 = float(geom.elevation_rad(175.0, d))
    A_ana = _analytic_A(p, phi0, nlos_mat)

    assert np.allclose(A_num, A_ana, atol=1e-3)


def test_beta_decreases_with_distance():
    p = default_params()
    assert sens.beta(p, 100.0) > sens.beta(p, 300.0)
    assert np.isclose(sens.beta(p, 200.0), p.sigma_k / 400.0)


def test_crb_decreases_with_sensing_power():
    # CRB ~ sigma^2 / (2 |beta|^2 * i^H A^H A i): more sensing power -> smaller CRB.
    # (CRB is NOT monotonic in distance alone because the steering-vector derivative
    # carries a cos(phi) factor, so near-broadside targets discriminate poorly.)
    p = default_params()
    rng = np.random.default_rng(4)
    uav = np.array([200.0, 200.0])
    tgt = np.array([330.0, 260.0])                            # moderate elevation ~40 deg
    nlos_mat = ch.cn0_matrix(rng, p.M)
    A = sens.A_matrix(p, uav, tgt, 180.0, nlos_mat)
    d = float(geom.distance3d(uav, tgt, 180.0))
    b = sens.beta(p, d)

    i_weak = np.zeros(p.M, dtype=complex)
    i_weak[0] = 0.05
    i_strong = i_weak.copy()
    i_strong[0] = 0.5
    assert sens.crb(p, A, i_strong, b) < sens.crb(p, A, i_weak, b)
    assert sens.crb(p, A, i_strong, b) > 0.0
