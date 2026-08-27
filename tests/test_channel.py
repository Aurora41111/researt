"""Channel / steering-vector unit tests."""

import numpy as np

from muav_isac import channel as ch
from muav_isac.config import default_params


def test_steering_vector_unit_modulus_and_first_element():
    p = default_params()
    a = ch.steering_vector(np.array(0.37), p.M, p.d_ant, p.lam).ravel()
    assert np.isclose(a[0], 1.0)
    assert np.allclose(np.abs(a), 1.0)


def test_steering_vector_phase_progression():
    p = default_params()
    phi = 0.21
    a = ch.steering_vector(np.array(phi), p.M, p.d_ant, p.lam).ravel()
    expected_step = np.exp(-1j * 2.0 * np.pi * (p.d_ant / p.lam) * np.sin(phi))
    assert np.isclose(a[1] / a[0], expected_step)
    assert np.isclose(a[2] / a[1], expected_step)


def test_comm_channel_gain_grows_as_uav_approaches():
    # High elevation -> plos ~ 1, so |h|^2 ~ alpha0 * M / d^2 (LoS-dominated).
    p = default_params()
    rng = np.random.default_rng(0)
    g_nlos = ch.cn0(rng, p.M)
    ground = np.array([250.0, 250.0])
    h_far = ch.comm_channel(p, np.array([450.0, 450.0]), ground, 180.0, g_nlos)
    h_near = ch.comm_channel(p, np.array([252.0, 250.0]), ground, 180.0, g_nlos)
    assert np.abs(h_near).sum() ** 2 > 0  # sanity
    # nearer 3D distance -> strictly larger channel power
    from muav_isac import geometry as geom

    d_far = geom.distance3d(np.array([450.0, 450.0]), ground, 180.0)
    d_near = geom.distance3d(np.array([252.0, 250.0]), ground, 180.0)
    assert d_near < d_far
    assert (np.abs(h_near) ** 2).sum() > (np.abs(h_far) ** 2).sum()


def test_sensing_matrix_shape_and_rank1_los():
    p = default_params()
    rng = np.random.default_rng(1)
    uav = np.array([200.0, 200.0])
    tgt = np.array([210.0, 205.0])
    nlos_mat = ch.cn0_matrix(rng, p.M)
    C = ch.sensing_channel_matrix(p, uav, tgt, 180.0, nlos_mat)
    assert C.shape == (p.M, p.M)
    # at high elevation the LoS term dominates -> matrix near rank-1
    sv = np.linalg.svd(np.abs(C), compute_uv=False)
    assert sv[0] / (sv.sum() + 1e-12) > 0.5
