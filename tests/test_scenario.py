"""End-to-end scenario tests: layout, channels, sum-rate, CRB, energy."""

import numpy as np

from muav_isac import scenario as sc
from muav_isac.config import default_params


def _mrt_beams(net, snap, per_beam_power=0.05):
    """Simple maximum-ratio communication beams + unit sensing beams (for tests)."""
    p = net.p
    g_beams, i_beams = {}, {}
    for u in range(p.U):
        for lv, v in enumerate(net.user_of_uav[u]):
            h = snap.h[u, v]                              # (N, M)
            norm = np.linalg.norm(h, axis=1, keepdims=True)
            g_beams[(u, lv)] = (h / norm) * np.sqrt(per_beam_power)
        for lt in range(len(net.target_of_uav[u])):
            i_beams[(u, lt)] = np.ones((p.N, p.M), dtype=complex) * np.sqrt(
                per_beam_power / p.M
            )
    return g_beams, i_beams


def test_network_topology_disjoint_and_complete():
    p = default_params()
    rng = np.random.default_rng(0)
    net = sc.sample_network(p, rng)
    assert sum(net.Vu_list) == net.V
    assert sum(net.Ku_list) == net.K
    # disjoint per-UAV user partition covers all users exactly once
    flat = [v for lst in net.user_of_uav for v in lst]
    assert sorted(flat) == list(range(net.V))
    assert net.serving_uav.shape == (net.V,)


def test_init_trajectory_endpoints_and_shape():
    p = default_params()
    net = sc.sample_network(p, np.random.default_rng(1))
    traj = sc.init_trajectory(net)
    assert traj.shape == (p.U, p.N, 3)
    assert np.allclose(traj[:, 0, :2], net.uav_start)
    assert np.allclose(traj[:, -1, :2], net.uav_end)


def test_compute_channels_shapes():
    p = default_params()
    net = sc.sample_network(p, np.random.default_rng(2))
    traj = sc.init_trajectory(net)
    snap = sc.compute_channels(net, traj)
    assert snap.h.shape == (p.U, net.V, p.N, p.M)
    assert snap.C.shape == (p.U, net.K, p.N, p.M, p.M)
    assert snap.A.shape == snap.C.shape
    assert snap.beta.shape == (p.U, net.K, p.N)
    assert np.all(np.isfinite(snap.beta)) and np.all(snap.beta > 0)


def test_sum_rate_crb_energy_finite_and_positive():
    p = default_params()
    net = sc.sample_network(p, np.random.default_rng(7))
    traj = sc.init_trajectory(net)
    snap = sc.compute_channels(net, traj)
    g_beams, i_beams = _mrt_beams(net, snap)

    sr = sc.sum_rate(net, snap, g_beams, i_beams)
    assert np.isfinite(sr) and sr > 0.0

    crbs = sc.crb_table(net, snap, i_beams)
    for arr in crbs.values():
        assert np.all(np.isfinite(arr)) and np.all(arr > 0.0)

    E = sc.uav_energy(net, snap, g_beams, i_beams)
    assert np.all(E["cs"] > 0.0)
    assert np.all(E["fly"] > 0.0)


def test_closer_trajectory_yields_higher_rate():
    # Move the UAV straight at users (mid area) vs a far straight line; rate should rise.
    p = default_params()
    net = sc.sample_network(p, np.random.default_rng(9))
    traj_far = sc.init_trajectory(net)                       # corner to corner
    snap_far = sc.compute_channels(net, traj_far)
    g_f, i_f = _mrt_beams(net, snap_far)
    sr_far = sc.sum_rate(net, snap_far, g_f, i_f)

    # hover near the centroid of the users
    centroid = net.user_xy.mean(axis=0)
    traj_near = np.empty((p.U, p.N, 3))
    for u in range(p.U):
        traj_near[u, :, :2] = centroid + (net.uav_start[u] - centroid) * 0.1
        traj_near[u, :, 2] = p.H_range[0]                    # lower altitude -> closer
    snap_near = sc.compute_channels(net, traj_near)
    g_n, i_n = _mrt_beams(net, snap_near)
    sr_near = sc.sum_rate(net, snap_near, g_n, i_n)
    assert sr_near > sr_far


def test_reproducibility_same_seed():
    p = default_params()
    net1 = sc.sample_network(p, np.random.default_rng(42))
    net2 = sc.sample_network(p, np.random.default_rng(42))
    assert np.allclose(net1.user_xy, net2.user_xy)
    assert np.allclose(net1.g_nlos, net2.g_nlos)
