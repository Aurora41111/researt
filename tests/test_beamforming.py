"""Beamforming FP tests: Algorithm 1 (comm) and Algorithm 2 (sensing).

Kept small (N=4, few FP iters, loose CRB) so the suite runs in a few seconds.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")  # SCS "may be inaccurate" is expected at loose eps

from muav_isac import scenario as sc
from muav_isac.beamforming import comm, rank1
from muav_isac.beamforming import sensing as sbf
from muav_isac.config import default_params

# Use the real config defaults (Pmax=70 dBm, common wideband sigma2, Gamma=1e-5); only N
# is shrunk for speed. Tests should exercise the actual physical regime.
PV = dataclasses.replace(default_params(), N=4)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(7)
    net = sc.sample_network(PV, rng)
    snap = sc.compute_channels(net, sc.init_trajectory(net))
    # CRB-feasible sensing init (matches the real BCD flow): with shared per-slot
    # power, comm must see the sensing beams so it reserves their power instead of
    # hogging the full Pmax and starving sensing. Zero init makes sensing fail.
    from muav_isac.bcd import _crb_feasible_beams
    i0 = _crb_feasible_beams(net, snap)
    return net, snap, i0


def test_comm_fp_monotone_and_improves_over_mrt(setup):
    net, snap, i0 = setup
    _, g, hist = comm.comm_beamforming(net, snap, i0, max_iter=5)
    assert len(hist) >= 2
    # FP objective is non-decreasing in exact arithmetic; allow a small RELATIVE
    # dip for SCS solver noise (eps=1e-3), scaled to the objective magnitude.
    assert np.all(np.diff(hist) >= -1e-4 * np.maximum(1.0, np.abs(hist[:-1])))

    # MRT baseline (equal-power along serving channel)
    gm = {}
    for u in range(net.p.U):
        for lv, v in enumerate(net.user_of_uav[u]):
            h = snap.h[u, v]
            gm[(u, lv)] = (h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)) * np.sqrt(
                net.p.Pmax_W / net.Vu_list[u]
            )
    assert sc.sum_rate(net, snap, g, i0) > sc.sum_rate(net, snap, gm, i0)


def test_sensing_fp_converges(setup):
    net, snap, i0 = setup
    _, g, _ = comm.comm_beamforming(net, snap, i0, max_iter=4)
    _, _, hist = sbf.sensing_beamforming(net, snap, g, max_iter=8)
    assert len(hist) >= 2
    # The sensing FP uses the upper-bound approximation (paper Eq.15) on top of
    # the quadratic transform, so -- unlike the comm FP -- its TRUE objective is
    # not strictly monotone from an arbitrary init (it can settle to a local
    # point below a good CRB-feasible start; the BCD keeps the better of the two
    # via best-tracking). What is guaranteed: it converges (last two within tol).
    assert abs(hist[-1] - hist[-2]) <= 1e-2 * max(1.0, abs(hist[-2]))


def test_sensing_crb_satisfied(setup):
    net, snap, i0 = setup
    _, g, _ = comm.comm_beamforming(net, snap, i0, max_iter=4)
    _, i_beams, _ = sbf.sensing_beamforming(net, snap, g, max_iter=5)
    crbs = sc.crb_table(net, snap, i_beams)
    # allow a small solver-tolerance overshoot
    violated = sum(int((arr > net.p.Gamma * 1.10).any()) for arr in crbs.values())
    assert violated == 0


def test_rank1_preserves_power():
    rng = np.random.default_rng(2)
    M = 3
    x = (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    G = np.outer(x, x.conj())                              # exactly rank-one PSD
    g = rank1.recover_beam(G)
    assert np.isclose((np.abs(g) ** 2).sum(), np.real(np.trace(G)), atol=1e-8)


def test_rank1_handles_indefinite():
    # solver-like slightly indefinite matrix -> recovery still valid & finite
    rng = np.random.default_rng(3)
    G = np.outer(rng.standard_normal(3) + 1j * rng.standard_normal(3),
                 rng.standard_normal(3) + 1j * rng.standard_normal(3))
    G = (G + G.conj().T) / 2 + np.diag([-1e-3, 0, 0])
    g = rank1.recover_beam(G)
    assert np.all(np.isfinite(g))


def test_end_to_end_comm_then_sensing(setup):
    net, snap, i0 = setup
    _, g, _ = comm.comm_beamforming(net, snap, i0, max_iter=4)
    _, i_beams, _ = sbf.sensing_beamforming(net, snap, g, max_iter=4)
    sr = sc.sum_rate(net, snap, g, i_beams)
    assert np.isfinite(sr) and sr > 0.0
