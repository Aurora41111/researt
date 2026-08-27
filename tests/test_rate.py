"""Rate-primitive unit tests and SDR (sum-of-traces) consistency."""

import numpy as np

from muav_isac import rate as R
from muav_isac.config import default_params


def test_received_power_vec_equals_sdr():
    rng = np.random.default_rng(0)
    M = 3
    h = (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    g = (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    H = np.outer(h, h.conj())
    G = np.outer(g, g.conj())
    assert np.isclose(R.received_power_vec(h, g), R.received_power_sdr(H, G))


def test_interference_vec_matches_manual_sum():
    rng = np.random.default_rng(1)
    M, U = 3, 2
    h_cross = (rng.standard_normal((U, M)) + 1j * rng.standard_normal((U, M)))
    beams = [(0, rng.standard_normal(M) + 1j * rng.standard_normal(M))]
    val = R.interference_vec(h_cross, beams)
    manual = np.abs(np.vdot(h_cross[0], beams[0][1])) ** 2
    assert np.isclose(val, manual)


def test_rate_monotonic_in_signal_and_interference():
    p = default_params()
    r_high_signal = R.rate(signal=1.0, interference=1.0, sigma2=p.sigma2)
    r_low_signal = R.rate(signal=0.1, interference=1.0, sigma2=p.sigma2)
    assert r_high_signal > r_low_signal

    r_high_interf = R.rate(signal=1.0, interference=10.0, sigma2=p.sigma2)
    r_low_interf = R.rate(signal=1.0, interference=0.1, sigma2=p.sigma2)
    assert r_low_interf > r_high_interf


def test_rate_is_log2_one_plus_sinr():
    p = default_params()
    assert np.isclose(R.rate(1.0, 1.0, p.sigma2), np.log2(1 + 1.0 / (1.0 + p.sigma2)))
