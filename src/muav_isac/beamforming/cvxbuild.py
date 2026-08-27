"""Precompute the constant matrices that appear in the SDR subproblems.

* ``HH[u, v, n]``  = h[u, v, n] h[u, v, n]^H   (M, M)   -- channel covariance
  from UAV ``u`` to user ``v`` at slot ``n``. Used both as the *serving* channel
  (u = serving UAV) and as the *interference* channel (u = any other UAV).
* ``AtA[u, k, n]`` = A[u, k, n]^H A[u, k, n]   (M, M)   -- enters the CRB
  constraint Tr(AtA I_k) >= sigma^2 / (2 Gamma |beta|^2)   (paper Eq.5 / 9b).

Derived from a :class:`scenario.ChannelSnapshot`; recomputed when the trajectory
changes.
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from ..scenario import ChannelSnapshot


def channel_covariances(snap: ChannelSnapshot) -> np.ndarray:
    """HH[u, v, n] = h h^H, shape (U, V, N, M, M)."""
    h = snap.h                                              # (U, V, N, M)
    return h[..., :, None] * np.conj(h[..., None, :])


def sensing_fisher_matrices(snap: ChannelSnapshot) -> np.ndarray:
    """AtA[u, k, n] = A^H A, shape (U, K, N, M, M)."""
    A = snap.A                                              # (U, K, N, M, M)
    return np.conj(A.transpose(0, 1, 2, 4, 3)) @ A


def crb_thresholds(snap: ChannelSnapshot, p: Params) -> np.ndarray:
    """Per (u, k, n) RHS of (9b): sigma^2 / (2 Gamma |beta|^2). Shape (U, K, N)."""
    return p.sigma2 / (2.0 * p.Gamma * np.abs(snap.beta) ** 2)
