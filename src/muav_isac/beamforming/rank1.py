"""Recover a rank-one beam vector from a (possibly higher-rank) SDR solution.

After the FP/SDR solvers return a PSD ``G``, the optimal beam is its principal
eigenvector scaled by sqrt(lambda_max). When the relaxation is not exactly
rank-one we fall back to Gaussian randomization: sample candidate vectors
g ~ CN(0, G) and keep the one that best preserves Tr(H G) / constraints.
"""

from __future__ import annotations

import numpy as np


def recover_beam(G: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Principal-eigenvector rank-one recovery that preserves total power.

    g = sqrt(max(Re Tr(G), 0)) * v_max, so the recovered beam carries the same
    total power as the SDR solution. This matters when the solver returns a
    slightly indefinite G (small negative eigenvalues): using ``sqrt(Tr)`` rather
    than ``sqrt(lambda_max)`` avoids inflating the beam power above the budget.
    """
    Gp = (G + G.conj().T) / 2.0
    _, V = np.linalg.eigh(Gp)
    vmax = V[:, -1]
    power = max(float(np.real(np.trace(Gp))), 0.0)
    return np.sqrt(power) * vmax


def best_rank1_by_trace(
    G: np.ndarray, H: np.ndarray, n_samples: int = 64, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Gaussian randomization keeping the unit-power g maximizing Tr(H g g^H).

    Returns a beam with the SAME total power as Tr(G) and direction chosen to
    maximize the signal Tr(H G) ~= g^H H g.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    G = (G + G.conj().T) / 2.0
    # project onto PSD (solvers sometimes return slightly indefinite matrices)
    w, V = np.linalg.eigh(G)
    w_clip = np.maximum(w, 0.0)
    Gp = (V * w_clip) @ V.conj().T
    power = float(np.real(np.trace(Gp)))
    if power <= 0:
        return np.zeros(G.shape[0], dtype=complex)
    L = np.linalg.cholesky(Gp + 1e-10 * np.eye(G.shape[0]))
    best_g, best_val = None, -np.inf
    for _ in range(n_samples):
        x = (rng.standard_normal(G.shape[0]) + 1j * rng.standard_normal(G.shape[0])) / np.sqrt(2)
        g = L @ x
        val = float(np.real(np.vdot(g, H @ g)))
        if val > best_val:
            best_val, best_g = val, g
    # rescale to the original trace power
    return best_g * np.sqrt(power / (np.abs(best_g) ** 2).sum())


def rank_ratio(G: np.ndarray) -> float:
    """Top eigenvalue / sum of eigenvalues (1.0 means exactly rank one)."""
    w = np.linalg.eigvalsh((G + G.conj().T) / 2.0)
    s = w.sum()
    return float(w[-1] / s) if s > 0 else 0.0
