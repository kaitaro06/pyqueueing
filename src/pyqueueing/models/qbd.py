"""QBD (Quasi-Birth-Death) process model.

A level-independent QBD process is a CTMC whose generator has a
block-tridiagonal repeating structure:

    Q = [ B1  B0             ]
        [ A2  A1  A0         ]
        [     A2  A1  A0     ]
        [         ...        ]

Applications:
    - M/PH/1, PH/M/1, PH/PH/1 queues
    - Priority queues
    - Retrial queues
    - Network protocol analysis (TCP congestion windows)

This module provides a high-level interface wrapping the matrix geometric solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyqueueing.solvers.matrix_geometric import (
    rate_matrix_iterative,
    rate_matrix_log_reduction,
    stationary_distribution,
)


class QBD:
    """Quasi-Birth-Death process model.

    Args:
        A0: Upward transition rate matrix (m × m). Transitions from level k to k+1.
        A1: Diagonal rate matrix (m × m). Transitions within level k.
            Diagonal entries should be negative (row sums of generator = 0).
        A2: Downward transition rate matrix (m × m). Transitions from level k to k-1.
        B0: Boundary upward block (level 0 → 1). Defaults to A0.
        B1: Boundary diagonal block (level 0). Defaults to A1.

    Examples:
        Simple M/M/1 as QBD (scalar phases):

        >>> import numpy as np
        >>> lam, mu = 2.0, 3.0
        >>> A0 = np.array([[lam]])
        >>> A1 = np.array([[-(lam + mu)]])
        >>> A2 = np.array([[mu]])
        >>> qbd = QBD(A0, A1, A2)
        >>> R = qbd.rate_matrix()
        >>> abs(R[0, 0] - lam/mu) < 1e-10
        True
    """

    def __init__(
        self,
        A0: NDArray[np.float64],
        A1: NDArray[np.float64],
        A2: NDArray[np.float64],
        B0: NDArray[np.float64] | None = None,
        B1: NDArray[np.float64] | None = None,
    ) -> None:
        self.A0 = np.asarray(A0, dtype=np.float64)
        self.A1 = np.asarray(A1, dtype=np.float64)
        self.A2 = np.asarray(A2, dtype=np.float64)
        self.B0 = np.asarray(B0, dtype=np.float64) if B0 is not None else self.A0.copy()
        self.B1 = np.asarray(B1, dtype=np.float64) if B1 is not None else self.A1.copy()

        # Validate dimensions
        m = self.A0.shape[0]
        for name, mat in [("A0", self.A0), ("A1", self.A1), ("A2", self.A2),
                          ("B0", self.B0), ("B1", self.B1)]:
            if mat.shape != (m, m):
                raise ValueError(f"{name} must be {m}×{m}, got {mat.shape}")

        self._phase_dim = m
        self._R: NDArray[np.float64] | None = None
        self._R_iters: int = 0

    @property
    def phase_dim(self) -> int:
        """Dimension of the phase space (m)."""
        return self._phase_dim

    def rate_matrix(
        self,
        method: str = "iterative",
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Compute the rate matrix R.

        R is the minimal non-negative solution to: A0 + R·A1 + R²·A2 = 0

        Args:
            method: 'iterative' (successive substitution) or 'log_reduction'.
            **kwargs: Passed to the solver (max_iter, tol).

        Returns:
            Rate matrix R (m × m).
        """
        if method == "iterative":
            R, iters = rate_matrix_iterative(self.A0, self.A1, self.A2, **kwargs)
        elif method == "log_reduction":
            R, iters = rate_matrix_log_reduction(self.A0, self.A1, self.A2, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iterative' or 'log_reduction'.")

        self._R = R
        self._R_iters = iters
        return R

    def spectral_radius(self) -> float:
        """Spectral radius of R: max |eigenvalue|.

        Must be < 1 for the QBD to be positive recurrent (stable).
        """
        R = self._ensure_R()
        eigenvalues = np.linalg.eigvals(R)
        return float(np.max(np.abs(eigenvalues)))

    def is_stable(self) -> bool:
        """Check if the QBD is positive recurrent (spectral radius of R < 1)."""
        return self.spectral_radius() < 1.0 - 1e-10

    def stationary(self, max_levels: int = 200) -> list[NDArray[np.float64]]:
        """Compute the stationary distribution π.

        π_k = π_0 · R^k, where π_0 is determined by the boundary equations.

        Args:
            max_levels: Maximum number of levels to compute.

        Returns:
            List of π_k vectors, k = 0, 1, 2, ...
        """
        R = self._ensure_R()
        return stationary_distribution(
            R, self.B1, self.B0, A2=self.A2, max_levels=max_levels
        )

    def mean_level(self) -> float:
        """E[level] = Σ k · ||π_k||₁.

        Mean level (analogous to mean system size L for simple queues).
        """
        pi = self.stationary()
        return sum(k * np.sum(lv) for k, lv in enumerate(pi))

    def level_distribution(self, max_levels: int = 200) -> NDArray[np.float64]:
        """Marginal probability of being at each level.

        Returns:
            1-D array where entry k = Σ_j π_k(j).
        """
        pi = self.stationary(max_levels=max_levels)
        return np.array([np.sum(lv) for lv in pi])

    def _ensure_R(self) -> NDArray[np.float64]:
        if self._R is None:
            self.rate_matrix()
        assert self._R is not None
        return self._R

    def __repr__(self) -> str:
        return f"QBD(phase_dim={self._phase_dim})"
