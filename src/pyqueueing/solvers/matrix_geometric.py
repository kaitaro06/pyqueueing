"""Matrix Geometric solver for Quasi-Birth-Death (QBD) processes.

Implements iterative algorithms to compute the rate matrix R
for level-independent QBD processes.

A QBD process has a block-tridiagonal generator:

    Q = [ B1  B0             ]
        [ B2  A1  A0         ]
        [     A2  A1  A0     ]
        [         A2  A1  A0 ]
        [             ...    ]

The rate matrix R satisfies:  A0 + R·A1 + R²·A2 = 0

Available algorithms:
    - Successive substitution (simple iteration)
    - Logarithmic reduction (quadratically convergent)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rate_matrix_iterative(
    A0: NDArray[np.float64],
    A1: NDArray[np.float64],
    A2: NDArray[np.float64],
    *,
    max_iter: int = 10000,
    tol: float = 1e-12,
) -> tuple[NDArray[np.float64], int]:
    """Compute rate matrix R by successive substitution.

    Iteration: R_{k+1} = -A0 · (A1 + R_k · A2)^{-1}

    This is equivalent to: R_{k+1} = -A0 · inv(A1 + R_k · A2)

    Args:
        A0: Upward transition rate matrix (m × m).
        A1: Level transition rate matrix (m × m), diagonal-dominant.
        A2: Downward transition rate matrix (m × m).
        max_iter: Maximum iterations.
        tol: Convergence tolerance (Frobenius norm of change).

    Returns:
        (R, iterations): Rate matrix R and number of iterations used.

    Raises:
        ValueError: If matrices have incompatible shapes.
        RuntimeError: If iteration does not converge.
    """
    m = A0.shape[0]
    _validate_qbd_matrices(A0, A1, A2)

    R = np.zeros((m, m), dtype=np.float64)

    for k in range(max_iter):
        M = A1 + R @ A2
        try:
            R_new = -A0 @ np.linalg.inv(M)
        except np.linalg.LinAlgError:
            R_new = -A0 @ np.linalg.pinv(M)

        diff = np.linalg.norm(R_new - R, ord="fro")
        R = R_new
        if diff < tol:
            return R, k + 1

    raise RuntimeError(
        f"Rate matrix iteration did not converge after {max_iter} iterations "
        f"(last change: {diff:.2e})"
    )


def rate_matrix_log_reduction(
    A0: NDArray[np.float64],
    A1: NDArray[np.float64],
    A2: NDArray[np.float64],
    *,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> tuple[NDArray[np.float64], int]:
    """Compute rate matrix R using logarithmic reduction.

    Converts the CTMC QBD to an embedded DTMC via uniformization,
    then applies cyclic reduction for quadratic convergence.

    Args:
        A0: Upward transition rate matrix (m × m).
        A1: Level transition rate matrix (m × m).
        A2: Downward transition rate matrix (m × m).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        (R, iterations): Rate matrix R and number of iterations.
    """
    _validate_qbd_matrices(A0, A1, A2)
    m = A0.shape[0]
    I = np.eye(m)

    # Uniformize: find max absolute diagonal rate
    q_max = float(np.max(np.abs(np.diag(A1))))
    if q_max == 0:
        q_max = float(np.max(np.abs(A0) + np.abs(A1) + np.abs(A2)))
    if q_max == 0:
        return np.zeros((m, m)), 0

    # DTMC transition matrices: P = I + Q/q_max
    P0 = A0 / q_max
    P1 = I + A1 / q_max
    P2 = A2 / q_max

    # Cyclic reduction on DTMC
    # We want R_dtmc satisfying: P0 + R·P1 + R²·P2 = R
    # CR sequences
    C0 = P0.copy()
    C1 = P1.copy()
    C2 = P2.copy()
    U = P1.copy()  # will converge to the "censored" matrix for R

    for k in range(max_iter):
        try:
            C1_inv = np.linalg.inv(I - C1)
        except np.linalg.LinAlgError:
            C1_inv = np.linalg.pinv(I - C1)

        T0 = C0 @ C1_inv @ C0
        T2 = C2 @ C1_inv @ C2

        U = U + C0 @ C1_inv @ C2

        C1_new = C1 + C0 @ C1_inv @ C2 + C2 @ C1_inv @ C0
        C0 = T0
        C2 = T2
        C1 = C1_new

        diff = np.linalg.norm(C0, ord="fro")
        if diff < tol:
            break

    # R_dtmc from U: R_dtmc = P0 @ inv(I - U) (approximately)
    # Actually: the stationary R satisfies R = P0(I - P1 - R·P2)^{-1}
    # With converged U: R_dtmc ≈ U columns give the answer
    # Simpler: just use U to get R_dtmc, then convert back
    try:
        R_dtmc = P0 @ np.linalg.inv(I - U)
    except np.linalg.LinAlgError:
        R_dtmc = P0 @ np.linalg.pinv(I - U)

    # Convert DTMC R back to CTMC R: R_ctmc = q_max · R_dtmc
    # Actually R_ctmc satisfies A0 + R·A1 + R²·A2 = 0
    # and R_dtmc satisfies R_dtmc = P0 + R_dtmc·P1 + R_dtmc²·P2
    # Since P_i = A_i/q for i≠1 and P1 = I + A1/q, we have R_ctmc = R_dtmc
    # (the rate matrix R is the same for CTMC and its uniformized DTMC)
    R = R_dtmc
    return R, k + 1


def stationary_distribution(
    R: NDArray[np.float64],
    B1: NDArray[np.float64],
    B0: NDArray[np.float64],
    B2: NDArray[np.float64] | None = None,
    A1: NDArray[np.float64] | None = None,
    A2: NDArray[np.float64] | None = None,
    *,
    max_levels: int = 200,
    tail_tol: float = 1e-10,
) -> list[NDArray[np.float64]]:
    """Compute the stationary distribution π of a QBD process.

    π_0 is found from the boundary equations, then π_k = π_0 · R^k.

    For a standard QBD:
        π_0 · B1 + π_1 · B2_boundary = 0   (boundary)
        π_k = π_{k-1} · R                    (geometric)
        Σ π_k · e = 1                         (normalization)

    Args:
        R: Rate matrix (from rate_matrix_iterative or rate_matrix_log_reduction).
        B1: Boundary level diagonal block.
        B0: Boundary upward block (transition from level 0 to level 1).
        B2: Boundary downward block (level 1 to level 0). If None, uses A2.
        A1: Level diagonal block (needed if B2 is None).
        A2: Level downward block (used as default for B2).
        max_levels: Maximum number of levels to compute.
        tail_tol: Stop when π_k norm falls below this.

    Returns:
        List of π_k vectors (one per level, starting from level 0).
    """
    m = R.shape[0]

    if B2 is None:
        if A2 is None:
            raise ValueError("Either B2 or A2 must be provided.")
        B2 = A2

    # Boundary equation: π_0 · (B1 + R · B2) = 0
    # This is a left eigenvector problem.
    M = B1 + R @ B2

    # Find left null vector of M: x · M = 0, i.e., M^T · x = 0
    _, _, Vh = np.linalg.svd(M.T)
    pi_0 = np.real(Vh[-1, :])

    # Ensure non-negative
    if np.sum(pi_0) < 0:
        pi_0 = -pi_0
    pi_0 = np.maximum(pi_0, 0.0)

    # Compute geometric tail: π_k = π_0 · R^k
    levels = [pi_0]
    Rk = np.eye(m, dtype=np.float64)
    total_mass = np.sum(pi_0)

    for k in range(1, max_levels):
        Rk = Rk @ R
        pi_k = pi_0 @ Rk
        levels.append(pi_k)
        mass = np.sum(pi_k)
        total_mass += mass
        if mass < tail_tol:
            break

    # Normalize so that Σ π_k · 1 = 1
    if total_mass > 0:
        levels = [lv / total_mass for lv in levels]

    return levels


def _validate_qbd_matrices(
    A0: NDArray[np.float64],
    A1: NDArray[np.float64],
    A2: NDArray[np.float64],
) -> None:
    """Validate QBD matrix dimensions."""
    if A0.ndim != 2 or A1.ndim != 2 or A2.ndim != 2:
        raise ValueError("All matrices must be 2-dimensional.")
    m = A0.shape[0]
    if A0.shape != (m, m) or A1.shape != (m, m) or A2.shape != (m, m):
        raise ValueError(
            f"All matrices must be square and same size. "
            f"Got A0={A0.shape}, A1={A1.shape}, A2={A2.shape}"
        )
