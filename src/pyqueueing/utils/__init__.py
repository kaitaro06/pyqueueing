"""Input validation utilities for pyqueueing."""

from __future__ import annotations


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is strictly positive.

    Args:
        value: The value to check.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is not positive.
        TypeError: If value is not a number.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative.

    Args:
        value: The value to check.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is negative.
        TypeError: If value is not a number.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_positive_integer(value: int, name: str) -> None:
    """Validate that a value is a positive integer.

    Args:
        value: The value to check.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is not a positive integer.
        TypeError: If value is not an integer.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def validate_stability(arrival_rate: float, service_rate: float, servers: int = 1) -> None:
    """Validate the stability condition λ < c·μ.

    Args:
        arrival_rate: Arrival rate λ.
        service_rate: Service rate μ per server.
        servers: Number of servers c.

    Raises:
        ValueError: If the system is unstable.
    """
    rho = arrival_rate / (servers * service_rate)
    if rho >= 1.0:
        raise ValueError(
            f"System is unstable: ρ = λ/(cμ) = {rho:.4f} ≥ 1. "
            f"Need λ < cμ (λ={arrival_rate}, c={servers}, μ={service_rate})"
        )


def validate_probability(value: float, name: str) -> None:
    """Validate that a value is a valid probability in (0, 1].

    Args:
        value: The value to check.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is not in (0, 1].
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not (0 < value <= 1):
        raise ValueError(f"{name} must be in (0, 1], got {value}")
