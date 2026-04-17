"""M/M/1 queueing model.

Single-server queue with Poisson arrivals and exponential service times.
System capacity is infinite, service discipline is FCFS.

Analytical formulas:
    ρ = λ / μ  (stability requires ρ < 1)
    p_k = (1 - ρ) ρ^k
    E[L] = ρ / (1 - ρ)
    E[W] = 1 / (μ - λ)
    E[Lq] = ρ² / (1 - ρ)
    E[Wq] = ρ / (μ(1 - ρ))
"""

from __future__ import annotations

import math
from typing import Any

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.utils import validate_positive, validate_stability


class MM1(BaseQueueModel):
    """M/M/1 queueing model.

    Args:
        arrival_rate: Arrival rate λ (must be positive).
        service_rate: Service rate μ (must be positive, and λ < μ).

    Examples:
        >>> q = MM1(arrival_rate=2.0, service_rate=3.0)
        >>> q.utilization()
        0.6666666666666666
        >>> q.mean_queue_length()
        1.3333333333333333
    """

    def __init__(self, arrival_rate: float, service_rate: float) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        validate_stability(arrival_rate, service_rate, servers=1)
        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)

    # --- Core metrics ---

    def utilization(self) -> float:
        """ρ = λ / μ"""
        return self.arrival_rate / self.service_rate

    def mean_system_size(self) -> float:
        """E[L] = ρ / (1 - ρ)"""
        rho = self.utilization()
        return rho / (1.0 - rho)

    def mean_queue_length(self) -> float:
        """E[Lq] = ρ² / (1 - ρ)"""
        rho = self.utilization()
        return rho**2 / (1.0 - rho)

    def mean_system_time(self) -> float:
        """E[W] = 1 / (μ - λ)"""
        return 1.0 / (self.service_rate - self.arrival_rate)

    def mean_wait(self) -> float:
        """E[Wq] = ρ / (μ(1 - ρ))"""
        rho = self.utilization()
        return rho / (self.service_rate * (1.0 - rho))

    def prob_wait(self) -> float:
        """Probability that an arriving customer must wait = ρ (for M/M/1)."""
        return self.utilization()

    def prob_n(self, n: int) -> float:
        """Steady-state probability of n customers in the system.

        p_n = (1 - ρ) ρ^n
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        rho = self.utilization()
        return (1.0 - rho) * rho**n

    def wait_time_cdf(self, t: float) -> float:
        """CDF of waiting time in queue: P(Wq ≤ t).

        P(Wq ≤ t) = 1 - ρ exp(-μ(1-ρ)t)  for t ≥ 0
        """
        if t < 0:
            return 0.0
        rho = self.utilization()
        return 1.0 - rho * math.exp(-(self.service_rate - self.arrival_rate) * t)

    def wait_time_pdf(self, t: float) -> float:
        """PDF of waiting time in queue for t > 0.

        f(t) = ρ(μ-λ) exp(-(μ-λ)t)  for t > 0

        Note: There is a probability mass (1-ρ) at t=0 (no wait).
        """
        if t < 0:
            return 0.0
        if t == 0.0:
            return float("inf")  # point mass at 0
        rate = self.service_rate - self.arrival_rate
        rho = self.utilization()
        return rho * rate * math.exp(-rate * t)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "MM1",
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MM1:
        return cls(arrival_rate=data["arrival_rate"], service_rate=data["service_rate"])
