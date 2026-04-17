"""M/M/1/K queueing model — single server, finite capacity K.

Analytical formulas:
    p_k = (1-ρ)ρ^k / (1-ρ^{K+1})    (ρ ≠ 1)
    p_k = 1 / (K+1)                   (ρ = 1)

    E[L] = ρ/(1-ρ) - (K+1)ρ^{K+1}/(1-ρ^{K+1})   (ρ ≠ 1)
    E[L] = K/2                                      (ρ = 1)

    Blocking probability = p_K
    Effective arrival rate = λ(1 - p_K)
"""

from __future__ import annotations

import math
from typing import Any

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.utils import validate_positive, validate_positive_integer


class MM1K(BaseQueueModel):
    """M/M/1/K queueing model (finite capacity).

    Args:
        arrival_rate: Arrival rate λ.
        service_rate: Service rate μ.
        capacity: System capacity K (max customers including the one in service).

    Note:
        This model is always stable regardless of ρ because the capacity is finite.
    """

    def __init__(self, arrival_rate: float, service_rate: float, capacity: int) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        validate_positive_integer(capacity, "capacity")
        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self.capacity = capacity

    def _rho(self) -> float:
        return self.arrival_rate / self.service_rate

    def utilization(self) -> float:
        """Server utilization = 1 - p_0."""
        return 1.0 - self.prob_n(0)

    def prob_n(self, n: int) -> float:
        """Steady-state probability of n customers in the system."""
        if n < 0 or n > self.capacity:
            return 0.0
        rho = self._rho()
        K = self.capacity
        if math.isclose(rho, 1.0, rel_tol=1e-12):
            return 1.0 / (K + 1)
        return (1.0 - rho) * rho**n / (1.0 - rho ** (K + 1))

    def prob_block(self) -> float:
        """Blocking probability p_K."""
        return self.prob_n(self.capacity)

    def effective_arrival_rate(self) -> float:
        """λ_eff = λ(1 - p_K)"""
        return self.arrival_rate * (1.0 - self.prob_block())

    def mean_system_size(self) -> float:
        """E[L]"""
        rho = self._rho()
        K = self.capacity
        if math.isclose(rho, 1.0, rel_tol=1e-12):
            return K / 2.0
        return rho / (1.0 - rho) - (K + 1) * rho ** (K + 1) / (1.0 - rho ** (K + 1))

    def mean_queue_length(self) -> float:
        """E[Lq] = E[L] - (1 - p_0)"""
        return self.mean_system_size() - self.utilization()

    def mean_system_time(self) -> float:
        """E[W] = E[L] / λ_eff"""
        lam_eff = self.effective_arrival_rate()
        if lam_eff == 0:
            return float("inf")
        return self.mean_system_size() / lam_eff

    def mean_wait(self) -> float:
        """E[Wq] = E[Lq] / λ_eff"""
        lam_eff = self.effective_arrival_rate()
        if lam_eff == 0:
            return float("inf")
        return self.mean_queue_length() / lam_eff

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "MM1K",
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "capacity": self.capacity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MM1K:
        return cls(
            arrival_rate=data["arrival_rate"],
            service_rate=data["service_rate"],
            capacity=data["capacity"],
        )
