"""M/M/c/K queueing model — multi-server, finite capacity K.

Formulas:
    p_0 = [Σ_{k=0}^{c-1} a^k/k! + (a^c/c!) Σ_{j=0}^{K-c} (a/c)^j]^{-1}
    where a = λ/μ

    p_k = (a^k / k!) p_0               for 0 <= k <= c
    p_k = (a^k / (c! c^{k-c})) p_0     for c < k <= K

    Blocking probability = p_K
    Effective arrival rate = λ(1 - p_K)
"""

from __future__ import annotations

import math
from typing import Any

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.utils import validate_positive, validate_positive_integer


class MMcK(BaseQueueModel):
    """M/M/c/K queueing model (multi-server, finite capacity).

    Args:
        arrival_rate: Arrival rate λ.
        service_rate: Service rate μ per server.
        servers: Number of servers c.
        capacity: System capacity K (K >= c).

    Note:
        Always stable due to finite capacity.
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        servers: int,
        capacity: int,
    ) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        validate_positive_integer(servers, "servers")
        validate_positive_integer(capacity, "capacity")
        if capacity < servers:
            raise ValueError(
                f"capacity ({capacity}) must be >= servers ({servers})"
            )
        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self.servers = servers
        self.capacity = capacity

    @property
    def offered_load(self) -> float:
        """a = λ / μ"""
        return self.arrival_rate / self.service_rate

    def _log_p0_denominator(self) -> float:
        """log(Σ p_n/p_0) in log-sum-exp form for numerical stability."""
        a = self.offered_load
        c = self.servers
        K = self.capacity

        log_terms = []
        # k = 0..c: log(a^k / k!)
        for k in range(c + 1):
            log_terms.append(k * math.log(a) - math.lgamma(k + 1) if a > 0 or k == 0 else -math.inf)

        # k = c+1..K: log(a^k / (c! * c^{k-c}))
        for k in range(c + 1, K + 1):
            lt = k * math.log(a) - math.lgamma(c + 1) - (k - c) * math.log(c)
            log_terms.append(lt)

        max_log = max(log_terms)
        return max_log + math.log(sum(math.exp(lt - max_log) for lt in log_terms))

    def prob_n(self, n: int) -> float:
        """Steady-state probability of n customers in the system.

        Uses log-space computation for numerical stability with large c.
        """
        if n < 0 or n > self.capacity:
            return 0.0
        a = self.offered_load
        c = self.servers

        if a == 0:
            return 1.0 if n == 0 else 0.0

        # log(p_n / p_0)
        if n <= c:
            log_ratio = n * math.log(a) - math.lgamma(n + 1)
        else:
            log_ratio = n * math.log(a) - math.lgamma(c + 1) - (n - c) * math.log(c)

        log_p0 = -self._log_p0_denominator()
        return math.exp(log_ratio + log_p0)

    def prob_block(self) -> float:
        """Blocking probability p_K."""
        return self.prob_n(self.capacity)

    def effective_arrival_rate(self) -> float:
        """λ_eff = λ(1 - p_K)"""
        return self.arrival_rate * (1.0 - self.prob_block())

    def utilization(self) -> float:
        """Server utilization = λ_eff / (c·μ)."""
        return self.effective_arrival_rate() / (self.servers * self.service_rate)

    def mean_system_size(self) -> float:
        """E[L] = Σ k·p_k"""
        return sum(k * self.prob_n(k) for k in range(self.capacity + 1))

    def mean_queue_length(self) -> float:
        """E[Lq] = Σ_{k=c+1}^{K} (k-c)·p_k"""
        c = self.servers
        return sum((k - c) * self.prob_n(k) for k in range(c + 1, self.capacity + 1))

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
            "model": "MMcK",
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "servers": self.servers,
            "capacity": self.capacity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MMcK:
        return cls(
            arrival_rate=data["arrival_rate"],
            service_rate=data["service_rate"],
            servers=data["servers"],
            capacity=data["capacity"],
        )
