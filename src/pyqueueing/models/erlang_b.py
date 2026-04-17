"""Erlang B model — M/M/c/c (loss system, no waiting room).

Analytical formulas:
    B(c, a) = (a^c / c!) / Σ_{k=0}^{c} a^k / k!

    Recursive formula (numerically stable):
        B(1, a) = a / (1 + a)
        B(c, a) = a·B(c-1, a) / (c + a·B(c-1, a))

    This formula is insensitive to service time distribution
    (also holds for M/G/c/c).
"""

from __future__ import annotations

import math
from typing import Any

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.utils import validate_positive, validate_positive_integer


def erlang_b_formula(c: int, a: float) -> float:
    """Compute the Erlang B blocking probability using the recursive formula.

    This is numerically stable even for large c.

    B(1, a) = a / (1 + a)
    B(c, a) = a * B(c-1, a) / (c + a * B(c-1, a))

    Args:
        c: Number of servers.
        a: Offered load (λ/μ).

    Returns:
        Blocking probability B(c, a).
    """
    b = a / (1.0 + a)  # B(1, a)
    for i in range(2, c + 1):
        b = a * b / (i + a * b)
    return b


class ErlangB(BaseQueueModel):
    """Erlang B (M/M/c/c) loss system model.

    Args:
        arrival_rate: Arrival rate λ.
        service_rate: Service rate μ per server.
        servers: Number of servers c.

    Examples:
        >>> q = ErlangB(arrival_rate=10, service_rate=1, servers=12)
        >>> round(q.prob_block(), 4)
        0.1025
    """

    def __init__(self, arrival_rate: float, service_rate: float, servers: int) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        validate_positive_integer(servers, "servers")
        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self.servers = servers

    @property
    def offered_load(self) -> float:
        """a = λ / μ"""
        return self.arrival_rate / self.service_rate

    def prob_block(self) -> float:
        """Erlang B blocking probability B(c, a)."""
        return erlang_b_formula(self.servers, self.offered_load)

    def effective_arrival_rate(self) -> float:
        """λ_eff = λ(1 - B(c, a))"""
        return self.arrival_rate * (1.0 - self.prob_block())

    def utilization(self) -> float:
        """Per-server utilization = λ_eff / (c·μ)."""
        return self.effective_arrival_rate() / (self.servers * self.service_rate)

    def mean_system_size(self) -> float:
        """E[L] = a(1 - B(c, a)) — all customers are in service."""
        return self.offered_load * (1.0 - self.prob_block())

    def mean_queue_length(self) -> float:
        """Lq = 0 — no waiting room in a loss system."""
        return 0.0

    def mean_system_time(self) -> float:
        """E[W] = 1/μ — no waiting, only service time."""
        return 1.0 / self.service_rate

    def mean_wait(self) -> float:
        """Wq = 0 — no waiting room."""
        return 0.0

    def required_servers(self, target_block_prob: float) -> int:
        """Find minimum number of servers to achieve target blocking probability.

        Args:
            target_block_prob: Target blocking probability (0 < p <= 1).

        Returns:
            Minimum c such that B(c, a) <= target_block_prob.
        """
        from pyqueueing.utils import validate_probability
        validate_probability(target_block_prob, "target_block_prob")
        a = self.offered_load
        c = 1
        while erlang_b_formula(c, a) > target_block_prob:
            c += 1
            if c > 10000:
                raise RuntimeError("Could not find solution within 10000 servers")
        return c

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "ErlangB",
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "servers": self.servers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErlangB:
        return cls(
            arrival_rate=data["arrival_rate"],
            service_rate=data["service_rate"],
            servers=data["servers"],
        )
