"""M/M/∞ queueing model — infinite-server (delay) system.

Every arriving customer immediately enters service (no waiting).
The number of busy servers follows a Poisson distribution.

Analytical formulas:
    a = λ/μ  (offered load = mean number of busy servers)
    p_n = e^{-a} a^n / n!  (Poisson distribution)
    E[L] = a
    E[W] = 1/μ
    E[Lq] = 0  (no queue)
    E[Wq] = 0  (no waiting)
"""

from __future__ import annotations

import math
from typing import Any

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.utils import validate_positive


class MMInf(BaseQueueModel):
    """M/M/∞ queueing model (infinite servers).

    Every customer is served immediately upon arrival.
    Useful as a baseline model and for modeling delay systems.

    Args:
        arrival_rate: Arrival rate λ.
        service_rate: Service rate μ per server.

    Examples:
        >>> q = MMInf(arrival_rate=10.0, service_rate=2.0)
        >>> q.mean_system_size()
        5.0
        >>> q.mean_wait()
        0.0
    """

    def __init__(self, arrival_rate: float, service_rate: float) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)

    @property
    def offered_load(self) -> float:
        """a = λ / μ (mean number of busy servers)."""
        return self.arrival_rate / self.service_rate

    def utilization(self) -> float:
        """Not meaningful for infinite servers; returns 0.0."""
        return 0.0

    def mean_queue_length(self) -> float:
        """E[Lq] = 0 (no queue ever forms)."""
        return 0.0

    def mean_system_size(self) -> float:
        """E[L] = a = λ/μ."""
        return self.offered_load

    def mean_wait(self) -> float:
        """E[Wq] = 0 (immediate service)."""
        return 0.0

    def mean_system_time(self) -> float:
        """E[W] = 1/μ."""
        return 1.0 / self.service_rate

    def prob_n(self, n: int) -> float:
        """Steady-state probability of n customers: Poisson(a).

        p_n = e^{-a} a^n / n!
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        a = self.offered_load
        # Use log-space for numerical stability with large n
        log_p = -a + n * math.log(a) - math.lgamma(n + 1)
        return math.exp(log_p)

    def prob_n_or_more(self, n: int) -> float:
        """P(N ≥ n) = 1 - Σ_{k=0}^{n-1} p_k.

        Useful for trunk provisioning: probability that n or more
        servers are simultaneously busy.
        """
        if n <= 0:
            return 1.0
        # Use regularized incomplete gamma function
        from scipy.special import gammainc
        return 1.0 - gammainc(n, self.offered_load)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "MMInf",
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MMInf:
        return cls(
            arrival_rate=data["arrival_rate"],
            service_rate=data["service_rate"],
        )
