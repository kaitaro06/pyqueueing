"""Erlang C model — wrapper around M/M/c with inverse-calculation utilities.

Erlang C is mathematically equivalent to the M/M/c wait probability.
This class provides convenience methods for capacity planning, especially
for call center workforce management.

Analytical formulas:
    C(c, a) = (a^c / ((c-1)!(c-a))) / (Σ_{k=0}^{c-1} a^k/k! + a^c/((c-1)!(c-a)))
    P(Wq > t) = C(c, a) exp(-(cμ - λ)t)
"""

from __future__ import annotations

import math
from typing import Any

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.models.mmc import MMC, _erlang_c
from pyqueueing.utils import validate_positive, validate_probability


class ErlangC(BaseQueueModel):
    """Erlang C model for capacity planning.

    Args:
        arrival_rate: Arrival rate λ.
        service_rate: Service rate μ per server.
        servers: Number of servers c (optional; used when querying metrics).

    Examples:
        >>> q = ErlangC(arrival_rate=100, service_rate=12)
        >>> q.required_servers(target_wait_prob=0.2)
        10
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        servers: int | None = None,
    ) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self._servers = servers
        self._mmc: MMC | None = None
        if servers is not None:
            self._mmc = MMC(arrival_rate, service_rate, servers)

    @property
    def offered_load(self) -> float:
        return self.arrival_rate / self.service_rate

    @property
    def servers(self) -> int:
        if self._servers is None:
            raise ValueError("servers not set. Use required_servers() or pass servers to constructor.")
        return self._servers

    def _ensure_mmc(self) -> MMC:
        if self._mmc is None:
            raise ValueError("servers not set. Use required_servers() or pass servers to constructor.")
        return self._mmc

    def prob_wait(self) -> float:
        """Erlang C probability (identical to MMC.prob_wait)."""
        return self._ensure_mmc().prob_wait()

    def utilization(self) -> float:
        return self._ensure_mmc().utilization()

    def mean_queue_length(self) -> float:
        return self._ensure_mmc().mean_queue_length()

    def mean_system_size(self) -> float:
        return self._ensure_mmc().mean_system_size()

    def mean_wait(self) -> float:
        return self._ensure_mmc().mean_wait()

    def mean_system_time(self) -> float:
        return self._ensure_mmc().mean_system_time()

    def service_level(self, target_time: float) -> float:
        """Probability of waiting less than target_time seconds.

        SL = 1 - C(c,a) * exp(-(cμ - λ) * t)

        Args:
            target_time: Target wait time threshold t.

        Returns:
            P(Wq <= t)
        """
        return self._ensure_mmc().wait_time_cdf(target_time)

    def required_servers(
        self,
        *,
        target_wait_prob: float | None = None,
        target_mean_wait: float | None = None,
        target_service_level: tuple[float, float] | None = None,
    ) -> int:
        """Find minimum servers to satisfy one of the target constraints.

        Exactly one of the three keyword arguments must be provided.

        Args:
            target_wait_prob: Target P(Wq > 0) <= value.
            target_mean_wait: Target E[Wq] <= value.
            target_service_level: Tuple (time, prob) such that P(Wq <= time) >= prob.

        Returns:
            Minimum number of servers c satisfying the constraint.
        """
        n_specified = sum(x is not None for x in [target_wait_prob, target_mean_wait, target_service_level])
        if n_specified != 1:
            raise ValueError("Exactly one target must be specified.")

        a = self.offered_load
        c = max(math.ceil(a) + 1, 1)  # start from minimum stable c

        for _ in range(10000):
            if a >= c:
                c += 1
                continue
            try:
                mmc = MMC(self.arrival_rate, self.service_rate, c)
            except ValueError:
                c += 1
                continue

            if target_wait_prob is not None:
                validate_probability(target_wait_prob, "target_wait_prob")
                if mmc.prob_wait() <= target_wait_prob:
                    return c
            elif target_mean_wait is not None:
                validate_positive(target_mean_wait, "target_mean_wait")
                if mmc.mean_wait() <= target_mean_wait:
                    return c
            elif target_service_level is not None:
                time_threshold, prob_threshold = target_service_level
                validate_positive(time_threshold, "target_service_level time")
                validate_probability(prob_threshold, "target_service_level prob")
                if mmc.wait_time_cdf(time_threshold) >= prob_threshold:
                    return c
            c += 1

        raise RuntimeError("Could not find solution within 10000 servers")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": "ErlangC",
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
        }
        if self._servers is not None:
            d["servers"] = self._servers
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErlangC:
        return cls(
            arrival_rate=data["arrival_rate"],
            service_rate=data["service_rate"],
            servers=data.get("servers"),
        )
