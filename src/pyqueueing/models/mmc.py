"""M/M/c queueing model.

Multi-server queue with Poisson arrivals and exponential service times.

Analytical formulas:
    a = λ/μ  (offered load)
    ρ = a/c = λ/(cμ)  (stability requires ρ < 1)

    p_0 = [ Σ_{k=0}^{c-1} a^k/k! + a^c/((c-1)!(c-a)) ]^{-1}
    p_k = (a^k / k!) p_0                      for k = 0, ..., c-1
    p_k = (a^k / (c! c^{k-c})) p_0            for k >= c

    Erlang C = P(Wq > 0) = a^c / ((c-1)!(c-a)) * p_0
    E[Lq] = C(c,a) * ρ / (1 - ρ)
    E[Wq] = E[Lq] / λ
    E[L] = E[Lq] + a
    E[W] = E[L] / λ
"""

from __future__ import annotations

import math
from typing import Any

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.utils import validate_positive, validate_positive_integer, validate_stability


def _erlang_c(c: int, a: float) -> float:
    """Compute the Erlang C probability C(c, a) in log-space to avoid overflow.

    C(c, a) = (a^c / ((c-1)!(c-a))) / (Σ_{k=0}^{c-1} a^k/k! + a^c/((c-1)!(c-a)))

    Args:
        c: Number of servers.
        a: Offered load (λ/μ).

    Returns:
        Erlang C probability.
    """
    if a >= c:
        raise ValueError(f"System unstable: offered load a={a} >= c={c}")

    # Compute in log-space: log(a^k / k!)
    log_terms = []
    for k in range(c):
        log_terms.append(k * math.log(a) - math.lgamma(k + 1))

    log_last = c * math.log(a) - math.lgamma(c) - math.log(c - a)

    # Use log-sum-exp for numerical stability
    max_log = max(*log_terms, log_last)
    sum_exp = sum(math.exp(lt - max_log) for lt in log_terms) + math.exp(log_last - max_log)

    erlang_c = math.exp(log_last - max_log) / sum_exp
    return erlang_c


class MMC(BaseQueueModel):
    """M/M/c queueing model.

    Args:
        arrival_rate: Arrival rate λ.
        service_rate: Service rate μ per server.
        servers: Number of servers c.

    Examples:
        >>> q = MMC(arrival_rate=10, service_rate=3, servers=4)
        >>> round(q.prob_wait(), 4)
        0.5541
    """

    def __init__(self, arrival_rate: float, service_rate: float, servers: int) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        validate_positive_integer(servers, "servers")
        validate_stability(arrival_rate, service_rate, servers)
        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self.servers = servers

    @property
    def offered_load(self) -> float:
        """a = λ / μ"""
        return self.arrival_rate / self.service_rate

    def utilization(self) -> float:
        """ρ = λ / (cμ)"""
        return self.arrival_rate / (self.servers * self.service_rate)

    def prob_wait(self) -> float:
        """Erlang C formula: probability that an arriving customer must wait."""
        return _erlang_c(self.servers, self.offered_load)

    def mean_queue_length(self) -> float:
        """E[Lq] = C(c,a) * ρ / (1 - ρ)"""
        rho = self.utilization()
        return self.prob_wait() * rho / (1.0 - rho)

    def mean_system_size(self) -> float:
        """E[L] = E[Lq] + a"""
        return self.mean_queue_length() + self.offered_load

    def mean_wait(self) -> float:
        """E[Wq] = E[Lq] / λ"""
        return self.mean_queue_length() / self.arrival_rate

    def mean_system_time(self) -> float:
        """E[W] = E[Wq] + 1/μ"""
        return self.mean_wait() + 1.0 / self.service_rate

    def prob_n(self, n: int) -> float:
        """Steady-state probability of n customers in the system.

        Uses log-space computation for numerical stability with large c.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        a = self.offered_load
        c = self.servers

        # log(p_n / p_0)
        if n < c:
            log_ratio = n * math.log(a) - math.lgamma(n + 1)
        else:
            log_ratio = (
                n * math.log(a) - math.lgamma(c + 1) - (n - c) * math.log(c)
            )

        # log(p_0)
        log_p0 = -self._log_p0_denominator()
        return math.exp(log_ratio + log_p0)

    def _log_p0_denominator(self) -> float:
        """log(Σ_{k=0}^{c-1} a^k/k! + a^c/((c-1)!(c-a))) in log-sum-exp form."""
        a = self.offered_load
        c = self.servers
        log_terms = [k * math.log(a) - math.lgamma(k + 1) for k in range(c)]
        log_last = c * math.log(a) - math.lgamma(c) - math.log(c - a)
        log_terms.append(log_last)
        max_log = max(log_terms)
        return max_log + math.log(sum(math.exp(lt - max_log) for lt in log_terms))

    def wait_time_cdf(self, t: float) -> float:
        """CDF of waiting time: P(Wq <= t).

        W_q(t) = 1 - C(c,a) * exp(-(cμ - λ)t)  for t >= 0

        Analytical result for the M/M/c waiting-time distribution.
        """
        if t < 0:
            return 0.0
        return 1.0 - self.prob_wait() * math.exp(
            -(self.servers * self.service_rate - self.arrival_rate) * t
        )

    def wait_time_pdf(self, t: float) -> float:
        """PDF of waiting time in queue for t > 0.

        f(t) = C(c,a)(cμ - λ) exp(-(cμ - λ)t)  for t > 0

        Note: There is a probability mass (1 - C(c,a)) at t=0.
        """
        if t < 0:
            return 0.0
        if t == 0.0:
            return float("inf")
        rate = self.servers * self.service_rate - self.arrival_rate
        return self.prob_wait() * rate * math.exp(-rate * t)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "MMC",
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "servers": self.servers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MMC:
        return cls(
            arrival_rate=data["arrival_rate"],
            service_rate=data["service_rate"],
            servers=data["servers"],
        )
