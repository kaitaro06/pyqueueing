"""M/M/c+M queueing model (Erlang A) — multi-server queue with impatient customers.

Customers who are not served immediately wait in queue but may abandon
(renege) after an exponentially distributed patience time with rate θ.

This is the most realistic model for call centers where callers hang up.

Analytical formulas (birth-death process):
    λ_n = λ                          for all n ≥ 0
    μ_n = min(n, c)·μ + max(0, n-c)·θ   (service + abandonment)

    p_0 = [Σ_{n=0}^{c-1} a^n/n! + (a^c/c!) Σ_{j=0}^{∞} Π_{i=1}^{j} a/(c + iα)]^{-1}

    where a = λ/μ, α = θ/μ (impatience ratio)

    Key metrics:
        P(wait) = probability of waiting (not immediately served)
        P(abandon) = fraction of customers who abandon
        ASA = average speed of answer (for answered calls)
        SL(t) = fraction of calls answered within t seconds
"""

from __future__ import annotations

import math
from typing import Any

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.utils import validate_positive, validate_positive_integer


def _compute_erlang_a(c: int, a: float, alpha: float, max_n: int = 500) -> dict[str, float]:
    """Compute Erlang A metrics using birth-death balance.

    Uses log-space computation for numerical stability.

    Args:
        c: Number of servers.
        a: Offered load λ/μ.
        alpha: Impatience ratio θ/μ.
        max_n: Truncation point for infinite sums.

    Returns:
        Dict with p0, prob_wait, prob_abandon, and auxiliary values.
    """
    # Compute log(p_n / p_0) for n = 0, 1, ..., max_n
    log_ratios = [0.0]  # n=0: log(p_0/p_0) = 0

    log_a = math.log(a) if a > 0 else -math.inf

    # For n = 1..c: birth rate = λ, death rate = n·μ
    # p_n/p_0 = a^n / n!
    for n in range(1, c + 1):
        log_ratios.append(n * log_a - math.lgamma(n + 1))

    # For n > c: death rate = c·μ + (n-c)·θ
    # p_n/p_0 = (a^c / c!) · Π_{j=1}^{n-c} [a / (c + j·α)]
    log_prefix = c * log_a - math.lgamma(c + 1)
    log_prod = 0.0
    for n in range(c + 1, max_n + 1):
        j = n - c
        denominator = c + j * alpha
        if denominator <= 0:
            break
        log_prod += log_a - math.log(denominator)
        log_ratios.append(log_prefix + log_prod)
        # Early termination if terms become negligible
        if log_prefix + log_prod < log_ratios[0] - 50:
            break

    # Normalize: p_0 = 1 / Σ exp(log_ratios[n])
    max_log = max(log_ratios)
    sum_exp = sum(math.exp(lr - max_log) for lr in log_ratios)
    log_p0 = -max_log - math.log(sum_exp)

    # P(wait) = P(N >= c) = Σ_{n=c}^{∞} p_n
    log_wait_terms = log_ratios[c:]
    if log_wait_terms:
        max_lw = max(log_wait_terms)
        prob_wait = math.exp(
            max_lw + math.log(sum(math.exp(lr - max_lw) for lr in log_wait_terms)) + log_p0
        )
    else:
        prob_wait = 0.0

    # E[N_q] = Σ_{n=c+1}^{∞} (n - c) · p_n
    enq = 0.0
    for n in range(c + 1, len(log_ratios)):
        enq += (n - c) * math.exp(log_ratios[n] + log_p0)

    # E[N] = Σ_{n=0}^{∞} n · p_n
    en = 0.0
    for n in range(1, len(log_ratios)):
        en += n * math.exp(log_ratios[n] + log_p0)

    return {
        "log_p0": log_p0,
        "prob_wait": min(prob_wait, 1.0),
        "mean_queue_length": enq,
        "mean_system_size": en,
        "log_ratios": log_ratios,
    }


class ErlangA(BaseQueueModel):
    """M/M/c+M queueing model (Erlang A) — with customer impatience.

    Args:
        arrival_rate: Arrival rate λ.
        service_rate: Service rate μ per server.
        servers: Number of servers c.
        patience_rate: Abandonment rate θ (1/mean patience time).

    Examples:
        >>> q = ErlangA(arrival_rate=100, service_rate=12, servers=10, patience_rate=0.5)
        >>> round(q.prob_abandon(), 4)  # fraction who abandon
        0.0384
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        servers: int,
        patience_rate: float,
    ) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        validate_positive_integer(servers, "servers")
        validate_positive(patience_rate, "patience_rate")

        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self.servers = servers
        self.patience_rate = float(patience_rate)

        # Pre-compute core metrics
        a = self.arrival_rate / self.service_rate
        alpha = self.patience_rate / self.service_rate
        self._metrics = _compute_erlang_a(self.servers, a, alpha)

    @property
    def offered_load(self) -> float:
        """a = λ/μ"""
        return self.arrival_rate / self.service_rate

    @property
    def impatience_ratio(self) -> float:
        """α = θ/μ"""
        return self.patience_rate / self.service_rate

    def utilization(self) -> float:
        """Server utilization = (λ - λ_abandon) / (c·μ).

        Only answered calls contribute to server utilization.
        """
        answered_rate = self.arrival_rate * (1.0 - self.prob_abandon())
        return answered_rate / (self.servers * self.service_rate)

    def prob_wait(self) -> float:
        """Probability that an arriving customer must wait (not immediately served)."""
        return self._metrics["prob_wait"]

    def prob_abandon(self) -> float:
        """Fraction of arriving customers who abandon before being served.

        P(abandon) = 1 - (c·μ·(1-p_0_idle) + λ·p_idle) / λ
        Computed via: abandon_rate = θ · E[Nq], P(abandon) = θ·E[Nq] / λ
        """
        enq = self._metrics["mean_queue_length"]
        return self.patience_rate * enq / self.arrival_rate

    def mean_queue_length(self) -> float:
        """E[Nq] = mean number of customers waiting in queue."""
        return self._metrics["mean_queue_length"]

    def mean_system_size(self) -> float:
        """E[N] = mean number of customers in the system."""
        return self._metrics["mean_system_size"]

    def mean_wait(self) -> float:
        """E[Wq] = mean wait time for ALL customers (including zero-wait and abandoners).

        E[Wq] = E[Nq] / λ  (by Little's law applied to the queue).
        """
        return self.mean_queue_length() / self.arrival_rate

    def mean_wait_answered(self) -> float:
        """ASA (Average Speed of Answer) — mean wait for answered customers only.

        ASA = E[Wq] / (1 - P(abandon))
        This is the most important metric for call center SLA.
        """
        p_abn = self.prob_abandon()
        if p_abn >= 1.0:
            return float("inf")
        return self.mean_wait() / (1.0 - p_abn)

    def mean_system_time(self) -> float:
        """E[W] = mean time in system (for all customers including abandoners).

        E[W] = E[N] / λ
        """
        return self.mean_system_size() / self.arrival_rate

    def service_level(self, target_time: float) -> float:
        """Fraction of offered calls answered within target_time seconds.

        Computed from the birth-death steady-state:
        SL(t) = (1 - P(wait)) + P(wait) · P(served within t | waited)

        For a customer at position j in queue, the time to service is
        the sum of j exponential stages (each at rate c·μ + k·θ for the
        abandonment-adjusted system).  We approximate the conditional
        wait-given-wait distribution as exponential with rate
        (c·μ + θ - λ_eff) where λ_eff = λ(1-P_abandon), which is the
        net drain rate from queue.

        The fraction answered (not abandoned) among those who waited is
        c·μ / (c·μ + θ) at the head-of-line, giving
        P(answered within t | waited) ≈ p_serve · (1 - exp(-drain · t))

        Args:
            target_time: Target answer time in seconds.

        Returns:
            Probability that a call is answered within target_time.
        """
        if target_time < 0:
            return 0.0

        pw = self.prob_wait()
        if pw <= 0:
            return 1.0

        pa = self.prob_abandon()
        c_mu = self.servers * self.service_rate
        theta = self.patience_rate

        # For the head-of-line customer: competes between service (rate c·μ)
        # and abandonment (rate θ).  P(served | head-of-line) = c·μ/(c·μ+θ)
        p_serve_hol = c_mu / (c_mu + theta)

        # The conditional wait time (given waited) has an approximate
        # drain rate = (c·μ + θ - λ·(1 - p_abandon))
        # This accounts for: servers completing (freeing slots) + abandonment
        # minus new arrivals that pass through
        lambda_eff = self.arrival_rate * (1.0 - pa)
        drain_rate = c_mu + theta - lambda_eff
        if drain_rate <= 0:
            # Extreme overload: drain by abandonment only
            drain_rate = theta

        # P(wait < t | waited) = 1 - exp(-drain_rate · t)
        p_exit_by_t = 1.0 - math.exp(-drain_rate * target_time)

        # Of those who exit queue by time t, fraction p_serve_hol are served
        # (vs abandoned)
        pa_given_wait = min(pa / pw, 1.0) if pw > 0 else 0.0
        p_served_given_wait = 1.0 - pa_given_wait

        # SL = (immediate service) + (waited and served within t)
        sl = (1.0 - pw) + pw * p_served_given_wait * p_exit_by_t
        return min(max(sl, 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "ErlangA",
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "servers": self.servers,
            "patience_rate": self.patience_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErlangA:
        return cls(
            arrival_rate=data["arrival_rate"],
            service_rate=data["service_rate"],
            servers=data["servers"],
            patience_rate=data["patience_rate"],
        )

    def summary(self) -> dict[str, float]:
        """Extended summary including abandonment metrics."""
        base = super().summary()
        base["prob_wait"] = self.prob_wait()
        base["prob_abandon"] = self.prob_abandon()
        base["mean_wait_answered_ASA"] = self.mean_wait_answered()
        base["service_level_20s"] = self.service_level(20.0)
        return base
