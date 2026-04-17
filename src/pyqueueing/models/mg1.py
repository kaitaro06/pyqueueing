"""M/G/1 queueing model — Pollaczek-Khinchine formula.

Single-server queue with Poisson arrivals and general service time distribution.

Analytical formulas:
    ρ = λb  where b = 1/μ = E[S] (mean service time)
    b² = E[S²] = Var[S] + b²  (second moment of service time)

    E[Lq] = λ²b² / (2(1-ρ))
    E[Wq] = λb² / (2(1-ρ))
    E[L]  = E[Lq] + ρ
    E[W]  = E[Wq] + b

    Using coefficient of variation Cs:
        b² = E[S²] = b²(1 + Cs²)   since Cs² = Var[S]/b²
        E[Lq] = ρ²(1 + Cs²) / (2(1-ρ))

    Special cases:
        M/M/1: Cs = 1, b² = 2/μ²
        M/D/1: Cs = 0, b² = 1/μ²
"""

from __future__ import annotations

from typing import Any

from pyqueueing.models.base import BaseQueueModel
from pyqueueing.utils import validate_non_negative, validate_positive, validate_stability


class MG1(BaseQueueModel):
    """M/G/1 queueing model (Pollaczek-Khinchine).

    Specify service time variability via exactly one of ``service_cv`` or ``service_var``.

    Args:
        arrival_rate: Arrival rate λ.
        service_rate: Service rate μ = 1/b.
        service_cv: Coefficient of variation of service time Cs (>= 0).
        service_var: Variance of service time Var[S] (>= 0).

    Examples:
        >>> q = MG1(arrival_rate=2.0, service_rate=3.0, service_cv=1.0)
        >>> round(q.mean_queue_length(), 4)
        1.3333
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        *,
        service_cv: float | None = None,
        service_var: float | None = None,
    ) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        validate_stability(arrival_rate, service_rate, servers=1)

        if (service_cv is None) == (service_var is None):
            raise ValueError("Specify exactly one of service_cv or service_var.")

        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)

        b = 1.0 / service_rate  # mean service time

        if service_cv is not None:
            validate_non_negative(service_cv, "service_cv")
            self.service_cv = float(service_cv)
            self.service_var = float(service_cv**2 * b**2)
        else:
            assert service_var is not None
            validate_non_negative(service_var, "service_var")
            self.service_var = float(service_var)
            self.service_cv = float((service_var**0.5) / b)

        # Second moment of service time: E[S²] = Var[S] + (E[S])²
        self._b2 = self.service_var + b**2

    @property
    def mean_service_time(self) -> float:
        """b = 1/μ"""
        return 1.0 / self.service_rate

    def utilization(self) -> float:
        """ρ = λb"""
        return self.arrival_rate / self.service_rate

    def mean_queue_length(self) -> float:
        """E[Lq] = λ²b⁽²⁾ / (2(1-ρ))"""
        rho = self.utilization()
        return self.arrival_rate**2 * self._b2 / (2.0 * (1.0 - rho))

    def mean_system_size(self) -> float:
        """E[L] = E[Lq] + ρ"""
        return self.mean_queue_length() + self.utilization()

    def mean_wait(self) -> float:
        """E[Wq] = λb⁽²⁾ / (2(1-ρ))"""
        rho = self.utilization()
        return self.arrival_rate * self._b2 / (2.0 * (1.0 - rho))

    def mean_system_time(self) -> float:
        """E[W] = E[Wq] + b"""
        return self.mean_wait() + self.mean_service_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "MG1",
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "service_cv": self.service_cv,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MG1:
        return cls(
            arrival_rate=data["arrival_rate"],
            service_rate=data["service_rate"],
            service_cv=data.get("service_cv"),
            service_var=data.get("service_var"),
        )
