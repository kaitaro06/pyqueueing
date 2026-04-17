"""Tests for M/M/1 model.

Textbook reference values:
    λ=2, μ=3 → ρ=2/3, Lq=4/3, Wq=2/3, W=1, L=2
    λ=0.5, μ=1 → ρ=0.5, L=1, p0=0.5, Wq=1
"""

import math

import pytest

from pyqueueing import MM1


class TestMM1Basic:
    """教科書既知値テスト"""

    def setup_method(self) -> None:
        self.q = MM1(arrival_rate=2.0, service_rate=3.0)

    def test_utilization(self) -> None:
        assert math.isclose(self.q.utilization(), 2.0 / 3.0)

    def test_mean_queue_length(self) -> None:
        # Lq = ρ²/(1-ρ) = (4/9)/(1/3) = 4/3
        assert math.isclose(self.q.mean_queue_length(), 4.0 / 3.0)

    def test_mean_system_size(self) -> None:
        # L = ρ/(1-ρ) = (2/3)/(1/3) = 2
        assert math.isclose(self.q.mean_system_size(), 2.0)

    def test_mean_wait(self) -> None:
        # Wq = ρ/(μ(1-ρ)) = (2/3)/(3·1/3) = 2/3
        assert math.isclose(self.q.mean_wait(), 2.0 / 3.0)

    def test_mean_system_time(self) -> None:
        # W = 1/(μ-λ) = 1
        assert math.isclose(self.q.mean_system_time(), 1.0)

    def test_prob_n(self) -> None:
        rho = 2.0 / 3.0
        assert math.isclose(self.q.prob_n(0), 1.0 - rho)
        assert math.isclose(self.q.prob_n(1), (1.0 - rho) * rho)

    def test_prob_sum_to_one(self) -> None:
        total = sum(self.q.prob_n(k) for k in range(200))
        assert math.isclose(total, 1.0, abs_tol=1e-10)


class TestMM1SecondExample:
    """λ=0.5, μ=1"""

    def setup_method(self) -> None:
        self.q = MM1(arrival_rate=0.5, service_rate=1.0)

    def test_mean_system_size(self) -> None:
        assert math.isclose(self.q.mean_system_size(), 1.0)

    def test_prob_0(self) -> None:
        assert math.isclose(self.q.prob_n(0), 0.5)


class TestMM1Validation:
    def test_unstable(self) -> None:
        with pytest.raises(ValueError, match="unstable"):
            MM1(arrival_rate=3.0, service_rate=2.0)

    def test_equal_rates(self) -> None:
        with pytest.raises(ValueError, match="unstable"):
            MM1(arrival_rate=3.0, service_rate=3.0)

    def test_negative_rate(self) -> None:
        with pytest.raises(ValueError):
            MM1(arrival_rate=-1.0, service_rate=3.0)

    def test_zero_rate(self) -> None:
        with pytest.raises(ValueError):
            MM1(arrival_rate=0.0, service_rate=3.0)


class TestMM1Serialization:
    def test_roundtrip(self) -> None:
        q = MM1(arrival_rate=2.0, service_rate=3.0)
        d = q.to_dict()
        q2 = MM1.from_dict(d)
        assert math.isclose(q.mean_wait(), q2.mean_wait())
