"""Tests for M/M/c model."""

import math

import pytest

from pyqueueing import MMC


class TestMMCBasic:
    """λ=10, μ=3, c=4 → a=10/3, ρ=10/12=5/6"""

    def setup_method(self) -> None:
        self.q = MMC(arrival_rate=10.0, service_rate=3.0, servers=4)

    def test_utilization(self) -> None:
        assert math.isclose(self.q.utilization(), 10.0 / 12.0)

    def test_offered_load(self) -> None:
        assert math.isclose(self.q.offered_load, 10.0 / 3.0)

    def test_prob_wait_range(self) -> None:
        pw = self.q.prob_wait()
        assert 0.0 < pw < 1.0

    def test_prob_sum_to_one(self) -> None:
        total = sum(self.q.prob_n(k) for k in range(200))
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_wait_time_cdf_boundary(self) -> None:
        assert math.isclose(self.q.wait_time_cdf(0.0), 1.0 - self.q.prob_wait())
        assert math.isclose(self.q.wait_time_cdf(1000.0), 1.0, abs_tol=1e-10)


class TestMMCSingleServer:
    """MMC(c=1) should match MM1."""

    def test_matches_mm1(self) -> None:
        from pyqueueing import MM1

        lam, mu = 2.0, 3.0
        q1 = MM1(arrival_rate=lam, service_rate=mu)
        qc = MMC(arrival_rate=lam, service_rate=mu, servers=1)

        assert math.isclose(q1.utilization(), qc.utilization())
        assert math.isclose(q1.mean_queue_length(), qc.mean_queue_length(), rel_tol=1e-10)
        assert math.isclose(q1.mean_system_size(), qc.mean_system_size(), rel_tol=1e-10)
        assert math.isclose(q1.mean_wait(), qc.mean_wait(), rel_tol=1e-10)
        assert math.isclose(q1.mean_system_time(), qc.mean_system_time(), rel_tol=1e-10)


class TestMMCValidation:
    def test_unstable(self) -> None:
        with pytest.raises(ValueError, match="unstable"):
            MMC(arrival_rate=12.0, service_rate=3.0, servers=4)

    def test_float_servers(self) -> None:
        with pytest.raises(TypeError):
            MMC(arrival_rate=2.0, service_rate=3.0, servers=1.5)  # type: ignore[arg-type]
