"""Edge-case tests for numerical stability with large parameters."""

from __future__ import annotations

import math

import pytest

from pyqueueing import MM1, MMC, MM1K, MMcK, ErlangB, ErlangA, MMInf, MG1


class TestLargeServerMMC:
    """MMC with c >= 200 where factorial would overflow."""

    def test_c200(self):
        q = MMC(arrival_rate=180, service_rate=1, servers=200)
        assert 0.0 < q.utilization() < 1.0
        assert 0.0 <= q.prob_wait() <= 1.0
        assert q.mean_wait() >= 0

    def test_c500(self):
        q = MMC(arrival_rate=450, service_rate=1, servers=500)
        pw = q.prob_wait()
        assert 0.0 <= pw <= 1.0

    def test_prob_n_sums_close_to_one(self):
        q = MMC(arrival_rate=180, service_rate=1, servers=200)
        total = sum(q.prob_n(n) for n in range(400))
        assert math.isclose(total, 1.0, rel_tol=1e-3)


class TestLargeServerMMcK:
    def test_c200(self):
        q = MMcK(arrival_rate=180, service_rate=1, servers=200, capacity=300)
        assert 0.0 <= q.prob_block() <= 1.0
        total = sum(q.prob_n(n) for n in range(301))
        assert math.isclose(total, 1.0, rel_tol=1e-3)


class TestLargeServerErlangB:
    def test_c500(self):
        q = ErlangB(arrival_rate=450, service_rate=1, servers=500)
        pb = q.prob_block()
        assert 0.0 <= pb <= 1.0


class TestRhoNearOne:
    """Systems near the stability boundary."""

    def test_mm1_high_rho(self):
        q = MM1(arrival_rate=0.999, service_rate=1.0)
        assert q.mean_system_size() > 500  # should be ~999

    def test_mm1k_rho_exactly_one(self):
        q = MM1K(arrival_rate=1.0, service_rate=1.0, capacity=10)
        assert math.isclose(q.mean_system_size(), 5.0, rel_tol=1e-6)
        total = sum(q.prob_n(n) for n in range(11))
        assert math.isclose(total, 1.0, rel_tol=1e-10)


class TestMG1HighCV:
    def test_cv_5(self):
        q = MG1(arrival_rate=0.5, service_rate=1.0, service_cv=5.0)
        assert q.mean_queue_length() > 0
        # Little's law
        assert math.isclose(
            q.mean_system_size(),
            q.arrival_rate * q.mean_system_time(),
            rel_tol=1e-10,
        )


class TestErlangAOverload:
    """λ > cμ — system only stable due to impatience."""

    def test_extreme_overload(self):
        q = ErlangA(arrival_rate=200, service_rate=1, servers=100, patience_rate=5)
        assert q.prob_abandon() > 0.1  # significant abandonment
        assert q.mean_system_size() < float("inf")
        # Little's law on queue
        assert math.isclose(
            q.mean_queue_length(),
            q.arrival_rate * q.mean_wait(),
            rel_tol=1e-6,
        )
