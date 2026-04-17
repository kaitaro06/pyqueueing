"""Tests for M/M/∞ model."""

from __future__ import annotations

import math

import pytest

from pyqueueing import MMInf


class TestMMInfBasic:
    def setup_method(self):
        self.q = MMInf(arrival_rate=10.0, service_rate=2.0)

    def test_mean_system_size(self):
        # E[L] = λ/μ = 10/2 = 5
        assert math.isclose(self.q.mean_system_size(), 5.0)

    def test_mean_system_time(self):
        # E[W] = 1/μ = 0.5
        assert math.isclose(self.q.mean_system_time(), 0.5)

    def test_no_queue(self):
        assert self.q.mean_queue_length() == 0.0
        assert self.q.mean_wait() == 0.0

    def test_utilization_zero(self):
        # Infinite servers → utilization is 0
        assert self.q.utilization() == 0.0

    def test_prob_n_poisson(self):
        """p_n should follow Poisson(a=5)."""
        a = 5.0
        for n in [0, 1, 3, 10]:
            expected = math.exp(-a) * a**n / math.factorial(n)
            assert math.isclose(self.q.prob_n(n), expected, rel_tol=1e-10)

    def test_prob_sum_to_one(self):
        total = sum(self.q.prob_n(n) for n in range(50))
        assert math.isclose(total, 1.0, rel_tol=1e-8)

    def test_prob_n_negative(self):
        with pytest.raises(ValueError):
            self.q.prob_n(-1)


class TestMMInfLittleLaw:
    def test_littles_law(self):
        q = MMInf(arrival_rate=7.0, service_rate=3.0)
        # L = λ·W
        assert math.isclose(
            q.mean_system_size(),
            q.arrival_rate * q.mean_system_time(),
            rel_tol=1e-10,
        )


class TestMMInfSerialization:
    def test_round_trip(self):
        q = MMInf(arrival_rate=5.0, service_rate=2.0)
        d = q.to_dict()
        q2 = MMInf.from_dict(d)
        assert math.isclose(q.mean_system_size(), q2.mean_system_size())
