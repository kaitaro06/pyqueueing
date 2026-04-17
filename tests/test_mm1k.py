"""Tests for M/M/1/K model.

Analytical check:
    ρ=1, K=1: p_0 = p_1 = 1/2
"""

import math

import pytest

from pyqueueing import MM1K


class TestMM1KRhoOne:
    """ρ = 1 special case."""

    def setup_method(self) -> None:
        self.q = MM1K(arrival_rate=3.0, service_rate=3.0, capacity=5)

    def test_prob_uniform(self) -> None:
        for k in range(6):
            assert math.isclose(self.q.prob_n(k), 1.0 / 6.0, rel_tol=1e-10)

    def test_mean_system_size(self) -> None:
        assert math.isclose(self.q.mean_system_size(), 2.5)

    def test_prob_sum(self) -> None:
        total = sum(self.q.prob_n(k) for k in range(6))
        assert math.isclose(total, 1.0)


class TestMM1KRhoLessThanOne:
    """ρ < 1"""

    def setup_method(self) -> None:
        self.q = MM1K(arrival_rate=2.0, service_rate=3.0, capacity=10)

    def test_prob_sum(self) -> None:
        total = sum(self.q.prob_n(k) for k in range(11))
        assert math.isclose(total, 1.0, abs_tol=1e-12)

    def test_block_prob_small(self) -> None:
        # With ρ < 1 and large K, blocking should be very small
        assert self.q.prob_block() < 0.01

    def test_outside_range(self) -> None:
        assert self.q.prob_n(-1) == 0.0
        assert self.q.prob_n(11) == 0.0


class TestMM1KRhoGreaterThanOne:
    """ρ > 1 — still stable for finite capacity."""

    def test_no_error(self) -> None:
        q = MM1K(arrival_rate=5.0, service_rate=3.0, capacity=10)
        assert 0 < q.prob_block() < 1
        assert q.mean_system_size() > 0


class TestMM1KLargeK:
    """K → large should approach M/M/1."""

    def test_converges_to_mm1(self) -> None:
        from pyqueueing import MM1

        lam, mu = 2.0, 3.0
        mm1 = MM1(arrival_rate=lam, service_rate=mu)
        mm1k = MM1K(arrival_rate=lam, service_rate=mu, capacity=100)

        assert math.isclose(mm1.mean_system_size(), mm1k.mean_system_size(), rel_tol=1e-6)
        assert math.isclose(mm1.mean_wait(), mm1k.mean_wait(), rel_tol=1e-4)
