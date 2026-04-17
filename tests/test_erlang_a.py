"""Tests for Erlang A (M/M/c+M) model."""

from __future__ import annotations

import math

import pytest

from pyqueueing import ErlangA, MMC


class TestErlangABasic:
    def setup_method(self):
        # 100 calls/hr, 5 min handle time (12/hr), 10 agents, avg patience 2 min (30/hr)
        self.q = ErlangA(
            arrival_rate=100, service_rate=12, servers=10, patience_rate=30
        )

    def test_prob_wait_range(self):
        pw = self.q.prob_wait()
        assert 0.0 <= pw <= 1.0

    def test_prob_abandon_range(self):
        pa = self.q.prob_abandon()
        assert 0.0 <= pa <= 1.0

    def test_abandon_less_than_wait(self):
        """Not everyone who waits abandons."""
        assert self.q.prob_abandon() <= self.q.prob_wait()

    def test_mean_wait_positive(self):
        assert self.q.mean_wait() >= 0.0

    def test_mean_wait_answered_positive(self):
        asa = self.q.mean_wait_answered()
        assert asa >= 0.0
        # ASA >= mean wait (since it excludes zero-wait customers who don't wait)
        assert asa >= self.q.mean_wait() - 1e-10

    def test_service_level_monotone(self):
        """SL should increase with target_time."""
        prev = 0.0
        for t in [0.0, 5.0, 10.0, 20.0, 60.0]:
            sl = self.q.service_level(t)
            assert sl >= prev - 1e-10
            prev = sl

    def test_service_level_range(self):
        assert 0.0 <= self.q.service_level(20.0) <= 1.0

    def test_utilization_less_than_one(self):
        assert 0.0 < self.q.utilization() < 1.0


class TestErlangAVsMMC:
    """With very low patience rate (very patient customers), Erlang A ≈ Erlang C."""

    def test_low_impatience_approaches_mmc(self):
        lam, mu, c = 10.0, 3.0, 5
        # θ very small → almost no abandonment
        ea = ErlangA(arrival_rate=lam, service_rate=mu, servers=c, patience_rate=0.001)
        mmc = MMC(arrival_rate=lam, service_rate=mu, servers=c)

        assert math.isclose(ea.prob_wait(), mmc.prob_wait(), rel_tol=0.05)
        assert math.isclose(ea.mean_wait(), mmc.mean_wait(), rel_tol=0.05)
        assert ea.prob_abandon() < 0.01  # negligible abandonment


class TestErlangAHighLoad:
    """With impatience, the system is stable even when λ > c·μ."""

    def test_overloaded_stable(self):
        # λ=100, c·μ=10·8=80, so λ > c·μ but impatience keeps it stable
        q = ErlangA(arrival_rate=100, service_rate=8, servers=10, patience_rate=5)
        assert q.prob_abandon() > 0  # significant abandonment expected
        assert q.mean_system_size() < float("inf")


class TestErlangASerialization:
    def test_round_trip(self):
        q = ErlangA(arrival_rate=50, service_rate=10, servers=6, patience_rate=2)
        d = q.to_dict()
        q2 = ErlangA.from_dict(d)
        assert math.isclose(q.prob_abandon(), q2.prob_abandon())


class TestErlangALittleLaw:
    def test_littles_law(self):
        q = ErlangA(arrival_rate=50, service_rate=10, servers=6, patience_rate=2)
        # Little's law: Lq = λ · Wq
        assert math.isclose(
            q.mean_queue_length(),
            q.arrival_rate * q.mean_wait(),
            rel_tol=1e-8,
        )
