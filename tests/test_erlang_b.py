"""Tests for Erlang B model.

Analytical checks:
    B(1, ρ) = ρ / (1 + ρ)
    Recursive: B(c, a) = a·B(c-1, a) / (c + a·B(c-1, a))
"""

import math

from pyqueueing import ErlangB
from pyqueueing.models.erlang_b import erlang_b_formula


class TestErlangBFormula:
    def test_b1(self) -> None:
        a = 5.0
        assert math.isclose(erlang_b_formula(1, a), a / (1 + a))

    def test_b2(self) -> None:
        a = 5.0
        b1 = a / (1 + a)
        b2_expected = a * b1 / (2 + a * b1)
        assert math.isclose(erlang_b_formula(2, a), b2_expected)

    def test_large_c_no_overflow(self) -> None:
        """c=200 should not overflow."""
        result = erlang_b_formula(200, 180.0)
        assert 0.0 < result < 1.0

    def test_probabilities_sum(self) -> None:
        """For M/M/c/c, Σp_k = 1 implicitly via p_K = B(c,a)."""
        q = ErlangB(arrival_rate=10.0, service_rate=1.0, servers=12)
        pb = q.prob_block()
        assert 0 < pb < 1


class TestErlangBRequiredServers:
    def test_required_servers(self) -> None:
        q = ErlangB(arrival_rate=10.0, service_rate=1.0, servers=1)
        c = q.required_servers(target_block_prob=0.01)
        assert isinstance(c, int)
        assert erlang_b_formula(c, 10.0) <= 0.01
        if c > 1:
            assert erlang_b_formula(c - 1, 10.0) > 0.01


class TestErlangBMetrics:
    def test_no_waiting(self) -> None:
        q = ErlangB(arrival_rate=10.0, service_rate=1.0, servers=12)
        assert q.mean_wait() == 0.0
        assert q.mean_queue_length() == 0.0
        assert math.isclose(q.mean_system_time(), 1.0)
