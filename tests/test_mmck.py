"""Tests for M/M/c/K model."""

import math

from pyqueueing import MMcK


class TestMMcKBasic:
    def setup_method(self) -> None:
        self.q = MMcK(arrival_rate=10.0, service_rate=3.0, servers=4, capacity=10)

    def test_prob_sum(self) -> None:
        total = sum(self.q.prob_n(k) for k in range(11))
        assert math.isclose(total, 1.0, abs_tol=1e-12)

    def test_block_prob_range(self) -> None:
        assert 0 < self.q.prob_block() < 1


class TestMMcKMatchesErlangB:
    """MMcK(capacity=servers) should equal Erlang B."""

    def test_matches_erlang_b(self) -> None:
        from pyqueueing import ErlangB

        lam, mu, c = 10.0, 1.0, 12
        mmck = MMcK(arrival_rate=lam, service_rate=mu, servers=c, capacity=c)
        eb = ErlangB(arrival_rate=lam, service_rate=mu, servers=c)
        assert math.isclose(mmck.prob_block(), eb.prob_block(), rel_tol=1e-10)


class TestMMcKMatchesMMC:
    """MMcK with large K should approach MMC."""

    def test_large_k(self) -> None:
        from pyqueueing import MMC

        lam, mu, c = 10.0, 3.0, 4
        mmck = MMcK(arrival_rate=lam, service_rate=mu, servers=c, capacity=200)
        mmc = MMC(arrival_rate=lam, service_rate=mu, servers=c)
        assert math.isclose(mmck.mean_system_size(), mmc.mean_system_size(), rel_tol=1e-4)
