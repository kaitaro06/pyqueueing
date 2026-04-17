"""Tests for Erlang C model."""

import math

import pytest

from pyqueueing import ErlangC, MMC


class TestErlangCConsistency:
    """ErlangC.prob_wait() must equal MMC.prob_wait()."""

    def test_prob_wait_matches_mmc(self) -> None:
        lam, mu, c = 10.0, 3.0, 4
        ec = ErlangC(arrival_rate=lam, service_rate=mu, servers=c)
        mmc = MMC(arrival_rate=lam, service_rate=mu, servers=c)
        assert math.isclose(ec.prob_wait(), mmc.prob_wait(), rel_tol=1e-12)

    def test_all_metrics_match_mmc(self) -> None:
        lam, mu, c = 10.0, 3.0, 4
        ec = ErlangC(arrival_rate=lam, service_rate=mu, servers=c)
        mmc = MMC(arrival_rate=lam, service_rate=mu, servers=c)
        assert math.isclose(ec.mean_wait(), mmc.mean_wait(), rel_tol=1e-12)
        assert math.isclose(ec.mean_queue_length(), mmc.mean_queue_length(), rel_tol=1e-12)


class TestErlangCRequiredServers:
    def test_target_wait_prob(self) -> None:
        ec = ErlangC(arrival_rate=100, service_rate=12)
        c = ec.required_servers(target_wait_prob=0.2)
        assert isinstance(c, int)
        assert c >= math.ceil(100 / 12) + 1

        # Verify the solution meets the constraint
        mmc = MMC(arrival_rate=100, service_rate=12, servers=c)
        assert mmc.prob_wait() <= 0.2

        # And c-1 does not
        if c > math.ceil(100 / 12) + 1:
            mmc_prev = MMC(arrival_rate=100, service_rate=12, servers=c - 1)
            assert mmc_prev.prob_wait() > 0.2

    def test_target_mean_wait(self) -> None:
        ec = ErlangC(arrival_rate=10, service_rate=3)
        c = ec.required_servers(target_mean_wait=0.1)
        mmc = MMC(arrival_rate=10, service_rate=3, servers=c)
        assert mmc.mean_wait() <= 0.1

    def test_target_service_level(self) -> None:
        ec = ErlangC(arrival_rate=10, service_rate=3)
        c = ec.required_servers(target_service_level=(5.0, 0.80))
        mmc = MMC(arrival_rate=10, service_rate=3, servers=c)
        assert mmc.wait_time_cdf(5.0) >= 0.80

    def test_no_target_raises(self) -> None:
        ec = ErlangC(arrival_rate=10, service_rate=3)
        with pytest.raises(ValueError, match="Exactly one"):
            ec.required_servers()

    def test_multiple_targets_raises(self) -> None:
        ec = ErlangC(arrival_rate=10, service_rate=3)
        with pytest.raises(ValueError, match="Exactly one"):
            ec.required_servers(target_wait_prob=0.2, target_mean_wait=1.0)


class TestErlangCNoServers:
    def test_no_servers_raises_on_metric(self) -> None:
        ec = ErlangC(arrival_rate=10, service_rate=3)
        with pytest.raises(ValueError, match="servers not set"):
            ec.prob_wait()
