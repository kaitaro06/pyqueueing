"""Tests for M/G/1 model.

Special cases:
    M/M/1: Cs=1 → b²=2/μ², should match MM1 results
    M/D/1: Cs=0 → b²=1/μ², Lq is half of M/M/1
"""

import math

import pytest

from pyqueueing import MG1, MM1


class TestMG1MatchesMM1:
    """M/G/1 with Cs=1 (exponential service) should match M/M/1."""

    def test_all_metrics(self) -> None:
        lam, mu = 2.0, 3.0
        mm1 = MM1(arrival_rate=lam, service_rate=mu)
        mg1 = MG1(arrival_rate=lam, service_rate=mu, service_cv=1.0)

        assert math.isclose(mg1.utilization(), mm1.utilization())
        assert math.isclose(mg1.mean_queue_length(), mm1.mean_queue_length(), rel_tol=1e-10)
        assert math.isclose(mg1.mean_system_size(), mm1.mean_system_size(), rel_tol=1e-10)
        assert math.isclose(mg1.mean_wait(), mm1.mean_wait(), rel_tol=1e-10)
        assert math.isclose(mg1.mean_system_time(), mm1.mean_system_time(), rel_tol=1e-10)


class TestMG1DeterministicService:
    """M/D/1: Cs=0, Lq should be half of M/M/1."""

    def test_lq_half_of_mm1(self) -> None:
        lam, mu = 2.0, 3.0
        mm1 = MM1(arrival_rate=lam, service_rate=mu)
        md1 = MG1(arrival_rate=lam, service_rate=mu, service_cv=0.0)

        assert math.isclose(md1.mean_queue_length(), mm1.mean_queue_length() / 2.0, rel_tol=1e-10)


class TestMG1VarianceInput:
    """Using service_var instead of service_cv."""

    def test_var_matches_cv(self) -> None:
        lam, mu = 2.0, 3.0
        cv = 1.5
        var = cv**2 / mu**2
        q_cv = MG1(arrival_rate=lam, service_rate=mu, service_cv=cv)
        q_var = MG1(arrival_rate=lam, service_rate=mu, service_var=var)

        assert math.isclose(q_cv.mean_wait(), q_var.mean_wait(), rel_tol=1e-12)

    def test_must_specify_one(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            MG1(arrival_rate=2.0, service_rate=3.0)

    def test_cannot_specify_both(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            MG1(arrival_rate=2.0, service_rate=3.0, service_cv=1.0, service_var=0.1)
