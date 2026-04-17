"""Tests for wait_time_cdf / wait_time_pdf on MM1 and MMC."""

from __future__ import annotations

import math

import pytest

from pyqueueing import MM1, MMC


class TestMM1WaitTimeCDF:
    def setup_method(self):
        self.q = MM1(arrival_rate=2.0, service_rate=3.0)

    def test_cdf_at_zero(self):
        # P(Wq <= 0) = 1 - ρ
        rho = self.q.utilization()
        assert math.isclose(self.q.wait_time_cdf(0.0), 1.0 - rho, rel_tol=1e-10)

    def test_cdf_negative(self):
        assert self.q.wait_time_cdf(-1.0) == 0.0

    def test_cdf_large_t(self):
        assert self.q.wait_time_cdf(100.0) > 0.999

    def test_cdf_monotone(self):
        prev = 0.0
        for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
            val = self.q.wait_time_cdf(t)
            assert val >= prev
            prev = val

    def test_pdf_integrates_to_rho(self):
        """The continuous part integrates to ρ (mass at 0 is 1-ρ)."""
        from scipy.integrate import quad

        area, _ = quad(self.q.wait_time_pdf, 1e-10, 50)
        rho = self.q.utilization()
        assert math.isclose(area, rho, rel_tol=1e-4)

    def test_pdf_negative(self):
        assert self.q.wait_time_pdf(-1.0) == 0.0


class TestMMCWaitTimePDF:
    def setup_method(self):
        self.q = MMC(arrival_rate=10.0, service_rate=3.0, servers=4)

    def test_pdf_integrates_to_prob_wait(self):
        from scipy.integrate import quad

        area, _ = quad(self.q.wait_time_pdf, 1e-10, 50)
        assert math.isclose(area, self.q.prob_wait(), rel_tol=1e-4)

    def test_cdf_pdf_consistent(self):
        """CDF derivative should match PDF."""
        t = 1.0
        dt = 1e-6
        numerical_pdf = (self.q.wait_time_cdf(t + dt) - self.q.wait_time_cdf(t)) / dt
        assert math.isclose(numerical_pdf, self.q.wait_time_pdf(t), rel_tol=1e-3)
