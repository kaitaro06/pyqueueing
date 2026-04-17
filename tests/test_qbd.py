"""Tests for QBD model and Matrix Geometric solver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyqueueing import QBD


class TestQBDasMM1:
    """M/M/1 queue represented as a scalar QBD — validates the solver."""

    def setup_method(self):
        lam, mu = 2.0, 3.0
        self.lam = lam
        self.mu = mu
        self.rho = lam / mu
        self.A0 = np.array([[lam]])
        self.A1 = np.array([[-(lam + mu)]])
        self.A2 = np.array([[mu]])
        self.qbd = QBD(self.A0, self.A1, self.A2)

    def test_rate_matrix(self):
        R = self.qbd.rate_matrix(method="iterative")
        assert R.shape == (1, 1)
        assert math.isclose(R[0, 0], self.rho, rel_tol=1e-8)

    def test_rate_matrix_log_reduction(self):
        R = self.qbd.rate_matrix(method="log_reduction")
        assert math.isclose(R[0, 0], self.rho, rel_tol=1e-6)

    def test_spectral_radius(self):
        self.qbd.rate_matrix()
        sr = self.qbd.spectral_radius()
        assert sr < 1.0
        assert math.isclose(sr, self.rho, rel_tol=1e-8)

    def test_is_stable(self):
        self.qbd.rate_matrix()
        assert self.qbd.is_stable()

    def test_mean_level(self):
        """Mean level should approximate E[L] = ρ/(1-ρ) for M/M/1."""
        self.qbd.rate_matrix()
        ml = self.qbd.mean_level()
        expected = self.rho / (1 - self.rho)
        assert math.isclose(ml, expected, rel_tol=0.05)

    def test_stationary_geometric(self):
        """π_k should be approximately (1-ρ)ρ^k for M/M/1."""
        self.qbd.rate_matrix()
        pi = self.qbd.stationary()
        # Check first few levels
        for k in range(min(5, len(pi))):
            expected = (1 - self.rho) * self.rho**k
            actual = float(np.sum(pi[k]))
            assert math.isclose(actual, expected, rel_tol=0.05), \
                f"Level {k}: expected {expected:.4f}, got {actual:.4f}"


class TestQBD2Phase:
    """M/M/1 modeled with 2 phases — tests matrix dimensions > 1."""

    def setup_method(self):
        # A simple 2-phase QBD where we know it's stable
        lam = 1.0
        mu = 2.0
        # Phase transitions within a level
        self.A0 = np.array([[lam, 0], [0, lam]])
        self.A1 = np.array([[-(lam + mu + 0.5), 0.5], [0.5, -(lam + mu + 0.5)]])
        self.A2 = np.array([[mu, 0], [0, mu]])
        self.qbd = QBD(self.A0, self.A1, self.A2)

    def test_rate_matrix_shape(self):
        R = self.qbd.rate_matrix()
        assert R.shape == (2, 2)

    def test_stable(self):
        self.qbd.rate_matrix()
        assert self.qbd.is_stable()

    def test_stationary_sums_to_one(self):
        self.qbd.rate_matrix()
        pi = self.qbd.stationary()
        total = sum(np.sum(lv) for lv in pi)
        assert math.isclose(total, 1.0, rel_tol=0.01)


class TestQBDValidation:
    def test_incompatible_shapes(self):
        A0 = np.array([[1, 0], [0, 1]])
        A1 = np.array([[1]])
        A2 = np.array([[1, 0], [0, 1]])
        with pytest.raises(ValueError):
            QBD(A0, A1, A2)

    def test_non_square(self):
        with pytest.raises(ValueError):
            QBD(np.array([[1, 2]]), np.array([[1]]), np.array([[1]]))


class TestQBDLevelDistribution:
    def test_level_distribution(self):
        lam, mu = 1.5, 3.0
        qbd = QBD(np.array([[lam]]), np.array([[-(lam + mu)]]), np.array([[mu]]))
        qbd.rate_matrix()
        ld = qbd.level_distribution()
        assert math.isclose(np.sum(ld), 1.0, rel_tol=0.01)
        # Should be monotonically decreasing for M/M/1
        for i in range(1, min(10, len(ld))):
            assert ld[i] <= ld[i - 1] + 1e-10
