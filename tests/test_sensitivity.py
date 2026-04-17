"""Tests for sensitivity analysis (sweep / sweep2d)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyqueueing import MM1, MMC
from pyqueueing.sensitivity import sweep, sweep2d


class TestSweep:
    def test_basic_sweep(self):
        result = sweep(
            MM1,
            {"service_rate": 3.0},
            "arrival_rate",
            [1.0, 2.0, 2.5],
        )
        assert "arrival_rate" in result
        assert "utilization" in result
        assert len(result["utilization"]) == 3
        np.testing.assert_allclose(result["utilization"], [1 / 3, 2 / 3, 5 / 6], rtol=1e-10)

    def test_unstable_returns_nan(self):
        result = sweep(
            MM1,
            {"service_rate": 3.0},
            "arrival_rate",
            [2.0, 3.5],  # 3.5 > 3.0 → unstable
        )
        assert math.isnan(result["utilization"][1])

    def test_custom_metrics(self):
        result = sweep(
            MM1,
            {"service_rate": 3.0},
            "arrival_rate",
            [1.0],
            metrics=["mean_wait"],
        )
        assert "mean_wait" in result
        assert "utilization" not in result


class TestSweep2D:
    def test_basic_2d(self):
        X, Y, Z = sweep2d(
            MMC,
            {"service_rate": 3.0},
            ("arrival_rate", [5.0, 8.0]),
            ("servers", [3, 4]),  # int values
            metric="utilization",
        )
        assert X.shape == (2, 2)
        assert Y.shape == (2, 2)
        assert Z.shape == (2, 2)
        # servers=3, arrival=5 → ρ = 5/(3*3) = 5/9
        assert math.isclose(Z[0, 0], 5 / 9, rel_tol=1e-10)
