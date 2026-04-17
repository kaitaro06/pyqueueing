"""Tests for CallCenterPlanner."""

from __future__ import annotations

import math

from pyqueueing import CallCenterPlanner


class TestCallCenterPlannerBasic:
    def setup_method(self):
        # 100 calls/hr, 5 min AHT (12/hr), avg patience 2 min (30/hr)
        self.planner = CallCenterPlanner(
            arrival_rate=100, service_rate=12, patience_rate=30
        )

    def test_required_agents_service_level(self):
        result = self.planner.required_agents(target_service_level=(20, 0.80))
        assert "erlang_c" in result
        assert "erlang_a" in result
        # Erlang A should need fewer agents (accounts for abandonment)
        assert result["erlang_a"] <= result["erlang_c"]
        # Both should be reasonable
        assert result["erlang_c"] >= int(self.planner.offered_load)
        assert result["erlang_a"] >= int(self.planner.offered_load)

    def test_required_agents_abandon_rate(self):
        result = self.planner.required_agents(target_abandon_rate=0.05)
        assert "erlang_a" in result
        assert result["erlang_a"] > 0

    def test_required_agents_asa(self):
        result = self.planner.required_agents(target_asa=30.0)
        assert "erlang_a" in result
        assert result["erlang_a"] > 0

    def test_multiple_targets_error(self):
        import pytest
        with pytest.raises(ValueError):
            self.planner.required_agents(
                target_service_level=(20, 0.80),
                target_asa=30.0,
            )


class TestCallCenterPlannerNoImpatience:
    def test_erlang_c_only(self):
        planner = CallCenterPlanner(arrival_rate=50, service_rate=10)
        result = planner.required_agents(target_service_level=(20, 0.80))
        assert "erlang_c" in result
        assert "erlang_a" not in result


class TestStaffingTable:
    def test_table_structure(self):
        planner = CallCenterPlanner(
            arrival_rate=50, service_rate=10, patience_rate=5
        )
        table = planner.staffing_table(range(6, 10))
        assert len(table) == 4
        assert table[0]["servers"] == 6
        assert "utilization" in table[0]
        assert "prob_abandon" in table[0]


class TestCostOptimal:
    def test_finds_optimum(self):
        planner = CallCenterPlanner(
            arrival_rate=50, service_rate=10, patience_rate=5
        )
        result = planner.cost_optimal_staffing(
            agent_cost=20.0,
            abandon_cost=50.0,
        )
        assert "optimal_servers" in result
        assert "total_cost" in result
        assert result["optimal_servers"] >= int(planner.offered_load)
