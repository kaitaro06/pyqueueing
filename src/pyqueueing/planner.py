"""Call center staffing planner.

High-level utility that wraps Erlang C (M/M/c) and Erlang A (M/M/c+M) models
to provide practical staffing recommendations for call centers.

Supports:
    - Finding minimum servers for SLA targets
    - Staffing tables across varying demand
    - Cost-optimal staffing with abandon/wait penalties
"""

from __future__ import annotations

from typing import Any

from pyqueueing.models.erlang_a import ErlangA
from pyqueueing.models.erlang_c import ErlangC
from pyqueueing.utils import validate_positive


class CallCenterPlanner:
    """Call center staffing planner.

    Provides staffing recommendations using both Erlang C (no abandonment)
    and Erlang A (with abandonment) models.

    Args:
        arrival_rate: Call arrival rate λ (calls per unit time).
        service_rate: Service rate μ per agent (1 / average handle time).
        patience_rate: Abandonment rate θ (optional, enables Erlang A).
            If None, uses Erlang C only (conservative estimate).

    Examples:
        >>> planner = CallCenterPlanner(
        ...     arrival_rate=100,     # 100 calls/hour
        ...     service_rate=12,      # 5 min avg handle time → 12/hour
        ...     patience_rate=2.0,    # avg patience 30 sec = 2/min... or use matching units
        ... )
        >>> planner.required_agents(target_service_level=(20, 0.80))
        {'erlang_c': 11, 'erlang_a': 10}
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        patience_rate: float | None = None,
    ) -> None:
        validate_positive(arrival_rate, "arrival_rate")
        validate_positive(service_rate, "service_rate")
        if patience_rate is not None:
            validate_positive(patience_rate, "patience_rate")

        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self.patience_rate = float(patience_rate) if patience_rate is not None else None

    @property
    def offered_load(self) -> float:
        """a = λ/μ (Erlang offered load)."""
        return self.arrival_rate / self.service_rate

    def required_agents(
        self,
        *,
        target_service_level: tuple[float, float] | None = None,
        target_abandon_rate: float | None = None,
        target_asa: float | None = None,
        max_servers: int = 1000,
    ) -> dict[str, int]:
        """Find minimum number of agents to meet the specified target.

        Specify exactly one target. Returns both Erlang C and Erlang A
        estimates when patience_rate is provided.

        Args:
            target_service_level: (time, probability) e.g. (20, 0.80) means
                80% of calls answered within 20 seconds.
            target_abandon_rate: Maximum acceptable fraction of abandoned calls.
            target_asa: Maximum acceptable Average Speed of Answer (seconds).
            max_servers: Search upper bound.

        Returns:
            Dict with 'erlang_c' and optionally 'erlang_a' server counts.
        """
        targets_given = sum(
            x is not None
            for x in [target_service_level, target_abandon_rate, target_asa]
        )
        if targets_given != 1:
            raise ValueError("Specify exactly one target.")

        result: dict[str, int] = {}

        # --- Erlang C ---
        ec = ErlangC(arrival_rate=self.arrival_rate, service_rate=self.service_rate)
        if target_service_level is not None:
            t, prob = target_service_level
            result["erlang_c"] = ec.required_servers(target_service_level=(t, prob))
        elif target_asa is not None:
            result["erlang_c"] = ec.required_servers(target_mean_wait=target_asa)
        elif target_abandon_rate is not None:
            # Erlang C doesn't model abandonment — use wait prob as proxy
            result["erlang_c"] = ec.required_servers(target_wait_prob=target_abandon_rate)

        # --- Erlang A ---
        if self.patience_rate is not None:
            min_c = max(1, int(self.offered_load))
            for c in range(min_c, max_servers + 1):
                ea = ErlangA(
                    arrival_rate=self.arrival_rate,
                    service_rate=self.service_rate,
                    servers=c,
                    patience_rate=self.patience_rate,
                )
                if target_service_level is not None:
                    t, prob = target_service_level
                    if ea.service_level(t) >= prob:
                        result["erlang_a"] = c
                        break
                elif target_abandon_rate is not None:
                    if ea.prob_abandon() <= target_abandon_rate:
                        result["erlang_a"] = c
                        break
                elif target_asa is not None:
                    if ea.mean_wait_answered() <= target_asa:
                        result["erlang_a"] = c
                        break

        return result

    def staffing_table(
        self,
        server_range: range | list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate a staffing table showing metrics for each server count.

        Args:
            server_range: Range of server counts to evaluate.
                Defaults to offered_load .. offered_load + 15.

        Returns:
            List of dicts, one per server count, with key metrics.
        """
        if server_range is None:
            min_c = max(1, int(self.offered_load) + 1)
            server_range = range(min_c, min_c + 15)

        rows = []
        for c in server_range:
            row: dict[str, Any] = {"servers": c}

            # Erlang C metrics
            if c * self.service_rate > self.arrival_rate:
                ec_model = ErlangC(
                    arrival_rate=self.arrival_rate,
                    service_rate=self.service_rate,
                )
                # Use MMC internally via ErlangC
                from pyqueueing.models.mmc import MMC
                mmc = MMC(
                    arrival_rate=self.arrival_rate,
                    service_rate=self.service_rate,
                    servers=c,
                )
                row["utilization"] = mmc.utilization()
                row["prob_wait_erlc"] = mmc.prob_wait()
                row["mean_wait_erlc"] = mmc.mean_wait()
                row["sl_20s_erlc"] = mmc.wait_time_cdf(20.0)
            else:
                row["utilization"] = float("inf")
                row["prob_wait_erlc"] = 1.0
                row["mean_wait_erlc"] = float("inf")
                row["sl_20s_erlc"] = 0.0

            # Erlang A metrics
            if self.patience_rate is not None:
                ea = ErlangA(
                    arrival_rate=self.arrival_rate,
                    service_rate=self.service_rate,
                    servers=c,
                    patience_rate=self.patience_rate,
                )
                row["prob_wait_erla"] = ea.prob_wait()
                row["prob_abandon"] = ea.prob_abandon()
                row["mean_wait_erla"] = ea.mean_wait()
                row["asa"] = ea.mean_wait_answered()
                row["sl_20s_erla"] = ea.service_level(20.0)

            rows.append(row)

        return rows

    def cost_optimal_staffing(
        self,
        agent_cost: float,
        abandon_cost: float,
        wait_cost_per_second: float = 0.0,
        server_range: range | list[int] | None = None,
    ) -> dict[str, Any]:
        """Find the staffing level that minimizes total cost.

        Total cost = c · agent_cost + λ · P(abandon) · abandon_cost
                     + λ · E[Wq] · wait_cost_per_second

        Args:
            agent_cost: Cost per agent per unit time.
            abandon_cost: Cost per abandoned call.
            wait_cost_per_second: Cost per second of customer waiting.
            server_range: Range to search. Defaults to sensible range.

        Returns:
            Dict with optimal_servers, min_cost, and breakdown.
        """
        if self.patience_rate is None:
            raise ValueError("Cost optimization requires patience_rate (Erlang A model).")

        if server_range is None:
            min_c = max(1, int(self.offered_load))
            server_range = range(min_c, min_c + 30)

        best: dict[str, Any] | None = None

        for c in server_range:
            ea = ErlangA(
                arrival_rate=self.arrival_rate,
                service_rate=self.service_rate,
                servers=c,
                patience_rate=self.patience_rate,
            )
            staff_cost = c * agent_cost
            abn_cost = self.arrival_rate * ea.prob_abandon() * abandon_cost
            w_cost = self.arrival_rate * ea.mean_wait() * wait_cost_per_second
            total = staff_cost + abn_cost + w_cost

            if best is None or total < best["total_cost"]:
                best = {
                    "optimal_servers": c,
                    "total_cost": total,
                    "agent_cost": staff_cost,
                    "abandon_cost": abn_cost,
                    "wait_cost": w_cost,
                    "utilization": ea.utilization(),
                    "prob_abandon": ea.prob_abandon(),
                    "mean_wait": ea.mean_wait(),
                }

        assert best is not None
        return best
