"""Cross-model consistency tests.

Little's Law: L = λW, Lq = λWq
Model equivalence tests.
"""

from __future__ import annotations

import math

from pyqueueing import MM1, MMC, MM1K, MMcK, MG1, ErlangB, ErlangC


class TestLittlesLaw:
    """L = λW and Lq = λWq must hold for all models."""

    def _check_littles_law(self, model, lam_eff=None):
        """Verify Little's Law for a model."""
        lam = lam_eff if lam_eff is not None else getattr(model, "arrival_rate")
        L = model.mean_system_size()  # type: ignore[union-attr]
        W = model.mean_system_time()  # type: ignore[union-attr]
        Lq = model.mean_queue_length()  # type: ignore[union-attr]
        Wq = model.mean_wait()  # type: ignore[union-attr]

        assert math.isclose(L, lam * W, rel_tol=1e-8), f"L={L}, λW={lam*W}"
        assert math.isclose(Lq, lam * Wq, rel_tol=1e-8), f"Lq={Lq}, λWq={lam*Wq}"

    def test_mm1(self) -> None:
        self._check_littles_law(MM1(arrival_rate=2.0, service_rate=3.0))

    def test_mmc(self) -> None:
        self._check_littles_law(MMC(arrival_rate=10.0, service_rate=3.0, servers=4))

    def test_mm1k(self) -> None:
        q = MM1K(arrival_rate=2.0, service_rate=3.0, capacity=10)
        self._check_littles_law(q, lam_eff=q.effective_arrival_rate())

    def test_mm1k_rho_gt_1(self) -> None:
        q = MM1K(arrival_rate=5.0, service_rate=3.0, capacity=10)
        self._check_littles_law(q, lam_eff=q.effective_arrival_rate())

    def test_mmck(self) -> None:
        q = MMcK(arrival_rate=10.0, service_rate=3.0, servers=4, capacity=10)
        self._check_littles_law(q, lam_eff=q.effective_arrival_rate())

    def test_mg1(self) -> None:
        self._check_littles_law(MG1(arrival_rate=2.0, service_rate=3.0, service_cv=1.5))

    def test_mg1_deterministic(self) -> None:
        self._check_littles_law(MG1(arrival_rate=2.0, service_rate=3.0, service_cv=0.0))


class TestModelEquivalence:
    """Cross-model consistency checks."""

    def test_mmc_c1_equals_mm1(self) -> None:
        lam, mu = 2.0, 3.0
        mm1 = MM1(arrival_rate=lam, service_rate=mu)
        mmc = MMC(arrival_rate=lam, service_rate=mu, servers=1)
        for attr in ["utilization", "mean_queue_length", "mean_system_size", "mean_wait", "mean_system_time"]:
            v1 = getattr(mm1, attr)()
            vc = getattr(mmc, attr)()
            assert math.isclose(v1, vc, rel_tol=1e-10), f"{attr}: MM1={v1}, MMC(1)={vc}"

    def test_erlang_c_equals_mmc_prob_wait(self) -> None:
        lam, mu, c = 10.0, 3.0, 4
        ec = ErlangC(arrival_rate=lam, service_rate=mu, servers=c)
        mmc = MMC(arrival_rate=lam, service_rate=mu, servers=c)
        assert math.isclose(ec.prob_wait(), mmc.prob_wait(), rel_tol=1e-12)

    def test_mmck_cap_eq_c_equals_erlang_b(self) -> None:
        lam, mu, c = 10.0, 1.0, 12
        mmck = MMcK(arrival_rate=lam, service_rate=mu, servers=c, capacity=c)
        eb = ErlangB(arrival_rate=lam, service_rate=mu, servers=c)
        assert math.isclose(mmck.prob_block(), eb.prob_block(), rel_tol=1e-10)

    def test_mg1_cv1_equals_mm1(self) -> None:
        lam, mu = 2.0, 3.0
        mm1 = MM1(arrival_rate=lam, service_rate=mu)
        mg1 = MG1(arrival_rate=lam, service_rate=mu, service_cv=1.0)
        assert math.isclose(mm1.mean_queue_length(), mg1.mean_queue_length(), rel_tol=1e-10)
        assert math.isclose(mm1.mean_system_time(), mg1.mean_system_time(), rel_tol=1e-10)
