"""Aligner-built NetworkSimplex configs gate warm repair on large components."""
import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, WarmMode
import wnetalign
from wnetalign import WNetAligner, Spectrum

pytestmark = pytest.mark.skipif(not hasattr(NetworkSimplex(), "repair_max_nodes"),
                                reason="wnet without repair_max_nodes")


def dense_1d(n=3000, seed=31):
    # 3000 peaks ~0.0033 apart with a 0.03 window: one connected component.
    rng = np.random.default_rng(seed)
    e = np.sort(rng.uniform(0, 10, n))[None, :]
    ei = rng.lognormal(0, 1, n)
    t = e + 0.01 * np.sin(e) + rng.normal(0, 0.003, e.shape)
    ti = ei * rng.lognormal(0, 0.2, n)
    return Spectrum(e, ei), [Spectrum(t, ti)]


def test_default_config_gates_large_components():
    emp, theos = dense_1d(50)
    a = WNetAligner(emp, theos, DistanceMetric.L1, 0.03, 0.03)
    assert a.solver.repair_max_nodes == wnetalign.aligner.REPAIR_MAX_NODES
    a = WNetAligner(emp, theos, DistanceMetric.L1, 0.03, 0.03, method="network_simplex")
    assert a.solver.repair_max_nodes == wnetalign.aligner.REPAIR_MAX_NODES


def test_caller_config_is_used_unchanged():
    emp, theos = dense_1d(50)
    cfg = NetworkSimplex()
    a = WNetAligner(emp, theos, DistanceMetric.L1, 0.03, 0.03, solver=cfg)
    assert a.solver.repair_max_nodes == -1


def test_gated_alignment_matches_cold_restarts():
    emp, theos = dense_1d()
    gated = WNetAligner(emp, theos, DistanceMetric.L1, 0.03, 0.03)
    cold_cfg = NetworkSimplex()
    cold_cfg.warm = WarmMode.NONE
    cold = WNetAligner(emp, theos, DistanceMetric.L1, 0.03, 0.03, solver=cold_cfg)
    assert gated.no_subgraphs() == 1   # one 6000-node component: the gate is live
    for w in ([1.0], [0.7], [1.3], [0.9]):
        gated.set_point(w)
        cold.set_point(w)
        assert gated.total_cost() == pytest.approx(cold.total_cost(), rel=1e-12)
