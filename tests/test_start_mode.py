"""NetworkSimplex StartMode.Trash through WNetAligner: identical alignments."""
import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, WarmMode
from wnetalign import WNetAligner, Spectrum

wnet_cpp = pytest.importorskip("wnet.wnet_cpp")
if not hasattr(wnet_cpp, "StartMode"):
    pytest.skip("wnet without StartMode", allow_module_level=True)
StartMode = wnet_cpp.StartMode


def shifted_pair(dim, n=400, seed=5):
    rng = np.random.default_rng(seed)
    e = rng.uniform(0, 20, (dim, n))
    ei = rng.lognormal(0, 1, n)
    t = e + 0.05 + rng.normal(0, 0.01, e.shape)
    return Spectrum(e, ei), [Spectrum(t, ei * rng.lognormal(0, 0.2, n))]


def config(start, warm):
    cfg = NetworkSimplex()
    cfg.start = start
    cfg.warm = warm
    return cfg


@pytest.mark.parametrize("dim", [1, 2, 4])
@pytest.mark.parametrize("warm", [WarmMode.NONE, WarmMode.Dual, WarmMode.DualRatio])
@pytest.mark.parametrize("trash", [dict(trash_cost=0.2),
                                   dict(experimental_trash_cost=0.2, theoretical_trash_cost=0.3)])
def test_trash_start_same_alignment(dim, warm, trash):
    emp, theos = shifted_pair(dim)
    a = WNetAligner(emp, theos, DistanceMetric.L1, 0.3, solver=config(StartMode.Artificial, warm), **trash)
    b = WNetAligner(emp, theos, DistanceMetric.L1, 0.3, solver=config(StartMode.Trash, warm), **trash)
    for w in ([1.0], [0.6], [1.4], [1.0]):
        a.set_point(w)
        b.set_point(w)
        assert a.total_cost() == b.total_cost()
