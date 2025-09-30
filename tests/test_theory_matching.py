import math
import numpy as np

from wnetalign.spectrum import Spectrum_1D
from wnetalign.aligner import WNetAligner as Solver


def test_matching():
    s1 = Spectrum_1D([0], [10])
    s2 = Spectrum_1D([1, 2], [5, 5])
    wasserstein_network = Solver(
        s1, [s2], lambda x, y: np.linalg.norm(x - y, axis=0), 100, 10, 10000
    )
    wasserstein_network.set_point([1])
    assert math.isclose(wasserstein_network.total_cost(), 15.0)


def test_matching2():
    s1 = Spectrum_1D([0], [10])
    s2 = Spectrum_1D([1], [4])
    s3 = Spectrum_1D([2], [6])
    wasserstein_network = Solver(
        s1, [s2, s3], lambda x, y: np.linalg.norm(x - y, axis=0), 100, 10, 10000
    )
    wasserstein_network.set_point([1, 1])
    assert math.isclose(wasserstein_network.total_cost(), 16.0)


def test_matching3():
    s1 = Spectrum_1D([0], [10])
    s2 = Spectrum_1D([1], [4])
    s3 = Spectrum_1D([200], [6])
    wasserstein_network = Solver(
        s1, [s2, s3], lambda x, y: np.linalg.norm(x - y, axis=0), 10, 10, 100
    )
    wasserstein_network.set_point([1, 1])
    wasserstein_network.print_diagnostics()
    assert math.isclose(wasserstein_network.total_cost(), 64.0)


if __name__ == "__main__":
    test_matching()
    test_matching2()
    test_matching3()
    print("Everything passed")
