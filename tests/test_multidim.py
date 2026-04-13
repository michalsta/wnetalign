import numpy as np

from wnet import WassersteinNetwork
from wnet.distances import DistanceMetric
from wnetalign import WNetAligner
from wnetalign.spectrum import Spectrum, Spectrum_1D


def test_1d():
    empirical_spectrum = Spectrum_1D([1], [1])
    theoretical_spectrum = Spectrum_1D([2], [1])
    solver = WNetAligner(
        empirical_spectrum,
        [theoretical_spectrum],
        DistanceMetric.L2,
        10,
        10,
        100,
    )
    # solver = WassersteinSolver(E, [theoretical_spectrum], [SimpleTrash(10)])
    solver.set_point([1])
    assert solver.total_cost() == 1


def test_2d():
    s1_pos = np.array([[0, 1, 0], [0, 0, 1]])
    s1_int = np.array([1, 1, 1])
    s1 = Spectrum(s1_pos, s1_int)
    s2_pos = np.array([[1, 1, 0], [1, 0, 1]])
    s2_int = np.array([1, 1, 1])
    s2 = Spectrum(s2_pos, s2_int)
    solver = WNetAligner(
        s1, [s2], DistanceMetric.L2, 1000000, 1000
    )
    solver.set_point([1])
    # print(solver.run())
    print(solver.total_cost())
    print(1000*solver.total_cost())
    assert int(1000.0*solver.total_cost()) == 1414

    # new algo
    wasserstein_network = WassersteinNetwork(
        s1.positions_intensities_scaled(1000.0),
        [s2.positions_intensities_scaled(1000.0)],
        DistanceMetric.L2,
        5000,
    )
    wasserstein_network.add_simple_trash(10000)
    wasserstein_network.build()

    wasserstein_network.solve()
    print("New algo total cost:", wasserstein_network.total_cost() / 1000.0)
    print("New algo total cost (scaled):", 1000.0*wasserstein_network.total_cost())

    assert int(wasserstein_network.total_cost()/1000) == 1414


if __name__ == "__main__":
    test_1d()
    test_2d()
    print("Everything passed")
