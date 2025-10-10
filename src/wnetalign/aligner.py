from collections import namedtuple
from collections.abc import Callable, Sequence
import numpy as np

from wnet import Distribution, WassersteinNetwork


class WNetAligner:
    """
    Aligns an empirical spectrum to one or more theoretical spectra using a Wasserstein network approach.
    Alignment of two empirical spectra E1, E2 can be performed by setting E1 as the empirical_spectrum
    and E2 as the only element of theoretical_spectra.

    Parameters
    ----------
    empirical_spectrum : Distribution
        The empirical spectrum to be aligned.
    theoretical_spectra : Sequence[Distribution]
        A sequence of theoretical spectra to align against.
    distance_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function to compute the distance between empirical and theoretical peaks.
    max_distance : int or float
        Maximum allowed distance for matching peaks.
    trash_cost : int or float
        Cost for assigning unmatched peaks to 'trash'.
    scale_factor : None, int, or float, optional
        Scaling factor for intensities and costs. If None, it is computed automatically.

    Attributes
    ----------
    scale_factor : float
        The scaling factor used for intensities and costs.
    empirical_spectrum : Distribution
        The scaled empirical spectrum.
    theoretical_spectra : list[Distribution]
        The scaled theoretical spectra.
    graph : WassersteinNetwork
        The underlying Wasserstein network graph.
    point : Sequence[float] or np.ndarray or None
        The current point for solving the alignment.

    Methods
    -------
    set_point(point)
        Sets the point for solving the alignment and runs the solver.
    total_cost()
        Returns the total cost of the alignment, rescaled to original units.
    print()
        Prints a string representation of the underlying graph.
    flows()
        Returns a list of flows (alignments) between empirical and theoretical peaks.
    no_subgraphs()
        Returns the number of subgraphs in the alignment network.
    print_diagnostics(subgraphs_too=False)
        Prints diagnostic information about the alignment and optionally about each subgraph.
    """
    def __init__(
        self,
        empirical_spectrum : Distribution,
        theoretical_spectra : Sequence[Distribution],
        distance_function : Callable[[np.ndarray, np.ndarray], np.ndarray],
        max_distance : int | float,
        trash_cost : int | float,
        scale_factor: None | int | float = None,
    ) -> None:

        assert isinstance(empirical_spectrum, Distribution)
        assert isinstance(theoretical_spectra, Sequence)
        assert all(isinstance(t, Distribution) for t in theoretical_spectra)
        assert callable(distance_function)
        assert isinstance(max_distance, (int, float))
        assert isinstance(trash_cost, (int, float))
        assert scale_factor is None or isinstance(scale_factor, (int, float))

        if scale_factor is None:
            ALMOST_MAXINT = 2**60
            empirical_sum_intensity = empirical_spectrum.sum_intensities
            theoretical_sum_intensity = sum(
                t.sum_intensities for t in theoretical_spectra
            )
            max_sum_intensity = max(empirical_sum_intensity, theoretical_sum_intensity)
            scale_factor = np.sqrt(ALMOST_MAXINT / (max_sum_intensity * trash_cost))
            assert (
                scale_factor > 0
            ), "Can't auto-compute a sensible scale factor. You might have some luck with setting it manually, but it probably means something about your data or trash_cost is off."

        self.scale_factor = scale_factor
        self.empirical_spectrum = empirical_spectrum.scaled(scale_factor)
        self.theoretical_spectra = [t.scaled(scale_factor) for t in theoretical_spectra]

        def wrapped_dist(p, y):
            i = p.index
            x = p.positions[:, i : i + 1]
            return distance_function(x[: np.newaxis], y) * scale_factor

        self.graph = WassersteinNetwork(
            self.empirical_spectrum,
            self.theoretical_spectra,
            wrapped_dist,
            max_distance * scale_factor,
        )
        self.graph.add_simple_trash(int(trash_cost * scale_factor))
        self.graph.build()
        self.point = None

    def set_point(self, point : Sequence[float] | np.ndarray) -> None:
        """
        Set proportions of theoretical spectra and solve the graph at the given point.

        Parameters
        ----------
        point : Sequence[float] or np.ndarray
            Proportions for each theoretical spectrum.

        Returns
        -------
        None
        """
        self.point = point
        self.graph.solve(point)

    def total_cost(self) -> float:
        """
        Calculates the total cost of the graph. Can only be called after set_point().

        Returns:
            float: The normalized total cost.
        """
        return self.graph.total_cost() / self.scale_factor / self.scale_factor

    def print(self) -> None:
        """
        Prints a string representation of the graph associated with this aligner instance.

        Returns:
            None
        """
        print(str(self.graph))

    def flows(self) -> list[namedtuple]:
        """
        Computes and returns a list of flow information for each theoretical spectrum.

        Each flow is represented as a namedtuple containing the empirical peak index,
        theoretical peak index, and the scaled flow value (divided by self.scale_factor).

        Returns:
            list[namedtuple]: A list of Flow namedtuples, one for each theoretical
            spectrum, each containing:
                - empirical_peak_idx (int): Index of the empirical peak.
                - theoretical_peak_idx (int): Index of the theoretical peak.
                - flow (float): Scaled flow value between the peaks.
        """
        result = []
        for i in range(len(self.theoretical_spectra)):
            empirical_peak_idx, theoretical_peak_idx, flow = (
                self.graph.flows_for_target(i)
            )
            result.append(
                namedtuple(
                    "Flow", ["empirical_peak_idx", "theoretical_peak_idx", "flow"]
                )(empirical_peak_idx, theoretical_peak_idx, flow / self.scale_factor)
            )
        return result

    def no_subgraphs(self) -> int:
        """
        Returns the number of subgraphs in the underlying Wasserstein network.

        Returns:
            int: The number of subgraphs present in the graph.
        """
        return self.graph.no_subgraphs()

    def print_diagnostics(self, subgraphs_too=False):
        """
        Prints diagnostic information about the current state of the alignment.

        Parameters
        ----------
        subgraphs_too : bool, optional
            If True, prints diagnostics for each subgraph in addition to the overall graph.

        Diagnostics Printed
        ------------------
        - Number of subgraphs
        - Number of empirical nodes
        - Number of theoretical nodes
        - Matching density
        - Scale factor (and its log10 value)
        - Total cost

        If `subgraphs_too` is True, for each subgraph:
        - Number of empirical nodes
        - Number of theoretical nodes
        - Cost
        - Matching density
        - Theoretical spectra involved
        """
        print("Diagnostics:")
        print("No subgraphs:", self.graph.no_subgraphs())
        print("No empirical nodes:", self.graph.count_empirical_nodes())
        print("No theoretical nodes:", self.graph.count_theoretical_nodes())
        print("Matching density:", self.graph.matching_density())
        print(
            "Scale factor:", self.scale_factor, f" log10: {np.log10(self.scale_factor)}"
        )
        print("Total cost:", self.graph.total_cost())
        if not subgraphs_too:
            return
        for ii in range(self.graph.no_subgraphs()):
            s = self.graph.get_subgraph(ii)
            print("Subgraph", ii, ":")
            print("  No. empirical nodes:", s.count_empirical_nodes())
            print("  No. theoretical nodes:", s.count_theoretical_nodes())
            print("  Cost:", s.total_cost())
            print("  Matching density:", s.matching_density())
            print("  Theoretical spectra involved:", s.theoretical_spectra_involved())
