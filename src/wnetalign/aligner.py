from collections import namedtuple
from collections.abc import Sequence
from typing import Optional, Union
import numpy as np

from wnet import Distribution
from wnet.distances import DistanceMetric
from wnetalign import wnetalign_cpp
from wnetalign.spectrum import Spectrum


def _get_cpp_aligner_class(dim: int):
    return getattr(wnetalign_cpp, f"WNetAligner{dim}")


class WNetAligner:
    """
    Aligns an empirical spectrum to one or more theoretical spectra using a Wasserstein network approach.
    Thin wrapper around the C++ WNetAligner<DIM> class.
    """

    def __init__(
        self,
        empirical_spectrum: Spectrum,
        theoretical_spectra: Sequence[Spectrum],
        distance: DistanceMetric,
        max_distance: Union[int, float],
        trash_cost: Union[int, float],
        scale_factor: Optional[Union[int, float]] = None,
    ) -> None:
        # Ensure all spectra have their C++ backing objects
        assert hasattr(
            empirical_spectrum, "_cpp"
        ), "empirical_spectrum must be a Spectrum with a C++ backing object"
        assert all(
            hasattr(t, "_cpp") for t in theoretical_spectra
        ), "all theoretical spectra must have C++ backing objects"

        cpp_cls = _get_cpp_aligner_class(empirical_spectrum.positions.shape[0])
        self._cpp = cpp_cls(
            empirical_spectrum._cpp,
            [t._cpp for t in theoretical_spectra],
            distance.value,
            float(max_distance),
            float(trash_cost),
            float(scale_factor) if scale_factor is not None else 0.0,
        )
        self.scale_factor = self._cpp.scale_factor()
        self.point = None

    def set_point(self, point: Union[Sequence[float], np.ndarray]) -> None:
        """
        Set proportions of theoretical spectra and solve the graph at the given point.
        """
        self.point = point
        self._cpp.set_point(
            list(point) if isinstance(point, np.ndarray) else list(point)
        )

    def total_cost(self) -> float:
        """
        Calculates the total cost of the alignment, rescaled to original units.
        """
        return self._cpp.total_cost()

    def print(self) -> None:
        """
        Prints a string representation of the underlying graph.
        """
        print(str(self._cpp))

    def flows(self) -> list[namedtuple]:
        """
        Returns a list of Flow namedtuples for each theoretical spectrum.
        """
        result = []
        for i in range(self._cpp.no_theoretical_spectra()):
            empirical_peak_idx, theoretical_peak_idx, flow = self._cpp.flows_for_target(
                i
            )
            result.append(
                namedtuple(
                    "Flow", ["empirical_peak_idx", "theoretical_peak_idx", "flow"]
                )(empirical_peak_idx, theoretical_peak_idx, flow / self.scale_factor)
            )
        return result

    def no_subgraphs(self) -> int:
        """
        Returns the number of subgraphs in the alignment network.
        """
        return self._cpp.no_subgraphs()

    def print_diagnostics(self, subgraphs_too=False):
        """
        Prints diagnostic information about the alignment.
        """
        print("Diagnostics:")
        print("No subgraphs:", self._cpp.no_subgraphs())
        print("No empirical nodes:", self._cpp.count_empirical_nodes())
        print("No theoretical nodes:", self._cpp.count_theoretical_nodes())
        print("Matching density:", self._cpp.matching_density())
        print(
            "Scale factor:", self.scale_factor, f" log10: {np.log10(self.scale_factor)}"
        )
        print("Total cost:", self._cpp.total_cost())
