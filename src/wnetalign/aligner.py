from collections import namedtuple
from collections.abc import Sequence
from typing import Optional, Union
import numpy as np

from wnet.distances import DistanceMetric
from wnet.wnet_cpp import (
    NetworkSimplex,
    CostScaling,
    CycleCanceling,
    CapacityScaling,
)
from wnetalign import wnetalign_cpp
from wnetalign.spectrum import Spectrum


def _get_cpp_aligner_class(dim: int):
    return getattr(wnetalign_cpp, f"WNetAligner{dim}")


_Flow = namedtuple("Flow", ["empirical_peak_idx", "theoretical_peak_idx", "flow"])

_SOLVER_METHODS = {
    "network_simplex": NetworkSimplex,
    "cycle_canceling": CycleCanceling,
    "cost_scaling": CostScaling,
    "capacity_scaling": CapacityScaling,
}


class WNetAligner:
    """
    Aligns an empirical spectrum to one or more theoretical spectra using a Wasserstein network approach.
    Thin wrapper around the C++ WNetAligner<DIM> class.

    Parameters
    ----------
    solver : NetworkSimplex | CostScaling | CycleCanceling | CapacityScaling, optional
        Solver configuration object.  Takes precedence over ``method`` when both are given.
        Defaults to ``NetworkSimplex()`` (warm restarts, BLOCK_SEARCH pivot).
    method : str, optional
        Min-cost flow algorithm as a string: ``"network_simplex"`` (default),
        ``"cycle_canceling"``, ``"cost_scaling"``, or ``"capacity_scaling"``.
        Ignored when ``solver`` is provided.
    scale_factor : int or float, optional
        Overrides the **intensity** scale: real intensities are multiplied by
        it and truncated to integer supplies.  Leave it ``None`` (recommended)
        to let the aligner size that grid from the faintest peak present.

        This used to be a single tied factor that was *also* multiplied into
        the positions to buy distance resolution.  It no longer touches the
        positions at all — the network quantizes real distances onto its own
        cost grid, sized against whatever int64 budget the intensity scale
        leaves.  Code that passed a ``scale_factor`` to sharpen distances
        should simply stop passing one.

    Attributes
    ----------
    intensity_scale : float
        The intensity grid actually used; flows are denominated in these units.
    cost_scale : int
        The distance/cost grid the network chose.
    scale_factor : float
        Back-compatible alias for ``intensity_scale``.
    """

    def __init__(
        self,
        empirical_spectrum: Spectrum,
        theoretical_spectra: Sequence[Spectrum],
        distance: DistanceMetric,
        max_distance: Union[int, float],
        trash_cost: Optional[Union[int, float]] = None,
        scale_factor: Optional[Union[int, float]] = None,
        experimental_trash_cost: Optional[Union[int, float]] = None,
        theoretical_trash_cost: Optional[Union[int, float]] = None,
        method: Optional[str] = None,
        solver=None,
    ) -> None:
        if trash_cost is None and experimental_trash_cost is None and theoretical_trash_cost is None:
            raise ValueError("At least one of trash_cost, experimental_trash_cost, or theoretical_trash_cost must be provided.")

        if solver is None and method is None:
            solver = NetworkSimplex()
        elif solver is None:
            if method not in _SOLVER_METHODS:
                raise ValueError(f"Unknown method {method!r}. Choose from: {list(_SOLVER_METHODS)}")
            solver = _SOLVER_METHODS[method]()

        if not hasattr(empirical_spectrum, "vecdist"):
            raise TypeError("empirical_spectrum must be a Distribution/Spectrum")
        if not all(hasattr(t, "vecdist") for t in theoretical_spectra):
            raise TypeError("all theoretical spectra must be Distributions/Spectra")

        # Pass wnet's C++ distribution object (CVectorDistributionFloat{DIM});
        # the aligner scales positions and quantizes intensities internally.
        cpp_cls = _get_cpp_aligner_class(empirical_spectrum.positions.shape[0])
        self._cpp = cpp_cls(
            empirical_spectrum.vecdist,
            [t.vecdist for t in theoretical_spectra],
            distance.value,
            float(max_distance),
            float(trash_cost) if trash_cost is not None else -1.0,
            float(scale_factor) if scale_factor is not None else 0.0,
            float(experimental_trash_cost) if experimental_trash_cost is not None else -1.0,
            float(theoretical_trash_cost)  if theoretical_trash_cost  is not None else -1.0,
            solver,
        )
        #: Factor real intensities are multiplied by before truncation to
        #: integer supplies.  Flows come back in these units.
        self.intensity_scale = self._cpp.intensity_scale()
        #: Integer grid the network chose for the real distances.  Independent
        #: of the intensity scale: they share an int64 budget but not a value.
        self.cost_scale = self._cpp.cost_scale()
        #: Back-compatible alias for :attr:`intensity_scale`.  It used to be a
        #: single factor applied to positions *and* intensities at once; the
        #: two grids are chosen separately now, and this name follows the one
        #: flows are denominated in.
        self.scale_factor = self.intensity_scale
        self.point = None

    def set_point(self, point: Union[Sequence[float], np.ndarray]) -> None:
        """
        Set proportions of theoretical spectra and solve the graph at the given point.
        """
        self.point = point
        self._cpp.set_point(list(point))

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

    def flows(self) -> list[_Flow]:
        """
        Returns a list of Flow namedtuples for each theoretical spectrum.
        """
        result = []
        for i in range(self._cpp.no_theoretical_spectra()):
            empirical_peak_idx, theoretical_peak_idx, flow = self._cpp.flows_for_target(
                i
            )
            result.append(
                _Flow(
                    empirical_peak_idx,
                    theoretical_peak_idx,
                    flow / self.intensity_scale,
                )
            )
        return result

    def consensus(self, target_id: int = 0):
        """
        Returns (empirical_ids, theoretical_ids) of greedy 1-to-1 consensus pairs
        for the given target spectrum, selected by descending flow.
        """
        return self._cpp.consensus_for_target(target_id)

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
            "Intensity scale:",
            self.intensity_scale,
            f" log10: {np.log10(self.intensity_scale)}",
        )
        print("Cost scale:", self.cost_scale)
        print("Total cost:", self._cpp.total_cost())
