from typing import Optional

import numpy as np

from wnet import Distribution
from wnetalign import wnetalign_cpp


def _get_cpp_spectrum_class(dim: int):
    return getattr(wnetalign_cpp, f"Spectrum{dim}")


class Spectrum(Distribution):
    """
    A class representing NMR or MS spectrum data.
    Thin wrapper around the C++ Spectrum<DIM> class.

    The scaling/normalization helpers (``scaled``,
    ``positions_intensities_scaled``, ``normalized``, ``as_distribution``,
    ``sum_intensities``) are inherited from Distribution, whose polymorphic
    constructor returns a Spectrum here (rebuilding the C++ backing object via
    ``__init__``).
    """

    def __init__(
        self,
        positions: np.ndarray,
        intensities: np.ndarray,
        label: Optional[str] = None,
    ):
        """
        Initialize a Spectrum object.  Retains the original (non-int)
        intensities in ``original_intensities``; they are converted to int
        before running any alignment algorithms.

        Parameters
        ----------
        positions : np.ndarray
            The spatial coordinates of the spectrum (e.g., m/z and RT for MS).
            Shape: (dim, num_points)
        intensities : np.ndarray
            The intensity values corresponding to the spatial coordinates.
        """
        super().__init__(positions, intensities, label=label)
        dimension = positions.shape[0]
        cpp_cls = _get_cpp_spectrum_class(dimension)
        self._cpp = cpp_cls(
            positions.astype(np.float64),
            intensities.astype(np.float64),
        )
        self.original_intensities = intensities

    @staticmethod
    def FromFeatureXML(path):
        """
        Parse a featureXML file and return a Spectrum object.
        """
        import pyopenms as oms

        # load the featureXML file
        featureXML = oms.FeatureXMLFile()
        features = oms.FeatureMap()
        featureXML.load(path, features)
        # load m/z, rt, and intensity values from the features
        mzs = []
        rts = []
        intensities = []
        for feature in features:
            mzs.append(feature.getMZ())
            rts.append(feature.getRT())
            intensities.append(feature.getIntensity())
        # create a Spectrum object
        spectrum = Spectrum(np.array([mzs, rts]), np.array(intensities))
        return spectrum

    def to_vector_distribution(self):
        """
        Return the C++ CVectorDistribution (int64 intensities) directly
        from the underlying C++ Spectrum.
        """
        return self._cpp.to_vector_distribution()


def Spectrum_1D(
    positions: np.ndarray, intensities: np.ndarray, label: Optional[str] = None
) -> Spectrum:
    """
    Create a 1D Spectrum object.

    Parameters
    ----------
    positions : np.ndarray
        The spatial coordinates of the spectrum (e.g., m/z for MS).
    intensities : np.ndarray
        The intensity values corresponding to the spatial coordinates.
    label : str, optional
        An optional label for the spectrum.

    Returns
    -------
    Spectrum
        A 1D Spectrum object.
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    if not isinstance(intensities, np.ndarray):
        intensities = np.array(intensities)
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got shape {positions.shape}")
    if intensities.ndim != 1:
        raise ValueError(f"intensities must be 1D, got shape {intensities.shape}")
    if positions.shape[0] != intensities.shape[0]:
        raise ValueError(
            f"positions and intensities must have the same length, got {positions.shape[0]} and {intensities.shape[0]}"
        )
    return Spectrum(positions[np.newaxis, :], intensities, label=label)
