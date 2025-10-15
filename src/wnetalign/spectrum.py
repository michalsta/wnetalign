from functools import cached_property

import numpy as np

from wnet import Distribution


class Spectrum(Distribution):
    """
    A class representing NMR or MS spectrum data.
    """
    def __init__(self, coords: np.ndarray, intensities: np.ndarray, label: str | None = None):
        """
        Initialize a Spectrum object. Compared to Distribution, this class
        retains the original intensities (not converted to int) for more precise
        scaling operations. They are stored in the `original_intensities` attribute.
        They are still converted to int before running any alignment algorithms.

        Parameters
        ----------
        coords : np.ndarray
            The spatial coordinates of the spectrum (e.g., m/z and RT for MS).
        intensities : np.ndarray
            The intensity values corresponding to the spatial coordinates.
        """
        self.original_intensities = intensities
        super().__init__(coords, intensities, label=label)

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

    @cached_property
    def sum_intensities(self) -> float:
        """
        Return the sum of the original intensities.
        """
        return np.sum(self.original_intensities)

    def scaled(self, factor: float) -> 'Spectrum':
        """
        Return a new Spectrum object with intensities scaled by the given factor.

        Parameters
        ----------
        factor : float
            The scaling factor to apply to the intensities.

        Returns
        -------
        Spectrum
            A new Spectrum object with scaled intensities.
        """
        return Spectrum(self.coords, self.original_intensities * factor, label=self.label)
    
    def normalized(self) -> 'Spectrum':
        """
        Return a new Spectrum object with intensities normalized to sum to 1.

        Returns
        -------
        Spectrum
            A new Spectrum object with normalized intensities.
        """
        total = self.sum_intensities
        if total == 0:
            raise ValueError("Cannot normalize a spectrum with total intensity of 0.")
        return Spectrum(self.coords, self.original_intensities / total, label=self.label)
    
def Spectrum1D(coords: np.ndarray, intensities: np.ndarray, label: str | None = None) -> Spectrum:
    """
    Create a 1D Spectrum object.

    Parameters
    ----------
    coords : np.ndarray
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
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if not isinstance(intensities, np.ndarray):
        intensities = np.array(intensities)
    assert len(coords.shape) == 1
    assert len(intensities.shape) == 1
    assert coords.shape[0] == intensities.shape[0]
    return Spectrum(coords[np.newaxis, :], intensities)