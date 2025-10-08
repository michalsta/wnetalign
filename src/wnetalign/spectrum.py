import numpy as np

from wnet import Distribution as Spectrum
from wnet import Distribution_1D as Spectrum_1D


def parse_featurexml_to_spectrum(path):
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


Spectrum.FromFeatureXML = staticmethod(parse_featurexml_to_spectrum)
