import pyopenms as oms
import pandas as pd
import numpy as np
import warnings
from wnetalign import Spectrum
from wnetalign import WNetAligner as Solver

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



def parse_featurexml_to_spectrum(path):
    """
    Parse a featureXML file and return a Spectrum object.
    """
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


def scale_mz_values(spectrum: Spectrum, scale_factor: int | float) -> Spectrum:
    """
    Scale the m/z values of a Spectrum object by a given factor.
    """
    scaled_positions = spectrum.positions.copy()
    scaled_positions[0, :] = scaled_positions[0, :] * scale_factor
    return Spectrum(scaled_positions, spectrum.intensities)


def spectrum_to_dataframe(spectrum: Spectrum) -> pd.DataFrame:
    """
    Convert a Spectrum object to a DataFrame.
    """
    return pd.DataFrame(data={'m/z': spectrum.positions[0, :], 
                            'Retention time': spectrum.positions[1, :], 
                            'Intensity': spectrum.intensities})


def align_spectra(S1: Spectrum, 
                  S2: Spectrum, 
                  max_mz_shift: int | float, 
                  max_rt_shift: int | float, 
                  order= np.inf, 
                  normalize: bool = True, 
                  find_consensus= True) -> pd.DataFrame:
    """
    Align two spectra using the Wasserstein distance.

    Parameters
    ----------
    S1 : Spectrum
        The first spectrum to align.
    S2 : Spectrum
        The second spectrum to align.
    max_mz_shift : int | float
        The maximum allowed m/z shift.
    max_rt_shift : int | float
        The maximum allowed retention time shift.
    order : int, optional
        The order of the norm to use for the distance calculation. Default is np.inf.
    normalize : bool, optional
        Whether to normalize the intensity values of the spectra. Default is True.
    find_consensus : bool, optional
        Whether to find consensus features after alignment. Default is True.
    """
    # calculate the scale factor
    scale_mz = max_rt_shift / max_mz_shift
    mtd = round(max_rt_shift)

    # create copies of the spectra
    sp1 = spectrum_to_dataframe(S1)
    sp2 = spectrum_to_dataframe(S2)

    # create Spectrum objects
    S1 = scale_mz_values(S1, scale_mz)
    S2 = scale_mz_values(S2, scale_mz)

    # normalize the intensity values
    if normalize:
        S1 = S1.scaled(1000000/sum(S1.intensities))
        S2 = S2.scaled(1000000/sum(S2.intensities))
    # define the distance function
    dist_fun = lambda x, y: np.linalg.norm(x - y, axis=0, ord=order)

    # calculate the transport plan
    results = Solver(
        empirical_spectrum=S1,
        theoretical_spectra=[S2],
        distance_function=dist_fun,
        max_distance=mtd,
        trash_cost=mtd,
        scale_factor=10000,
    )
    results.set_point([1])
    results = results.flows()[0]
    # retrieve the aligned features for evaluation
    if find_consensus:
        # create a DataFrame with the transport plan
        tp = pd.DataFrame(data={'id1': results.empirical_peak_idx, 
                                'id2': results.theoretical_peak_idx, 
                                'transport': results.flow}).sort_values(by='transport', ascending=False)
        # find consensus features (by maximum transport flow)
        ids1 = set()
        ids2 = set()
        ids1_list = []
        ids2_list = []
        for _, row in tp.iterrows():
            if row['id1'] not in ids1 and row['id2'] not in ids2:
                ids1_list.append(row['id1'])
                ids2_list.append(row['id2'])
                ids1.add(row['id1'])
                ids2.add(row['id2'])
        
        sp1_aligned = sp1.iloc[ids1_list].reset_index(drop=True)
        sp2_aligned = sp2.iloc[ids2_list].reset_index(drop=True)
        # create the transport plan
        transport_plan = sp1_aligned.join(sp2_aligned, 
                                        lsuffix='_S1', 
                                        rsuffix='_S2')

        return transport_plan
    else:
        return results
