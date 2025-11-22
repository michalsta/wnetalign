import pandas as pd
import numpy as np
import warnings
from wnet.distances import DistanceMetric
from wnetalign import Spectrum
from wnetalign import WNetAligner as Solver

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



def scale_mz_values(spectrum: Spectrum, scale_factor) -> Spectrum:
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
                  max_mz_shift,
                  max_rt_shift,
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
        S1 = S1.normalized()
        S2 = S2.normalized()

    # define the distance function
    if order == 2:
        dist_fun = DistanceMetric.L2
    elif order == 1:
        dist_fun = DistanceMetric.L1
    elif order == np.inf:
        dist_fun = DistanceMetric.LINF
    else:
        raise ValueError("Unsupported order for distance metric.")


    # calculate the transport plan
    results = Solver(
        empirical_spectrum=S1,
        theoretical_spectra=[S2],
        distance=dist_fun,
        max_distance=mtd,
        trash_cost=mtd,
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

def test_align_spectra():
    from pathlib import Path

    datadir = Path(__file__).parent.parent / 'tutorials' / 'lcms' / 'data'
    filename1 = '100825O2c1_MT-AU-0044-2010-08-15_038.csv'
    filename2 = '100820O2c1_MT-AU-0044-2010-08-1_030.csv'

    sp1 = pd.read_csv(datadir / filename1)
    sp2 = pd.read_csv(datadir / filename2)

    S1 = Spectrum(np.array([sp1['m/z'].values, sp1['Retention time'].values]), np.array(sp1['Intensity'].values))
    S2 = Spectrum(np.array([sp2['m/z'].values, sp2['Retention time'].values]), np.array(sp2['Intensity'].values))

    max_mz_shift = 0.005
    max_rt_shift = 800

    S = align_spectra(S1, S2, max_mz_shift, max_rt_shift)
    assert len(S) > 25000


if __name__ == "__main__":
    test_align_spectra()


