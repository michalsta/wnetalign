import numpy as np
import pandas as pd
from glob import glob

from wnetalign.spectrum import Spectrum

def read_label(path, dim=2):

    if dim == 2: return path.split("/")[-1].split(".")[0].split("_")[-1]
    elif dim == 4: return path.split("/")[-1].split(".")[0].split("_")[0]
    else: return path.split("/")[-1].split(".")[0]

def load_spectrum(path, 
                  dim=2, 
                  scale_nucl={}, #{'15N':10} - 2D, {'C':10} - 4D
                  max_peak_fraction=None, #0.1
                  intensity_threshold=0, #0.01
                  verbose=False,
                  ):
    label = read_label(path, dim=dim)
    df = pd.read_csv(path)
    if 'i' not in df.columns:
        if verbose: print("{}\nnumber of peaks: {}, no signal intensities available - setting 'i' to 1".format(label, df.shape[0]))
        verbose=False
    if verbose:
        s = df.i.sum()
        print("{}\nnumber of peaks: {}, total signal: {}".format(label, df.shape[0], round(s, 2)))
    if max_peak_fraction: 
        df = df[df['i'] > df.i.max()*max_peak_fraction]
        if verbose: 
            print("Peaks with intensities higher than max_intensity * {}".format(max_peak_fraction))
            print("number of peaks: {}, max_intensity * max_peak_fraction: {}, % of signal left: {}".format(df.shape[0], round(df.i.max()*max_peak_fraction, 3), round(100*df.i.sum()/s, 2)))
    elif intensity_threshold:
        df = df[df['i'] > intensity_threshold]
        if verbose: 
            print("Peaks with intensities > {}".format(intensity_threshold))
            print("number of peaks: {}, intensity_threshold: {}, % of signal left: {}".format(df.shape[0], intensity_threshold, round(100*df.i.sum()/s, 2)))
    if verbose: print()

    for nucl, scale in scale_nucl.items():
        for c in df.columns:
            if c.startswith(nucl): df[c] = df[c]/scale
    positions = df[df.columns[:dim]].T.to_numpy()

    if 'i' in df.columns: intensities = df['i'].to_numpy()
    else: intensities = np.ones(positions.shape[1])

    return Spectrum(positions = positions,
                        intensities = intensities,
                        label = label,
                        )

def load_spectra(data_path, **kwargs):

    return [load_spectrum(path, **kwargs) for path in sorted(glob(data_path + "/*.csv"))]
