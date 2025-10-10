import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

def postprocess_simulated_pair(s1, l1, s2, l2, r, classes, scale_nucl={'15N':10}, nuclei= ['15N', '1H'], find_max_flow=True):

    df = pd.DataFrame()
    df["empirical_peak_idx"] = r.empirical_peak_idx
    df["theoretical_peak_idx"] = r.theoretical_peak_idx
    df["flow"] = r.flow
    
    # empirical spectrum
    for nucl, pos in zip(nuclei, s1.positions[:, r.empirical_peak_idx]):
        if nucl in scale_nucl:df[f"{l1}_{nucl}"] = pos*scale_nucl[nucl]
        else: df[f"{l1}_{nucl}"] = pos
    df[f"{l1}_assignment"] = classes[r.empirical_peak_idx]
    
    # theorethical spectrum
    for nucl, pos in zip(nuclei, s2.positions[:, r.theoretical_peak_idx]):
        if nucl in scale_nucl:df[f"{l2}_{nucl}"] = pos*scale_nucl[nucl]
        else: df[f"{l2}_{nucl}"] = pos
    df[f"{l2}_assignment"] = classes[r.theoretical_peak_idx]

    # find the max flow 
    if find_max_flow == True: 
        df = df.loc[df.groupby('empirical_peak_idx')['flow'].idxmax()].reset_index(drop=True)
    return df