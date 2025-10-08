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


# def postprocess_simulated_pair_2d(s1, l1, s2, l2, r, clusters, scale_N=10, find_max_flow=True):

#     df = pd.DataFrame()
#     df["empirical_peak_idx"] = r.empirical_peak_idx
#     df["theoretical_peak_idx"] = r.theoretical_peak_idx
#     df["flow"] = r.flow

#     # empirical spectrum
#     n, h = s1.positions[:, r.empirical_peak_idx]
#     df[f"{l1}_15N"] = (scale_N*n).round(3).tolist()
#     df[f"{l1}_1H"]  = h.tolist()
#     df[f"{l1}_assignment"] = clusters[r.empirical_peak_idx]
    
#     # theorethical spectrum
#     n, h = s2.positions[:, r.theoretical_peak_idx]
#     df[f"{l2}_15N"] = (scale_N*n).round(3).tolist()
#     df[f"{l2}_1H"]  = h.tolist()
#     df[f"{l2}_assignment"] = clusters[r.theoretical_peak_idx]

#     # find the max flow 
#     if find_max_flow == True: 
#         df = df.loc[df.groupby('empirical_peak_idx')['flow'].idxmax()].reset_index(drop=True)
#     return df

# def postprocess_simulated_pair_4d(s1, l1, s2, l2, r, clusters, scale_C=10, nuclei= ['C1', 'C2', 'H1', 'H2'], find_max_flow=True):

#     df = pd.DataFrame()
#     df["empirical_peak_idx"] = r.empirical_peak_idx
#     df["theoretical_peak_idx"] = r.theoretical_peak_idx
#     df["flow"] = r.flow
    
#     # empirical spectrum
#     for nucl, pos in zip(nuclei, s1.positions[:, r.empirical_peak_idx]):
#         if nucl.startswith("C"): df[f"{l1}_{nucl}"] = pos*scale_C
#         else: df[f"{l1}_{nucl}"] = pos
#     df[f"{l1}_assignment"] = clusters[r.empirical_peak_idx]#.to_list()
    
#     # theorethical spectrum
#     for nucl, pos in zip(nuclei, s2.positions[:, r.theoretical_peak_idx]):
#         if nucl.startswith("C"): df[f"{l2}_{nucl}"] = pos*scale_C
#         else: df[f"{l2}_{nucl}"] = pos
#     df[f"{l2}_assignment"] = clusters[r.theoretical_peak_idx]#.to_list()

#     # find the max flow 
#     if find_max_flow == True: 
#         df = df.loc[df.groupby('empirical_peak_idx')['flow'].idxmax()].reset_index(drop=True)
#     return df

# def postprocess_simulated_pair_7d(s1, l1, s2, l2, r, aa, nuclei= ['HN', 'N', 'CO', 'CA', 'CB', 'HA', 'HB'], find_max_flow=True):

#     df = pd.DataFrame()
#     df["empirical_peak_idx"] = r.empirical_peak_idx
#     df["theoretical_peak_idx"] = r.theoretical_peak_idx
#     df["flow"] = r.flow
    
#     # empirical spectrum
#     for nucl, pos in zip(nuclei, s1.positions[:, r.empirical_peak_idx]):
#         df[f"{l1}_{nucl}"] = pos
#     df[f"{l1}_assignment"] = aa[r.empirical_peak_idx].to_list()
    
#     # theorethical spectrum
#     for nucl, pos in zip(nuclei, s2.positions[:, r.theoretical_peak_idx]):
#         df[f"{l2}_{nucl}"] = pos
#     df[f"{l2}_assignment"] = aa[r.theoretical_peak_idx].to_list()

#     # find the max flow 
#     if find_max_flow == True: 
#         df = df.loc[df.groupby('empirical_peak_idx')['flow'].idxmax()].reset_index(drop=True)
#     return df