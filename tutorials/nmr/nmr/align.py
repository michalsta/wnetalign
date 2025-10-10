import numpy as np
import pandas as pd
import os
from collections import namedtuple
from itertools import combinations, permutations

from wnetalign.aligner import WNetAligner
from nmr.load_spectra import load_spectra

def align_pair(S1,
               S2,
               max_distance=0.05,
               trash_cost=0.09,
               intensity_scaling=None,
               normalize = True,
               verbose=False,
               ):
    
    if verbose: print(f"Aligning {S1.label} and {S2.label}")
    if normalize: 
        S1 = S1.normalized()
        S2 = S2.normalized()
    solver = WNetAligner(
            empirical_spectrum = S1,
            theoretical_spectra = [S2], 
            distance_function = lambda x, y: np.linalg.norm(x - y, axis=0),
            max_distance = max_distance,
            trash_cost = trash_cost,
            scale_factor = intensity_scaling,
            )
    solver.set_point([1])
    if verbose: print(f"Total cost: {solver.total_cost()}\n")
  
    res = solver.flows()[0]

    return solver, namedtuple(
        "Alignment", ["name", "max_distance", "trash_cost", "cost", "empirical_peak_idx", "theoretical_peak_idx", "flow"]
        )(f"{S1.label}_vs_{S2.label}", max_distance, trash_cost, solver.total_cost(), res.empirical_peak_idx, res.theoretical_peak_idx, res.flow)

def prepare_label(S, dim=2):

    if dim==2:
        return S.label.split(" ")[-1]
    elif dim==7:
        return S.label.split("_")[0]
    else: return S.label

def save_alignment(res, S1, S2, scale_nucl={'15N':10}, nuclei=['15N', '1H'], OUT_FOLDER="Alignment_results"):

    l1 = prepare_label(S1, S1.positions.shape[0])
    l2 = prepare_label(S2, S2.positions.shape[0])

    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    df = pd.DataFrame()
    df["empirical_peak_idx"] = res.empirical_peak_idx
    df["theoretical_peak_idx"] = res.theoretical_peak_idx
    df["flow"] = res.flow

    # empirical spectrum
    for nucl, pos in zip(nuclei, S1.positions[:, res.empirical_peak_idx]):
        if nucl in scale_nucl: df[f"{l1}_{nucl}"] = pos*scale_nucl[nucl]
        else: df[f"{l1}_{nucl}"] = pos
    # theorethical spectrum
    for nucl, pos in zip(nuclei, S2.positions[:, res.theoretical_peak_idx]):
        if nucl in scale_nucl: df[f"{l2}_{nucl}"] = pos*scale_nucl[nucl]
        else: df[f"{l2}_{nucl}"] = pos

    df.to_csv(f"./{OUT_FOLDER}/{l1}_vs_{l2}_{res.max_distance}_{res.trash_cost}_{res.cost}.csv", index=False)

def align_all(DATA_PATH = "./15N_HSQC_GB1_reduced",  
              dim=2,
              scale_nucl={'15N':10}, 
              nuclei = ['15N', '1H'],
              max_peak_fraction=0.1,
              intensity_threshold=0,
              intensity_scaling=None, 
              max_distance=0.05, 
              trash_cost=0.09, 
              normalize = True,
              permute=False,
              temp_series=True,
              OUT_FOLDER=None,
              ):

    spectra = load_spectra(DATA_PATH, dim=dim, scale_nucl=scale_nucl, max_peak_fraction=max_peak_fraction, intensity_threshold=intensity_threshold)

    if permute: spectra = permutations(spectra, 2)
    else: spectra = combinations(spectra, 2)

    if temp_series: spectra = zip(spectra[:-1], spectra[1:])

    solvers, results = [], []

    for s1, s2 in spectra:
        if normalize: s1, s2 = s1.normalized(), s2.normalized()
        s, r = align_pair(s1, s2, 
                          intensity_scaling = intensity_scaling,
                          max_distance = max_distance,
                          trash_cost = trash_cost,
                          normalize = normalize,
                          )
        solvers.append(s)
        results.append(r)
    
        if OUT_FOLDER: save_alignment(r, s1, s2, scale_nucl=scale_nucl, nuclei=nuclei, OUT_FOLDER=OUT_FOLDER)

    return solvers, results
