from wnetalign.spectrum import Spectrum

from nmr.load_spectra import load_spectrum, load_spectra

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist 

def simulate_4d_shifts(file_path,
                       clusters_path,
                       lims,
                       scale_nucl={"C": 10},
                       dim = 4,
                       OUT_FOLDER = "4D/2LX7/2LX7_0.1_simulated",
                       seed = 0,
                       default_rng = False,
                       ):

    spectrum = load_spectrum(file_path, dim = 4, scale_nucl=scale_nucl)
    X = spectrum.positions.T

    df_c = pd.read_csv(clusters_path)
    for nucl, scale in scale_nucl.items():
        for c in df_c.columns:
            if c.startswith(nucl): df_c[c] = df_c[c]/scale
    C = np.array([df_c[c] for c in df_c.columns[:dim]]).T
    k = C.shape[0]
    print("number of clusters:", k)

    # each cluster shifted by the same values along H and N axis
    if default_rng == True: rn = np.random.default_rng(seed)
    else: rn = np.random.RandomState(seed)

    x = np.fromiter((rn.uniform(l, h, k) for (l, h) in lims.values()), dtype=np.dtype((float, k))).T

    distances = cdist(X, C)  # Compute distances (n_samples x n_clusters)
    labels = np.argmin(distances, axis=1)  # Assign to closest centroid

    X_sorted = np.concatenate([X[labels == l] for l in range(k)], axis=0)
    I_sorted = np.concatenate([spectrum.intensities[labels == l] for l in range(k)], axis=0)
    clusters = np.sort(labels)

    spectrum.positions = X_sorted.T
    spectrum.intensities == I_sorted
    
    X_simulated = np.concatenate([X[labels == l] + x[l] for l in range(k)], axis=0)

    simulated_spectrum = Spectrum(
        positions = X_simulated.T,
        intensities = I_sorted,
        label = spectrum.label + "_simulated",
    )
    
    if OUT_FOLDER:
        if not os.path.exists(OUT_FOLDER):
            os.makedirs(OUT_FOLDER)
        # save original sorted spectrum
        for s in [spectrum, simulated_spectrum]:
            data = np.array(
                [s.positions[0, :]*scale_nucl['C'], 
                 s.positions[1, :]*scale_nucl['C'], 
                 s.positions[2, :], 
                 s.positions[3, :], 
                 s.intensities, 
                 clusters]
            )
            df = pd.DataFrame(data.T, columns=["C1", "C2", "H1", "H2", "i", "Assignment"])
            df = df.astype({"Assignment":'int64'})
            df.to_csv(f"{OUT_FOLDER}/{s.label}.csv", index=False)
    
    return simulated_spectrum, spectrum, clusters, x

def save_4d_spectrum(spectrum, nuclei, assignment, OUT_FOLDER="simulated_spectra"):

    X = spectrum.positions.T
    df = pd.DataFrame(X, columns=nuclei)
    df['i'] = spectrum.intensities
    if len(nuclei) == 7: df['amino'] = assignment
    else: df['Assignment'] = assignment
    df.to_csv(f"{OUT_FOLDER}/{spectrum.label}.csv", index=False)

def plot_2d_projections_4d(file_path,
                            x_proj,
                            y_proj,
                            x_lim=None,
                            y_lim=None,
                            figsize=(10, 6),
                            s=3,
                            legend=False,
                            bbox = (1.01, 0.95),
                            ncol=1,
                          ):

    label = file_path.split("/")[-1].split(".")[0]
    df = pd.read_csv(file_path)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.scatter(df[x_proj], df[y_proj], s=s)
        
    plt.xlabel(f'{x_proj} (ppm)')
    if x_lim: plt.xlim(x_lim[0], x_lim[1])
    plt.ylabel(f'{y_proj} (ppm)')
    if y_lim: plt.ylim(y_lim[0], y_lim[1])
    plt.title(f'{label}')
    plt.gca().invert_xaxis()  # Often used for NMR spectra
    plt.gca().invert_yaxis()  
        
    if legend: ax.legend(bbox_to_anchor=bbox, ncol=ncol)
    return ax