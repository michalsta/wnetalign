from wnetalign.spectrum import Spectrum

import numpy as np
import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt

from nmr.load_spectra import load_spectrum

def simulate_7d_shifts(data_path,
                       lims_nucl, 
                       lims_aa: dict,
                       dim = 7,
                       shift_independently = True,
                       OUT_FOLDER = "./simulated_spectra",
                       seed = 0,
                       default_rng = False,
                       ):

    if OUT_FOLDER:
        if os.path.isdir(data_path): out_folder = f"{OUT_FOLDER}_{data_path.split("/")[-1]}"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    AA = sorted(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR', 'VAL'])

    if os.path.isdir(data_path): paths = sorted(glob(data_path + "/*.csv"))
    elif os.path.isfile(data_path) and data_path[-4:] == ".csv": paths = [data_path]
    else: raise Exception("Incorrect file format: data path should be directory with csv files or csv file.")

    if default_rng: rn = np.random.default_rng(seed)
    else: rn = np.random.RandomState(seed)

    # select shifts for each aa and each dimention
    if lims_aa: # sample shifts for each aa
        shifts_per_aa = np.fromiter((rn.uniform(l, h) for (l, h) in lims_aa.values()), dtype=np.dtype(float)) # shift by aa
        if shift_independently: # for each aa sample shifts for each dimension independently
            shifts_aa_x_nucl = np.fromiter((rn.uniform(l, h, dim) for (l, h) in lims_aa.values()), dtype=np.dtype((float, dim)))
        else: # shift all aa the same
            if lims_nucl: # each nucleus has different min and max values 
                shifts_per_nucl = np.fromiter((rn.uniform(l, h) for (l, h) in lims_nucl.values()), dtype=np.dtype(float)) # sample shift by nucl
                shifts_aa_x_nucl = np.fromiter((aa + shifts_per_nucl for aa in shifts_per_aa), dtype=np.dtype((float, len(shifts_per_nucl)))) # add shifts for nucl to shifts for aa - this one was used in the paper
            else:
                shifts_aa_x_nucl = np.fromiter((np.full(dim, aa) for aa in shifts_per_aa), dtype=np.dtype((float, dim))) # shift only by aa shift
    else: # sample shifts only for each nucl
        if lims_nucl:
            if shift_independently: # sample nucl shifts for each aa independently
                shifts_aa_x_nucl = np.fromiter((rn.uniform(l, h, len(AA)) for (l, h) in lims_nucl.values()), dtype=np.dtype((float, len(AA)))).T
            else: 
                shifts_aa_x_nucl = np.fromiter((np.full(len(AA), nucl) for nucl in shifts_per_nucl), dtype=np.dtype((float, len(AA)))).T # shift only by nucl 
        else: raise Exception("No limits of chemical shifts provided either for aa or nuclei")

    simulated_spectra, spectra, aa_by_spectrum = [], [], []
    for path in paths:
        df = pd.read_csv(path)
        labels = df.amino.to_numpy()

        spectrum = load_spectrum(path, dim = 7, scale_nucl={}, verbose=False)
        X = spectrum.positions.T
        X_sorted = np.concatenate([X[labels == l] for l in AA], axis=0)
        I_sorted = np.concatenate([spectrum.intensities[labels == l] for l in AA], axis=0)

        classes = np.sort(labels)
        aa_by_spectrum.append(classes)

        spectrum.positions = X_sorted.T
        spectrum.intensities == I_sorted
        spectra.append(spectrum)

        X_simulated = np.concatenate([X_sorted[classes == l] + shifts_aa_x_nucl[i] for i, l in enumerate(AA)], axis=0)

        simulated_spectrum = Spectrum(positions = X_simulated.T,
                        intensities = I_sorted,
                        label = spectrum.label + " shifted",
                        )
        simulated_spectra.append(simulated_spectrum)

        if OUT_FOLDER:
            save_7d_spectrum(simulated_spectrum, nuclei=lims_nucl.keys(), assignment=classes, OUT_FOLDER=out_folder)
    
    return simulated_spectra, spectra, aa_by_spectrum, shifts_aa_x_nucl

def save_7d_spectrum(spectrum, nuclei, assignment, OUT_FOLDER="simulated_spectra"):

    X = spectrum.positions.T
    df = pd.DataFrame(X, columns=nuclei)
    df['i'] = spectrum.intensities
    if len(nuclei) == 7: df['amino'] = assignment
    else: df['Assignment'] = assignment
    df.to_csv(f"{OUT_FOLDER}/{spectrum.label}.csv", index=False)


def plot_2d_projections_7d(data, protein, x_proj, y_proj,
                       x_lim = None, 
                       y_lim = None,
                       figsize = (10, 6),
                       cmap_name = "tab20",
                       s = 5,
                       legend = True,
                       bbox = (1.01, 0.95),
                       ncol=1,
                      ):
    

    AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(AA)))
    map_colors = dict(zip(AA, colors))
    
    df = data[protein]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for aa in AA:
        ax.scatter(df[df.amino == aa][x_proj], df[df.amino ==aa][y_proj], s=s, color=map_colors[aa], label=aa)
    
    plt.xlabel(f'{x_proj} (ppm)')
    if x_lim: plt.xlim(x_lim[0], x_lim[1])
    plt.ylabel(f'{y_proj} (ppm)')
    if y_lim: plt.ylim(y_lim[0], y_lim[1])
    plt.title(f'{protein}')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Often used for NMR spectra
    plt.gca().invert_yaxis()  
    
    if legend: ax.legend(bbox_to_anchor=bbox, ncol=ncol)
    return ax


