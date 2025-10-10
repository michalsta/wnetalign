import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

from wnetalign.spectrum import Spectrum
from nmr.load_spectra import load_spectra, load_spectrum

def simulate_2d_shifts(data_path,
                       k=54,
                       scale_nucl={"15N": 10},
                       nuclei = ["15N", "1H"],
                       max_peak_fraction=0.1,
                       h_lim = (0.01, 0.025),
                       n_lim = (0.0, 0.015),
                       h_t_lim = (0, 0.01),
                       n_t_lim = (0, 0.005),
                       OUT_FOLDER = "./temps/15N_HSQC_GB1_simulated",
                       seed = 0,
                       default_rng = False,
                       ):

    spectra = load_spectra(data_path, 
                           dim = 2,
                           scale_nucl = scale_nucl,
                           max_peak_fraction = max_peak_fraction,
                          )
    if default_rng: rn = np.random.default_rng(seed)
    else: rn = np.random.RandomState(seed)
    
    # each cluster shifted by the same values along H and N axis
    h_x = rn.uniform(h_lim[0], h_lim[1], k)
    n_x = rn.uniform(n_lim[0], n_lim[1], k)
    h_x_t = rn.uniform(h_t_lim[0], h_t_lim[1], len(spectra)-1)
    n_x_t = rn.uniform(n_t_lim[0], n_t_lim[1], len(spectra)-1)
    
    simulated_spectra = []
    for i in range(len(spectra)):
        if i == 0:
            X_scaled = np.array([spectra[i].positions[0, :], spectra[i].positions[1, :]]).T
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_scaled)
            X = np.concatenate([X_scaled[kmeans.labels_ == l] for l in range(k)], axis=0)
            I = np.concatenate([spectra[i].intensities[kmeans.labels_ == l] for l in range(k)], axis=0)
            clusters = np.sort(kmeans.labels_)
            simulated_spectra.append(
                Spectrum(
                    positions = X.T,
                    intensities = I,
                    label = "simulated " + spectra[i].label,
                )
            )
        else:
            X = np.concatenate([X[clusters == l] + np.array([n_x[l], h_x[l]]) + np.array([n_x_t[i-1], h_x_t[i-1]]) for l in range(k)], axis=0)
            simulated_spectra.append(
                Spectrum(
                    positions = X.T,
                    intensities = I,
                    label = "simulated " + spectra[i].label,
                )
            )
    
    if OUT_FOLDER:
        if not os.path.exists(OUT_FOLDER):
            os.makedirs(OUT_FOLDER)
        for s in simulated_spectra:
            save_spectrum(s, clusters, nuclei, scale_nucl, OUT_FOLDER)
    
    return simulated_spectra, clusters, (h_x, n_x, h_x_t, n_x_t)


def simulate_4d_shifts(file_path,
                       clusters_path,
                       lims,
                       scale_nucl={"C": 10},
                       dim = 4,
                       OUT_FOLDER = "4D/2LX7/2LX7_0.1",
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

    spectrum = Spectrum(
        positions = X_sorted.T,
        intensities = I_sorted,
        label = spectrum.label
    )
    
    X_simulated = np.concatenate([X[labels == l] + x[l] for l in range(k)], axis=0)

    simulated_spectrum = Spectrum(
        positions = X_simulated.T,
        intensities = I_sorted,
        label = spectrum.label + "_simulated",
    )
    
    if OUT_FOLDER:
        save_spectrum(spectrum, clusters, lims.keys(), scale_nucl, f"{OUT_FOLDER}_sorted")
        save_spectrum(simulated_spectrum, clusters, lims.keys(), scale_nucl, f"{OUT_FOLDER}_simulated")
    
    return simulated_spectrum, spectrum, clusters, x

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

        spectrum = Spectrum(
            positions = X_sorted.T,
            intensities = I_sorted,
            label = spectrum.label
        )
        spectra.append(spectrum)

        X_simulated = np.concatenate([X_sorted[classes == l] + shifts_aa_x_nucl[i] for i, l in enumerate(AA)], axis=0)

        simulated_spectrum = Spectrum(
            positions = X_simulated.T,
            intensities = I_sorted,
            label = spectrum.label + " shifted",
        )
        simulated_spectra.append(simulated_spectrum)

        if OUT_FOLDER:
            save_spectrum(simulated_spectrum, assignment=classes, nuclei=lims_nucl.keys(), OUT_FOLDER=out_folder)
    
    return simulated_spectra, spectra, aa_by_spectrum, shifts_aa_x_nucl

def save_spectrum(spectrum, assignments, nuclei, scale_nucl, OUT_FOLDER="simulated_spectra"):

    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    df = pd.DataFrame(columns=nuclei)
    for i, nucl in enumerate(nuclei):
        if nucl in scale_nucl: df[nucl] = spectrum.positions[i, :]*scale_nucl[nucl]
        else: df[nucl] = spectrum.positions[i, :]
    df['i'] = spectrum.intensities
    if len(nuclei) == 7: df['amino'] = assignments
    else: df['Assignment'] = assignments
    # df = df.astype({"Assignment":'int64'})
    if len(nuclei) == 2: df.to_csv(f"{OUT_FOLDER}/{"_".join(spectrum.label.split())}.csv", index=False)
    else: df.to_csv(f"{OUT_FOLDER}/{spectrum.label}.csv", index=False)