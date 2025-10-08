from wnetalign.spectrum import Spectrum

from nmr.load_spectra import load_spectrum, load_spectra

import numpy as np
import pandas as pd
import os
from glob import glob

from sklearn.cluster import KMeans

def simulate_2d_shifts(data_path,
                       k=54,
                       scale_nucl={"N": 10},
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
            data = np.array([s.positions[0, :]*scale_nucl['N'], s.positions[1, :], s.intensities, clusters])
            df = pd.DataFrame(data.T, columns=["15N", "1H", "i", "Assignment"])
            df.to_csv(f"{OUT_FOLDER}/{"_".join(s.label.split())}.csv", index=False)
    
    return simulated_spectra, clusters, (h_x, n_x, h_x_t, n_x_t)

def save_2d_spectrum(spectrum, assignments, nuclei=["15N", "1H"], scale_nucl={"15N":10}, OUT_FOLDER="simulated_spectra"):

    data = np.array([spectrum.positions[0, :]*scale_nucl['N'], spectrum.positions[1, :], spectrum.intensities, assignments])
    df = pd.DataFrame(data.T, columns=["15N", "1H", "i", "Assignment"])
    df.to_csv(f"{OUT_FOLDER}/{"_".join(spectrum.label.split())}.csv", index=False)


def postprocess_simulated_chain_results(results_path,
                                  clusters,
                                 ):
    
    temps = [(25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
    paths = [glob(results_path + f"/{t1}C_vs_{t2}C_*.csv")[0] for t1, t2 in temps]
    
    mapped_df = pd.DataFrame()
    for path, t in zip(paths, temps):
        df = pd.read_csv(path)
        df[f"{t[0]}C_assignment"] = clusters[df['empirical_peak_idx']]
        df[f"{t[1]}C_assignment"] = clusters[df['theoretical_peak_idx']]
        
        df = df.loc[df.groupby('empirical_peak_idx')['flow'].idxmax()].reset_index(drop=True)
            
        df[f"{t[0]}C"] = list(zip(df[f'{t[0]}C_15N'].round(3), df[f'{t[0]}C_1H'].round(5)))
        df[f"{t[1]}C"] = list(zip(df[f'{t[1]}C_15N'].round(3), df[f'{t[1]}C_1H'].round(5)))
            
        if t[0] == 25:
            mapped_df = df[[f"{t[0]}C", f"{t[1]}C", f"{t[0]}C_assignment", f"{t[1]}C_assignment"]]
        else:
            mapped_df = pd.merge(mapped_df, df[[f"{t[0]}C", f"{t[1]}C", f"{t[1]}C_assignment"]], on=f"{t[0]}C", how="left")
    mapped_df = mapped_df.astype({f"{t}C_assignment": int for t in [25]})
    return mapped_df 

def postprocess_chain_results(results_path, separate=False):
    
    temps = [(25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
    paths = [glob(results_path + f"/{t1}C_vs_{t2}C_*.csv")[0] for t1, t2 in temps]
    
    mapped_df = pd.DataFrame()
    for path, t in zip(paths, temps):
        df = pd.read_csv(path)

        df = df.loc[df.groupby('empirical_peak_idx')['flow'].idxmax()].reset_index(drop=True)
        
        df[f"{t[0]}C"] = list(zip(df[f'{t[0]}C_15N'].round(3), df[f'{t[0]}C_1H'].round(5)))
        df[f"{t[1]}C"] = list(zip(df[f'{t[1]}C_15N'].round(3), df[f'{t[1]}C_1H'].round(5)))
        
        if t[0] == 25:
            mapped_df[f'{t[0]}C_15N'] = df[f'{t[0]}C_15N'] 
            mapped_df[f'{t[0]}C_1H'] = df[f'{t[0]}C_1H']
            mapped_df[f"{t[0]}C"] = df[f'{t[0]}C']
            mapped_df[f"{t[1]}C"] = df[f'{t[1]}C']
            if separate:
                mapped_df[f'{t[1]}C_15N'] = df[f'{t[1]}C_15N'] 
                mapped_df[f'{t[1]}C_1H'] = df[f'{t[1]}C_1H']
        else:
            if separate: 
                mapped_df = pd.merge(mapped_df, df[[f"{t[0]}C", f"{t[1]}C", f'{t[1]}C_15N', f'{t[1]}C_1H', ]], on=f"{t[0]}C", how="left")
            else:
                mapped_df = pd.merge(mapped_df, df[[f"{t[0]}C", f"{t[1]}C"]], on=f"{t[0]}C", how="left")

    return mapped_df 

def compute_distance_between_aligned(results_path):
    
    mapped_df = postprocess_chain_results(results_path)

    res_df = pd.DataFrame()
    
    temps = [(25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
    for t in temps:
        t0 = np.array(mapped_df[~mapped_df[f'{t[1]}C'].isna()][f'{t[0]}C'].tolist())
        t1 = np.array(mapped_df[~mapped_df[f'{t[1]}C'].isna()][f'{t[1]}C'].tolist())
        assert t0.shape == t1.shape
        ind = mapped_df[~mapped_df[f'{t[1]}C'].isna()].index
        diff = t1 - t0
        if len(diff) > 0:
            ind_to_d = dict(zip(ind, zip(np.linalg.norm(diff, axis=1), diff[:, 0], diff[:, 1])))
        else: ind_to_d = {}
        
        # save to new df
        res_df[f'{t[0]}C'] = mapped_df[f'{t[0]}C']
        res_df[f'{t[1]}C'] = mapped_df[f'{t[1]}C']
        for i, d in enumerate(['distance', 'shift in 15N', 'shift in 1H']):
            res_df[f'{t[0]}C vs {t[1]}C {d}'] = [ind_to_d[j][i] if j in ind_to_d else np.nan for j in range(mapped_df.shape[0])]
    
    return res_df

