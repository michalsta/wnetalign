import numpy as np
import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

from nmr.utils_2d import postprocess_chain_results

def plot_temperatures(data_path, 
                      max_peak_fraction=0.1, 
                      intensity_threshold=0.01,
                      cmap_name="plasma", 
                      figsize=(9,6), 
                      dpi=200,
                      alpha=0.8,
                      s=3,
                      hlim = (6, 11), 
                      nlim = (100, 136),
                      save_path = None,
                     ):

    if isinstance(data_path, list):
        for path in data_path:
            if not os.path.isfile(path): raise Exception("Path in a list is not a file path.")
            if path[-4:] != ".csv": raise Exception("Incorrect file extention,. Must be csv")
            paths = data_path
    elif os.path.isdir(data_path):
        paths = sorted(glob(data_path + "/*.csv"))
    else:
        raise Exception("Incorrect path. Should be 1) directory path with csv files 2) list of paths to csv files")
        
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(paths)))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True) #constrained_layout=True, squeeze=False, 
    for i, path in enumerate(paths):
        label = path.split("/")[-1].split(".")[0].split("_")[-1]
        df = pd.read_csv(path)
        if max_peak_fraction: 
            df = df[df['i'] > df.i.max()*max_peak_fraction]
        else: 
            df = df[df['i'] > intensity_threshold]
        ax.scatter(df['1H'], df['15N'], s=s, alpha=alpha, color=colors[i], label=label)
    
    plt.xlabel('1H (ppm)')
    plt.xlim(hlim[0], hlim[1])
    plt.ylabel('15N (ppm)')
    plt.ylim(nlim[0], nlim[1])
    plt.title('$^1$H $^{15}$N HSQC spectra of GB1 protein')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Often used for NMR spectra
    plt.gca().invert_yaxis()  # Often used for NMR spectra
    plt.legend(loc="upper left")

    if save_path: plt.savefig(save_path)
    return ax

def plot_aligned_kmeans(results_path,
                        k=54, 
                        cmap_name="nipy_spectral", 
                        figsize=(9,6), 
                        alpha=0.8, 
                        s=5,
                        bbox = (1., -0.1),
                        ncols = 9,
                        hlim = (6, 11),
                        nlim = (100, 136),
                        legend = True,
                        save_path=None,
                        ):
    
    mapped_df = postprocess_chain_results(results_path)
    X = np.array([mapped_df["15N"], mapped_df["1H"]]).T
    X_scaled = np.array([mapped_df["15N"]/10, mapped_df["1H"]]).T
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_scaled)

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, k))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for l in range(k):
        new_X = [(round(x[0], 3), x[1]) for x in X[kmeans.labels_ == l]]
        new_df = mapped_df[mapped_df["25C"].isin(new_X)][mapped_df.columns[2:]]
        values = set(new_df.to_numpy().flatten()) # collapse dataframe to 1D array
        if np.nan in values: values.remove(np.nan) # remove np.nan
        values = np.array(list(values)) # cast to array
        if values.size > 0: ax.scatter(values[:, 1], values[:, 0], s=s, alpha=alpha, color=colors[l], label=l)

    plt.xlabel('1H (ppm)')
    plt.xlim(hlim[0], hlim[1])
    plt.ylabel('15N (ppm)')
    plt.ylim(nlim[0], nlim[1])
    plt.title('$^1$H $^{15}$N HSQC spectra of GB1 protein after alignment')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Often used for NMR spectra
    plt.gca().invert_yaxis()  # Often used for NMR spectra

    if legend: 
        handles, h_labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, h_labels)) if l not in h_labels[:i]]
        unique = sorted(unique, key=lambda x: int(x[1]))
        ax.legend(*zip(*unique), bbox_to_anchor = bbox, ncols = ncols)
    if save_path: plt.savefig(save_path)
    return ax

def plot_aligned_with_assignment(results_path, 
                                 peaks_path, 
                                 cmap_name="nipy_spectral",
                                 figsize=(9,6), 
                                 alpha=0.8, 
                                 s=5,
                                 bbox = (1., -0.1),
                                 ncols = 9,
                                 c_mod = 7,
                                 hlim = (6, 11),
                                 nlim = (100, 136),
                                 text = False,
                                 legend = True,
                                 save_path=None,
                       ):

    df = pd.read_csv(peaks_path + "/GB1_25.csv")
    df = df.sort_values(by=["Assignment"])
    df = df.reset_index(drop=True)
    C = np.array([df.w1, df.w2]).T

    mapped_df = postprocess_chain_results(results_path)
    X = np.array([mapped_df["15N"], mapped_df["1H"]]).T
    distances = cdist(X, C)            # Compute distances (n_samples x n_clusters)
    labels = np.argmin(distances, axis=1)  # Assign to closest centroid

    n_peaks = C.shape[0]
    # print("number of clusters:", n_peaks)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n_peaks))
    map_colors = {df.loc[l, "Assignment"]: colors[(c_mod*l + l%c_mod)%n_peaks] for l in range(n_peaks)}

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for l in np.unique(labels):
        new_X = [(round(x[0], 3), x[1]) for x in X[labels == l]]
        new_df = mapped_df[mapped_df["25C"].isin(new_X)][mapped_df.columns[2:]]
        values = set(new_df.to_numpy().flatten()) # collapse dataframe to 1D array
        if np.nan in values: values.remove(np.nan) # remove np.nan
        values = np.array(list(values)) # cast to array
        ax.scatter(values[:, 1], values[:, 0], s=s, alpha=alpha, color=map_colors[df.loc[l, "Assignment"]], label=df.loc[l, "Assignment"])
        if text: # plot 25C assigned peak
            if (hlim[0] < df.loc[l, "w2"] < hlim[1]) and (nlim[0] < df.loc[l, "w1"] < nlim[1]):
                ax.plot(df.loc[l, "w2"], df.loc[l, "w1"], marker="o", markersize=4, markeredgecolor="k", markerfacecolor="k") 
                ax.text(df.loc[l, "w2"]-0.1, df.loc[l, "w1"]-0.1, df.loc[l, "Assignment"], 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                       )
    
    plt.xlabel('1H (ppm)')
    plt.xlim(hlim[0], hlim[1])
    plt.ylabel('15N (ppm)')
    plt.ylim(nlim[0], nlim[1])
    plt.title('$^1$H $^{15}$N HSQC spectra of GB1 protein after alignment')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Often used for NMR spectra
    plt.gca().invert_yaxis()  # Often used for NMR spectra

    if legend: 
        handles, h_labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, h_labels)) if l not in h_labels[:i]]
        unique = sorted(unique, key=lambda x: x[1])
        ax.legend(*zip(*unique), bbox_to_anchor = bbox, ncols = ncols)

    if save_path: plt.savefig(save_path)
    return ax

def plot_simulated_alignment(mapped_df, 
                             clusters,
                             cmap_name="nipy_spectral", 
                             figsize=(9,6), 
                             dpi=200,
                             alpha=0.8, 
                             s=5,
                             bbox = (1., -0.1),
                             ncols = 9,
                             hlim = (6, 11),
                             nlim = (100, 136),
                             legend = True,
                             save_path=None,
                             ):
    
    """
    Plot chain alignment results colored by class determined by 25C.
    """
    
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(np.unique(clusters))))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    
    start = '25C'
    temps = sorted(list({c[:3] for c in mapped_df.columns}))
    
    # colored by predicted class (class from 25C is followed by the chain)
    for c in np.unique(clusters): # for each cluster from 25C
        new_df = mapped_df[mapped_df[f'{start}_assignment'] == c][temps]
        values = set(new_df.to_numpy().flatten())
        if np.nan in values: values.remove(np.nan) # remove np.nan
        values = np.fromiter(values, dtype=np.dtype((float, 2)))
        if values.size > 0: ax.scatter(values[:, 1], values[:, 0], s=s, alpha=alpha, color=colors[c], label=c)

    plt.xlabel('1H (ppm)')
    plt.xlim(hlim[0], hlim[1])
    plt.ylabel('15N (ppm)')
    plt.ylim(nlim[0], nlim[1])
    plt.title('GB1 simulated - aligned')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Often used for NMR spectra
    plt.gca().invert_yaxis()  # Often used for NMR spectra

    if legend: 
            handles, h_labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, h_labels)) if l not in h_labels[:i]]
            unique = sorted(unique, key=lambda x: int(x[1]))
            ax.legend(*zip(*unique), bbox_to_anchor = bbox, ncols = ncols)
        
    if save_path: plt.savefig(save_path)
    return ax