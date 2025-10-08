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
    
    mapped_df = postprocess_results(results_path)
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

def plot_aligned_kmeans_with_peaks(results_path,
                      peaks_path,
                      cols_to_cluster = 2,
                      k=54,
                      cmap_names=["nipy_spectral", "gist_ncar"],
                      figsize=(9,6),
                      alpha=0.8,
                      s=[5, 10],
                      bbox = (1., -0.1),
                      ncols = 6,
                      c_mod = 7,
                      hlim = (6.5, 10.75), 
                      nlim = (102, 136),
                      legend = True,
                     ):
    
    mapped_df = postprocess_results(results_path)
    mapped_df = mapped_df[mapped_df.columns[:2+cols_to_cluster]]
    X = np.array([mapped_df["15N"], mapped_df["1H"]]).T
    X_scaled = np.array([mapped_df["15N"]/10, mapped_df["1H"]]).T
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_scaled)
    
    cmap = plt.get_cmap(cmap_names[0])
    colors = cmap(np.linspace(0, 1, k))  
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    for l in range(k):
        new_X = [(round(x[0], 3), x[1]) for x in X[kmeans.labels_ == l]]
        new_df = mapped_df[mapped_df["25C"].isin(new_X)][mapped_df.columns[2:]]
        values = set(new_df.to_numpy().flatten()) # collapse dataframe to 1D array
        if np.nan in values: values.remove(np.nan) # remove np.nan
        values = np.array(list(values)) # cast to array
        if values.size > 0: ax.scatter(values[:, 1], values[:, 0], s=s[0], alpha=alpha, color=colors[l])
    
    peaks = {}
    for i, path in enumerate(sorted(glob(peaks_path + "/GB1_[2-5][0|5].csv"))):
        df = pd.read_csv(path)
        df = df.sort_values(by=["Assignment"])
        df = df.reset_index(drop=True)
        if i==0:
            cmap = plt.get_cmap(cmap_names[1])
            colors = cmap(np.linspace(0, 1, df.shape[0]))
            map_colors = {df.loc[l, "Assignment"]: colors[(c_mod*l + l%c_mod)%k] for l in range(df.shape[0])} 
            peaks = {df.loc[l, "Assignment"]: [] for l in range(df.shape[0])}
        for l in df["Assignment"]:
            peaks[l].append([df[df['Assignment'] == l]['w2'].item(), df[df['Assignment'] == l]['w1'].item()])
    
    for l in peaks:
        peaks[l] = pd.DataFrame(peaks[l], columns=["1H", "15N"])
        ax.plot(peaks[l]["1H"], peaks[l]["15N"], linewidth=1, ms=s[1], alpha=alpha, marker="X", color=map_colors[l], label=l)
    
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
                
    return ax

def plot_aligned_kmeans_with_peaks_restricted(results_path,
                                              peaks_path,
                                              cols_to_cluster = 2,
                                              k=54,
                                              cmap_names=["nipy_spectral", "gist_ncar"],
                                              figsize=(9,6),
                                              alpha=0.8,
                                              s=[5, 5],
                                              bbox = (1., -0.1),
                                              ncols = 6,
                                              hlim = (6.5, 10.75),
                                              nlim = (102, 136),
                                              text = False,
                                              text_shift = [0., 0.15],
                                              legend = True,
                                ):
    
    mapped_df = postprocess_results(results_path)
    mapped_df = mapped_df[mapped_df.columns[:2+cols_to_cluster]]
    
    X = np.array([mapped_df["15N"], mapped_df["1H"]]).T
    X_scaled = np.array([mapped_df["15N"]/10, mapped_df["1H"]]).T
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_scaled)
    
    condition = (nlim[0] < X[:, 0]) & (X[:, 0] < nlim[1]) & (hlim[0] < X[:, 1]) & (X[:, 1] < hlim[1])
    X = X[condition]
    kmeans_labels = kmeans.labels_[condition]
    
    cmap = plt.get_cmap(cmap_names[0])
    colors = cmap(np.linspace(0, 1, np.unique(kmeans_labels).shape[0]))
        
    fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    for i, l in enumerate(np.unique(kmeans_labels)):
        new_X = [(round(x[0], 3), x[1]) for x in X[kmeans_labels == l]]
        new_df = mapped_df[mapped_df["25C"].isin(new_X)][mapped_df.columns[2:]]
        values = set(new_df.to_numpy().flatten()) # collapse dataframe to 1D array
        if np.nan in values: values.remove(np.nan) # remove np.nan
        values = np.array(list(values)) # cast to array
        if values.size > 0: ax.scatter(values[:, 1], values[:, 0], s=s[0], alpha=alpha, color=colors[i])
    
    peaks = {}
    for i, path in enumerate(sorted(glob(peaks_path+ "/GB1_[2-5][0|5].csv"))):
        df = pd.read_csv(path)
        df = df.sort_values(by=["Assignment"])
        for l in df["Assignment"]:
            if (hlim[0] < df[df['Assignment'] == l]['w2'].item() < hlim[1]) and (nlim[0] < df[df['Assignment'] == l]['w1'].item() < nlim[1]):
                if l not in peaks:
                    peaks[l] = [[df[df['Assignment'] == l]['w2'].item(), df[df['Assignment'] == l]['w1'].item()]]
                else: peaks[l].append([df[df['Assignment'] == l]['w2'].item(), df[df['Assignment'] == l]['w1'].item()])
    
    cmap = plt.get_cmap(cmap_names[1])
    colors = cmap(np.linspace(0, 1, len(peaks)))
    colors = np.random.permutation(colors)
    map_colors = {l: colors[i] for i, l in enumerate(peaks)} 
    
    for l in peaks:
        if peaks[l]:
            peaks[l] = np.array(peaks[l])
            ax.plot(peaks[l][:, 0], peaks[l][:, 1], ms=s[1], alpha=alpha, marker="X", color=map_colors[l], label=l)
            if text: ax.plot(peaks[l][0, 0], peaks[l][0, 1], ms=s[1]+1, marker="s", markeredgecolor="k", markerfacecolor="None", markeredgewidth=2) 
            if text: ax.text(peaks[l][0, 0]+text_shift[0], peaks[l][0, 1]+text_shift[1], l, 
                        fontsize=2*s[0], 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        # transform=ax.transAxes,
                       )
    
    plt.xlabel('1H (ppm)')
    plt.xlim(hlim[0], hlim[1])
    plt.ylabel('15N (ppm)')
    plt.ylim(nlim[0], nlim[1])
    plt.title('$^1$H $^{15}$N HSQC spectra of GB1 protein after alignment')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Often used for NMR spectra
    plt.gca().invert_yaxis()  # Often used for NMR spectra
    
    if legend: ax.legend(bbox_to_anchor = bbox, ncols = ncols)        
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
    # df = df.sort_values(by=["w1", "w2"])
    df = df.sort_values(by=["Assignment"])
    df = df.reset_index(drop=True)
    C = np.array([df.w1, df.w2]).T

    mapped_df = postprocess_results(results_path)
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
                        # fontsize=9, 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        # transform=ax.transAxes,
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

def plot_aligned_with_peaks(results_path,
                      peaks_path,
                      cmap_names=["nipy_spectral", "gist_ncar"],
                      figsize=(9,6),
                      alpha=0.8,
                      s=[5, 10],
                      bbox = (1., -0.1),
                      ncols = 6,
                      c_mod = 7,
                      hlim = (6.5, 10.75), 
                      nlim = (102, 136),
                      text = False,
                      legend = True,
                     ):

    df = pd.read_csv(peaks_path + "/GB1_25.csv")
    df = df.sort_values(by=["Assignment"])
    df = df.reset_index(drop=True)
    C = np.array([df.w1, df.w2]).T

    mapped_df = postprocess_results(results_path)
    X = np.array([mapped_df["15N"], mapped_df["1H"]]).T
    distances = cdist(X, C)            # Compute distances (n_samples x n_clusters)
    labels = np.argmin(distances, axis=1)  # Assign to closest centroid

    n_peaks = C.shape[0]
    # print("number of clusters:", n_peaks)
    cmap = plt.get_cmap(cmap_names[0])
    colors = cmap(np.linspace(0, 1, n_peaks))
    map_colors = {df.loc[l, "Assignment"]: colors[(c_mod*l + l%c_mod)%n_peaks] for l in range(n_peaks)}

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for l in np.unique(labels):
        new_X = [(round(x[0], 3), x[1]) for x in X[labels == l]]
        new_df = mapped_df[mapped_df["25C"].isin(new_X)][mapped_df.columns[2:]]
        values = set(new_df.to_numpy().flatten()) # collapse dataframe to 1D array
        if np.nan in values: values.remove(np.nan) # remove np.nan
        values = np.array(list(values)) # cast to array
        ax.scatter(values[:, 1], values[:, 0], s=s[0], alpha=alpha, color=map_colors[df.loc[l, "Assignment"]])#, label=df.loc[l, "Assignment"])
        if text: # plot 25C assigned peak
            if (hlim[0] < df.loc[l, "w2"] < hlim[1]) and (nlim[0] < df.loc[l, "w1"] < nlim[1]):
                ax.plot(df.loc[l, "w2"], df.loc[l, "w1"], ms=s[1]+1, marker="s", markeredgecolor="k", markerfacecolor="None", markeredgewidth=2)
                ax.text(df.loc[l, "w2"]-0.1, df.loc[l, "w1"]-0.1, df.loc[l, "Assignment"], 
                        # fontsize=9, 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        # transform=ax.transAxes,
                       )
    # this part plots peaks in different temperatures
    peaks = {}
    for i, path in enumerate(sorted(glob(peaks_path + "/GB1_[2-5][0|5].csv"))):
        df = pd.read_csv(path)
        df = df.sort_values(by=["Assignment"])
        df = df.reset_index(drop=True)
        if i==0:
            cmap = plt.get_cmap(cmap_names[1])
            colors = cmap(np.linspace(0, 1, df.shape[0]))
            map_colors = {df.loc[l, "Assignment"]: colors[(c_mod*l + l%c_mod)%np.unique(labels).shape[0]] for l in range(df.shape[0])} 
            peaks = {df.loc[l, "Assignment"]: [] for l in range(df.shape[0])}
        for l in df["Assignment"]:
            peaks[l].append([df[df['Assignment'] == l]['w2'].item(), df[df['Assignment'] == l]['w1'].item()])
    
    for l in peaks:
        peaks[l] = pd.DataFrame(peaks[l], columns=["1H", "15N"])
        ax.plot(peaks[l]["1H"], peaks[l]["15N"], linewidth=1, ms=s[1], alpha=alpha, marker="X", color=map_colors[l], label=l)
    
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
                
    return ax 

def plot_aligned_with_peaks_restricted(results_path,
                                        peaks_path,
                                        peaks_for_clustering_path,
                                        cols_to_crop = 3,
                                        cmap_names=["nipy_spectral", "gist_ncar"],
                                        figsize=(9,6),
                                        alpha=0.8,
                                        s=[5, 5],
                                        bbox = (1., -0.1),
                                        ncols = 6,
                                        hlim = (6.5, 10.75),
                                        nlim = (102, 136),
                                        text = [False, False],
                                        text_shift = [0.1, 0],
                                        legend = True,
                                        permute=True,
                                ):

    df = pd.read_csv(peaks_for_clustering_path)
    df = df.sort_values(by=["Assignment"])
    df = df.reset_index(drop=True)
    C = np.array([df.w1, df.w2]).T

    mapped_df = postprocess_results(results_path)
    mapped_df = mapped_df[mapped_df.columns[:2+cols_to_crop]]

    X = np.array([mapped_df["15N"], mapped_df["1H"]]).T
    distances = cdist(X, C)            # Compute distances (n_samples x n_clusters)
    labels = np.argmin(distances, axis=1)  # Assign to closest centroid

    condition = (nlim[0] < X[:, 0]) & (X[:, 0] < nlim[1]) & (hlim[0] < X[:, 1]) & (X[:, 1] < hlim[1])
    X = X[condition]
    labels = labels[condition]
    
    cmap = plt.get_cmap(cmap_names[0])
    colors = cmap(np.linspace(0, 1, np.unique(labels).shape[0]))
    # map_colors = {df.loc[l, "Assignment"]: colors[(c_mod*l + l%c_mod)%np.unique(labels).shape[0]] for i, l in enumerate(np.unique(labels))}
    map_colors = {df.loc[l, "Assignment"]: colors[i] for i, l in enumerate(np.unique(labels))}

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for l in np.unique(labels):
        new_X = [(round(x[0], 3), x[1]) for x in X[labels == l]]
        new_df = mapped_df[mapped_df["25C"].isin(new_X)][mapped_df.columns[2:]]
        values = set(new_df.to_numpy().flatten()) # collapse dataframe to 1D array
        if np.nan in values: values.remove(np.nan) # remove np.nan
        values = np.array(list(values)) # cast to array
        # if values.size > 0: ax.scatter(values[:, 1], values[:, 0], s=s[0], alpha=alpha, color=colors[i], label=l)
        if values.size > 0: ax.scatter(values[:, 1], values[:, 0], s=s[0], alpha=alpha, color=map_colors[df.loc[l, "Assignment"]], label=df.loc[l, "Assignment"])
        if text[0]: # plot 25C assigned peak
            if (hlim[0] < df.loc[l, "w2"] < hlim[1]) and (nlim[0] < df.loc[l, "w1"] < nlim[1]):
                ax.plot(df.loc[l, "w2"], df.loc[l, "w1"], ms=s[1]+1, marker="s", markeredgecolor="k", markerfacecolor="None", markeredgewidth=2)
                ax.text(df.loc[l, "w2"]+text_shift[0], df.loc[l, "w1"]+text_shift[1], df.loc[l, "Assignment"], 
                        fontsize=2*s[0], 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        # transform=ax.transAxes,
                       )
    
    peaks = {}
    for i, path in enumerate(sorted(glob(peaks_path + "/GB1_[2-5][0|5].csv"))):
        df = pd.read_csv(path)
        df = df.sort_values(by=["Assignment"])
        for l in df["Assignment"]:
            if (hlim[0] < df[df['Assignment'] == l]['w2'].item() < hlim[1]) and (nlim[0] < df[df['Assignment'] == l]['w1'].item() < nlim[1]):
                if l not in peaks:
                    peaks[l] = [[df[df['Assignment'] == l]['w2'].item(), df[df['Assignment'] == l]['w1'].item()]]
                else: peaks[l].append([df[df['Assignment'] == l]['w2'].item(), df[df['Assignment'] == l]['w1'].item()])
    
    cmap = plt.get_cmap(cmap_names[1])
    colors = cmap(np.linspace(0, 1, len(peaks)))
    if permute: colors = np.random.permutation(colors)
    map_colors = {l: colors[i] for i, l in enumerate(peaks)} 
    
    for l in peaks:
        if peaks[l]:
            peaks[l] = np.array(peaks[l])
            ax.plot(peaks[l][:, 0], peaks[l][:, 1], ms=s[1], alpha=alpha, marker="X", color=map_colors[l], label=l)
            if text[1]: ax.text(peaks[l][-1, 0]+0.05, peaks[l][-1, 1]+0.05, l, 
                                    # fontsize=9, 
                                    horizontalalignment='center', 
                                    verticalalignment='center', 
                                    # transform=ax.transAxes,
                                   )
    
    plt.xlabel('1H (ppm)')
    plt.xlim(hlim[0], hlim[1])
    plt.ylabel('15N (ppm)')
    plt.ylim(nlim[0], nlim[1])
    plt.title('$^1$H $^{15}$N HSQC spectra of GB1 protein after alignment')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Often used for NMR spectra
    plt.gca().invert_yaxis()  # Often used for NMR spectra
    
    if legend: ax.legend(bbox_to_anchor = bbox, ncols = ncols)
                
    return ax

# Plot for simulated
def plot_simulated_ground_truth(mapped_df, 
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
    Plot results on simulated spectra colored by ground truth (original clusters)
    """
    
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(np.unique(clusters))))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)

    for c in np.unique(clusters): # for each cluster
        coords = np.concatenate([mapped_df[mapped_df[f'{t}_assignment'] == c][f'{t}'].to_numpy().flatten() for t in sorted(list({c[:3] for c in mapped_df.columns}))])
        coords = set(coords)
        if np.nan in coords: coords.remove(np.nan) # remove np.nan
        coords = np.fromiter(coords, dtype=np.dtype((float, 2)))
        if coords.size > 0: ax.scatter(coords[:, 1], coords[:, 0], s=s, alpha=alpha, color=colors[c], label=c)

    plt.xlabel('1H (ppm)')
    plt.xlim(hlim[0], hlim[1])
    plt.ylabel('15N (ppm)')
    plt.ylim(nlim[0], nlim[1])
    plt.title('GB1 simulated - ground truth')
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