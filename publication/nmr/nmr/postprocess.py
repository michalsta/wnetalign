import numpy as np
import pandas as pd
from glob import glob

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