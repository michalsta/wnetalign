import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix

from nmr.align import align_pair
from nmr.postprocess import postprocess_simulated_pair

def metrics_simulated_CM(spectra, max_distances, trash_costs, classes, scale_nucl={'15N':10}, nuclei=['15N', '1H'], extended=False, warn=True):

    metrics = {'Accuracy':{}, 'Precision (micro)':{}, 'Recall (micro)':{}, 'Precision (macro)':{}, 'Recall (macro)':{}}
    
    for i in range(len(spectra)-1):
        s1, s2 = spectra[i:i+2]
        l1 = s1.label
        l2 = s2.label
    
        df_acc = pd.DataFrame(0.0, columns=trash_costs, index=max_distances)
        df_prec_micro = pd.DataFrame(0.0, columns=trash_costs, index=max_distances)
        df_rec_micro = pd.DataFrame(0.0, columns=trash_costs, index=max_distances)
        df_prec_macro = pd.DataFrame(0.0, columns=trash_costs, index=max_distances)
        df_rec_macro = pd.DataFrame(0.0, columns=trash_costs, index=max_distances)
        
        for m_d in max_distances:
            for t_c in trash_costs:
                _, r = align_pair(s1,
                                  s2,
                                  max_distance = m_d,
                                  trash_cost = t_c,
                                  normalize = True,
                                 )
                
                df_max = postprocess_simulated_pair(s1, l1, s2, l2, r, classes, scale_nucl, nuclei, find_max_flow=True)
                
                if extended: cm = extended_confusion_matrix(df_max, s1, s2, l1, l2, classes)
                else: cm = confusion_matrix(df_max[f"{l1}_assignment"], df_max[f"{l2}_assignment"], labels=np.sort(np.unique(classes)))
                    
                metrics_cm = metrics_from_cm(cm, warn=warn)

                df_acc.loc[m_d, t_c] = metrics_cm["Accuracy"]
                df_prec_micro.loc[m_d, t_c] = metrics_cm['Precision (micro)']
                df_rec_micro.loc[m_d, t_c] = metrics_cm['Recall (micro)']
                df_prec_macro.loc[m_d, t_c] = metrics_cm['Precision (macro)']
                df_rec_macro.loc[m_d, t_c] = metrics_cm['Recall (macro)']
    
        metrics['Accuracy'][f"{l1} vs {l2}"] = df_acc
        metrics['Precision (micro)'][f"{l1} vs {l2}"] = df_prec_micro
        metrics['Recall (micro)'][f"{l1} vs {l2}"] = df_rec_micro
        metrics['Precision (macro)'][f"{l1} vs {l2}"] = df_prec_macro
        metrics['Recall (macro)'][f"{l1} vs {l2}"] = df_rec_macro
        
    return metrics

def metrics_from_cm(cm, warn=True):

    """Note that in case of extended CM micro averaging does not make sense, 
    since micro precision and micro recall are equal to accuracy"""

    metrics = {}
    
    assert cm.shape[0] == cm.shape[1]
    assert np.all(np.abs(cm) == cm)

    if np.sum(cm) > 0: metrics['Accuracy'] = np.trace(cm)/np.sum(cm)
    else: 
        if warn: warnings.warn("Accuracy incorrectly defined: division by 0. Accuracy set to 0.")
        metrics['Accuracy'] = 0

    denominator = np.sum(cm, axis=0)
    if np.sum(denominator) > 0: metrics['Precision (micro)'] = np.trace(cm) / np.sum(denominator)
    else:
        if warn: warnings.warn("Precision (micro) incorrectly defined: division by 0. Precision set to 0.")
        metrics['Precision (micro)'] = 0
        
    if np.all(denominator > 0): metrics['Precision (macro)'] = np.mean(np.diag(cm) / denominator)
    else: 
        if warn: warnings.warn("Precision (macro) incorrectly defined: division by 0. Precision set to 0.")
        denominator[denominator == 0] = 1 
        metrics['Precision (macro)'] = np.mean(np.diag(cm) / denominator)

    denominator = np.sum(cm, axis=1)
    if np.sum(denominator) > 0:
        metrics['Recall (micro)'] = np.trace(cm) / np.sum(denominator)
    else:
        if warn: warnings.warn("Recall (micro) incorrectly defined: division by 0. Recall set to 0.")
        metrics['Recall (micro)'] = 0
        
    if np.all(denominator > 0):
        metrics['Recall (macro)'] = np.mean(np.diag(cm) / denominator)
    else:
        if warn: warnings.warn("Recall (macro) incorrectly defined: division by 0. Recall set to 0.")
        denominator[denominator == 0] = 1 
        metrics['Recall (macro)'] = np.mean(np.diag(cm) / denominator)

    return metrics


def extended_confusion_matrix(df, s1, s2, l1, l2, classes):
    
    cm = confusion_matrix(df[f"{l1}_assignment"], df[f"{l2}_assignment"], labels=np.sort(np.unique(classes)))

    # count unmatched peaks in s1 - for extended false negatives
    unmatched_s1 = np.setdiff1d(np.arange(s1.positions.shape[1]), df.empirical_peak_idx)
    unmatched_s1 = classes[unmatched_s1]
    unmatched_s1 = dict(zip(*np.unique(unmatched_s1, return_counts=True)))
    unmatched_s1 = np.fromiter(( unmatched_s1[i] if i in unmatched_s1 else 0 for i in np.sort(np.unique(classes)) ), dtype=float)
    
    # count unmatched peaks in s2 - for extended false positives
    unmatched_s2 = np.setdiff1d(np.arange(s2.positions.shape[1]), df.theoretical_peak_idx)
    unmatched_s2 = classes[unmatched_s2]
    unmatched_s2 = dict(zip(*np.unique(unmatched_s2, return_counts=True)))
    unmatched_s2 = np.fromiter(( unmatched_s2[i] if i in unmatched_s2 else 0 for i in np.sort(np.unique(classes)) ), dtype=float)

    cm_extended = np.column_stack((cm, unmatched_s1))
    cm_extended = np.vstack((cm_extended, np.append(unmatched_s2, np.array([0]))))
    cm_extended = cm_extended.astype(np.dtype('int64'))

    return cm_extended

def flow_costs(spectra, max_distances, trash_costs, intensity_scaling=None):
    
    costs = {}
    for i in range(len(spectra)-1):
        s1, s2 = spectra[i:i+2]

        df_cost = pd.DataFrame(0.0, columns=trash_costs, index=max_distances)
        for m_d in max_distances:
            for t_c in trash_costs:
                s, _ = align_pair(s1,
                                  s2,
                                  max_distance = m_d,
                                  trash_cost = t_c,
                                  normalize = True,
                                  intensity_scaling=intensity_scaling,
                                 )
                
                df_cost.loc[m_d, t_c] = s.total_cost()

        costs[f"{s1.label} vs {s2.label}"] = df_cost
    return costs



