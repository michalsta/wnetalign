import numpy as np
from sklearn.metrics import confusion_matrix

from nmr.align import align_pair 
from nmr.utils_2d import simulate_2d_shifts
from nmr.postprocess import postprocess_simulated_pair
from nmr.metrics import extended_confusion_matrix, metrics_from_cm


def metrics_multiple_simulations(n, 
                                 DATA_PATH="./15N_HSQC_GB1_reduced", 
                                 k=54,
                                 scale_nucl={"N": 10},
                                 nuclei=['15N', '1H'],
                                 max_peak_fraction=0.01, 
                                 h_lim = (0.01, 0.025),
                                 n_lim = (0.0, 0.015),
                                 h_t_lim = (0, 0.01),
                                 n_t_lim = (0, 0.005),
                                 max_distance=0.05, 
                                 trash_cost=0.09, 
                                 extended = False,
                                 warn=True,
                                 default_rng = False,
                                ):
    metrics = {}
    for i in range(n):
        simulated_spectra, clusters, _ = simulate_2d_shifts(data_path=DATA_PATH,
                                                            k=k,
                                                            scale_nucl=scale_nucl,
                                                            max_peak_fraction=max_peak_fraction,
                                                            h_lim = h_lim,
                                                            n_lim = n_lim,
                                                            h_t_lim = h_t_lim,
                                                            n_t_lim = n_t_lim,
                                                            seed = i,
                                                            default_rng = default_rng,
                                                            )
        for i in range(len(simulated_spectra)-1):
            s1, s2 = simulated_spectra[i:i+2]
            l1 = s1.label.split(" ")[-1]
            l2 = s2.label.split(" ")[-1]
            if f"{l1} vs {l2}" not in metrics: 
                metrics[f"{l1} vs {l2}"] = {'Accuracy':[], 'Precision (micro)':[], 'Recall (micro)':[], 'Precision (macro)':[], 'Recall (macro)':[]}
                # metrics[f"{l1} vs {l2}"] = {'Accuracy':[], 'Precision (micro)':[], 'Recall (micro)':[], 'Precision (macro)':[], 'Recall (macro)':[]}
            _, r = align_pair(s1,
                              s2,
                              max_distance=max_distance,
                              trash_cost=trash_cost,
                              intensity_scaling=None,
                              normalize = True,
                              verbose=False,
                              )
                
            df_max = postprocess_simulated_pair(s1, l1, s2, l2, r, clusters, scale_nucl, nuclei, find_max_flow=True)
            
            if extended: cm = extended_confusion_matrix(df_max, s1, s2, l1, l2, clusters)
            else: cm = confusion_matrix(df_max[f"{l1}_assignment"], df_max[f"{l2}_assignment"], labels=np.sort(np.unique(clusters)))
                    
            metrics_cm = metrics_from_cm(cm, warn=warn)
            
            metrics[f"{l1} vs {l2}"]['Accuracy'].append(metrics_cm['Accuracy'])
            metrics[f"{l1} vs {l2}"] ['Precision (micro)'].append(metrics_cm['Precision (micro)'])
            metrics[f"{l1} vs {l2}"] ['Recall (micro)'].append(metrics_cm['Recall (micro)'])
            metrics[f"{l1} vs {l2}"] ['Precision (macro)'].append(metrics_cm['Precision (macro)'])
            metrics[f"{l1} vs {l2}"] ['Recall (macro)'].append(metrics_cm['Recall (macro)'])

    return metrics
