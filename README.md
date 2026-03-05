# WNetAlign

WNetAlign is a tool for MS (Mass Spectrometry) or NMR (Nuclear Magnetic Resonance) spectral alignment using Truncated Wasserstein Metric-based methods. It offers algorithms and utilities for comparing and mapping spectral features between datasets, leveraging advanced optimal transport techniques and spectral topology.

## Features

- Truncated Wasserstein metric-based spectral alignment
- Algorithms for spectral feature comparison and mapping
- Utilities for preprocessing and visualization

## Installation

```bash
pip install wnetalign
```

## Usage

Usage examples can be found in the `tutorials` directory, both for LCMS and NMR datasets. 

### NMR tutorials 

Tutorials and code allowing to reproduce the results and figures from publication (cite) can be found in `tutorials/nmr/` folder. 

#### 2D NMR data

The 2D 15N-1H HSQC spectra are available in `tutorials/nmr/2D/15_HSQC_GB1_reduced` folder. 

* `NMR_2D_original.ipynb` - plotting data distribution, chain alignment of the series of 15N-1H HSQC spectra measures in different temperatures 
* `NMR_2D_simulated.ipynb` - simulating synthetic temperature series, validation using standard and extended metrics, heatmaps
* `NMR_2D_seeds.ipynb` - performing alignments for simulated replicate spectra with different random seeds 

#### 4D NMR data

The 4D CCNOESY spectrum of 2LX7 pdb structure can be found in the `tutorials/nmr/4D/2LX7`

* `NMR_4D.ipynb` - simulating the replicate of the 4D 2LX7 CCNOESY spectrum and validating the alignment with standard and extended performance metrics

#### 7D NMR data

* `NMR_7D.ipynb` - simulating the replicates of the 17 hypothetical 7D NMR spectra and validating the alignments with extended performance metrics

## License

This project is licensed under the MIT License.
