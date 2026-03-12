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

## Tutorials 

Tutorials showing how to apply `wnetalign` to nmr data can be found in `tutorials/nmr` folder, while the example how to perform alignment of th LC-MS data can be found in `tutorials/lcms`. 

## Results from publication

To reproduce the results and figures from the paper *"WNetAlign: Fast and Accurate Spectra Alignment Using Truncated Wasserstein Distance and Network Simplex"* use the scripts and notebooks provided in the `publication/nmr` folder for NMR data and `publication/lcms` for LC-MS data.

#### 2D NMR data

The 2D 15N-1H HSQC spectra are available in `publication/nmr/2D/15_HSQC_GB1_reduced` folder. 

* `NMR_2D_original.ipynb` - plotting data distribution, chain alignment of the series of 15N-1H HSQC spectra measured in different temperatures
* `NMR_2D_simulated.ipynb` - simulating synthetic temperature series, validation of alignment procedure using standard and extended metrics, heatmaps
* `NMR_2D_seeds.ipynb` - performing alignments for the synthetic temperature series with different random seeds 

#### 4D NMR data

The 4D CCNOESY spectrum of 2LX7 pdb structure can be found in the `publication/nmr/4D/2LX7`.

* `NMR_4D.ipynb` - simulating the replicate of the 4D 2LX7 CCNOESY spectrum and validating the alignment between the original spectrum and the replicate with standard and extended performance metrics

#### 7D NMR data

The hypothetical 7D NMR spectra can be found in `publication/nmr/7D/NMR_7D`.

* `NMR_7D.ipynb` - simulating the replicates of the 17 hypothetical 7D NMR spectra and validating the alignments between the original hypothetical spectra and the replicates with extended performance metrics

## License

This project is licensed under the MIT License.
