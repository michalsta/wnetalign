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

```python
import numpy as np
from wnetalign import Spectrum, WNetAligner
from wnet.distances import DistanceMetric

# positions shape: (dim, num_peaks); intensities shape: (num_peaks,)
S1 = Spectrum(positions=np.array([[1.0, 2.5, 4.0]]), intensities=np.array([1.0, 3.0, 2.0]))
S2 = Spectrum(positions=np.array([[1.1, 2.4, 4.2]]), intensities=np.array([1.0, 2.5, 2.5]))

S1 = S1.normalized()
S2 = S2.normalized()

aligner = WNetAligner(
    empirical_spectrum=S1,
    theoretical_spectra=[S2],
    distance=DistanceMetric.L2,
    max_distance=0.5,
    trash_cost=0.9,
)
aligner.set_point([1])

print("Total cost:", aligner.total_cost())
flows = aligner.flows()[0]        # Flow(empirical_peak_idx, theoretical_peak_idx, flow)
emp_ids, theo_ids = aligner.consensus()  # greedy 1-to-1 pairs
```

For more complete examples see the `tutorials` directory.

## Tutorials

Tutorials showing how to apply `wnetalign` to NMR data can be found in `tutorials/nmr`, while an example of LC-MS alignment can be found in `tutorials/lcms`.

## Results from publication

To reproduce the results and figures from the paper *"WNetAlign: Fast and Accurate Spectra Alignment Using Truncated Wasserstein Distance and Network Simplex"* use the scripts and notebooks provided in the `publication/nmr` folder for NMR data and `publication/lcms` for LC-MS data.

#### 2D NMR data

The 2D 15N-1H HSQC spectra are available in `publication/nmr/2D/15N_HSQC_GB1_reduced` folder.

* `NMR_2D_original.ipynb` - plotting data distribution, chain alignment of the series of 15N-1H HSQC spectra measured in different temperatures
* `NMR_2D_simulated.ipynb` - simulating synthetic temperature series, validation of alignment procedure using standard and extended metrics, heatmaps
* `NMR_2D_seeds.ipynb` - performing alignments for the synthetic temperature series with different random seeds

#### 4D NMR data

The 4D CCNOESY spectrum of 2LX7 pdb structure can be found in the `publication/nmr/4D/2LX7`.

* `NMR_4D.ipynb` - simulating the replicate of the 4D 2LX7 CCNOESY spectrum and validating the alignment between the original spectrum and the replicate with standard and extended performance metrics

#### 7D NMR data

The hypothetical 7D NMR spectra can be found in `publication/nmr/7D/NMR_7D`.

* `NMR_7D.ipynb` - simulating the replicates of the 17 hypothetical 7D NMR spectra and validating the alignments between the original hypothetical spectra and the replicates with extended performance metrics

## Citation

If you use this software, please cite:

Król J, Bochenek M, Jopa S, Kazimierczuk K, Gambin A, Startek MP (2026).
WNetAlign: fast and accurate spectra alignment using truncated Wasserstein distance and network simplex.
*Briefings in Bioinformatics*, 27(3), bbag247.
https://doi.org/10.1093/bib/bbag247

```bibtex
@article{krol2026wnetalign,
  title   = {WNetAlign: fast and accurate spectra alignment using truncated Wasserstein distance and network simplex},
  author  = {Kr{\'o}l, Justyna and Bochenek, Maria and Jopa, Sylwia and Kazimierczuk, Krzysztof and Gambin, Anna and Startek, Micha{\l} Piotr},
  journal = {Briefings in Bioinformatics},
  volume  = {27},
  number  = {3},
  pages   = {bbag247},
  year    = {2026},
  doi     = {10.1093/bib/bbag247}
}
```

## License

This project is licensed under the MIT License.
