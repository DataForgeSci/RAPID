<div align="center">
<pre>
████████████╗╗    ██████████╗╗  ████████████╗╗  ████╗╗████████████╗╗
████████████╗╗    ██████████╗╗  ████████████╗╗  ████╗╗████████████╗╗
  ████╔╔════████╗╗████╔╔════████╗╗████╔╔════████╗╗████║║████╔╔════████╗╗
  ████╔╔════████╗╗████╔╔════████╗╗████╔╔════████╗╗████║║████╔╔════████╗╗
  ████████████╔╔╝╝██████████████║║████████████╔╔╝╝████║║████║║    ████║║
  ████████████╔╔╝╝██████████████║║████████████╔╔╝╝████║║████║║    ████║║
  ████╔╔════████╗╗████╔╔════████║║████╔╔══════╝╝  ████║║████║║    ████║║
  ████╔╔════████╗╗████╔╔════████║║████╔╔══════╝╝  ████║║████║║    ████║║
  ████║║    ████║║████║║    ████║║████║║          ████║║████████████╔╔╝╝
  ████║║    ████║║████║║    ████║║████║║          ████║║████████████╔╔╝╝
╚╚══╝╝    ╚╚══╝╝╚╚══╝╝    ╚╚══╝╝╚╚══╝╝          ╚╚══╝╝╚╚══════════╝╝
</pre>

</div>

[![DOI](https://img.shields.io/badge/DOI-10.1107%2FS1600576726001494-blue)](https://doi.org/10.1107/S1600576726001494)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-2.7%20%7C%203.11-blue)

# `RAPID` — Rietveld Analysis Pipeline with Intelligent Deep-learning

RAPID automates Rietveld refinement of powder X-ray diffraction (XRD) data using convolutional neural networks. A CNN predicts crystallographic and profile parameters from a diffraction pattern in a single forward pass (~2 ms), then initializes automated FullProf refinement. The three-stage pipeline covers data augmentation, CNN training, and automated refinement.

Please cite the following [paper](https://doi.org/10.1107/S1600576726001494) if you use this code in your research:

```
@article{Mun:yr5164,
    author = "Mun, Suk Jin and Nam, Yoonsoo and Choi, Sungkyun",
    title = "{Automation of Rietveld refinement through machine learning}",
    journal = "Journal of Applied Crystallography",
    year = "2026",
    volume = "59",
    number = "2",
    pages = "",
    month = "Apr",
    doi = {10.1107/S1600576726001494},
    url = {https://doi.org/10.1107/S1600576726001494},
}
```

## Requirements

This code requires Python 2.7 (for AutoFP/FullProf refinement), Python 3.11 with PyTorch, NumPy, SciPy, and matplotlib (for CNN training and inference), and the [FullProf Suite](https://www.ill.eu/sites/fullprof/). Windows only.

## Installation

To install via conda, create the two environments using the provided environment files.

**Python 2.7 environment** (for AutoFP data augmentation and Rietveld refinement):
```bash
conda env create -f environment_py27.yml
conda activate rapid_py27
```

**Python 3.11 environment** (for CNN training and inference):
```bash
conda env create -f environment_py311.yml
conda activate rapid_py311
```

For detailed setup instructions, see **INSTALLATION_SETUP_GUIDE.md**.

## Usage

1. **Data Augmentation**: Place `.cif` and `.dat` files in `dat_vestacif_files/`, configure `inputs.txt`, run `data_augmentation.bat`
2. **CNN Training**: Configure `CNN/macro_inputs/ML_inputs_1.txt`, run `CNN/train_CNN_macro.bat`
3. **XRD Analysis**: Run `xrd_analysis.bat` to identify unknown materials and perform automated refinement

For detailed usage instructions, input file formats, and configuration options, refer to **RAPID_manual.pdf**.

## Acknowledgments

This project uses [AutoFP](https://github.com/xpclove/autofp) for automated Rietveld refinement and the [FullProf Suite](https://www.ill.eu/sites/fullprof/) as the refinement engine.
