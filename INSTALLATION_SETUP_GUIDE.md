# RAPID Pipeline - Installation & Setup Guide

This document provides comprehensive setup instructions for the XRD RAPID Pipeline, covering environment setup, dependencies, and configuration requirements. Please note that this pipeline is designed for Windows environments only.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Installation](#quick-installation)
3. [Manual Installation](#manual-installation)
4. [AutoFP Setup](#autofp-setup)
5. [Setting up PowerShell for Conda](#setting-up-powershell-for-conda)
6. [Adding Conda to PATH](#adding-conda-to-path)
7. [FullProf Suite Installation](#fullprof-suite-installation)
8. [PCR/PRF Application Settings](#pcrprf-application-settings)
9. [Package Descriptions](#package-descriptions)

---

## Prerequisites

### Installing Anaconda

1. Go to (https://www.anaconda.com/download) and download the installer for your operating system. You can skip the email registration by clicking the 'skip' button.

2. Follow the installation instructions for your operating system.

## Quick Installation

RAPID requires two conda environments. Create both using the provided environment files:

**Python 2.7 environment** (for AutoFP data augmentation and Rietveld refinement):
```
conda env create -f environment_py27.yml
conda activate rapid_py27
pip install PyQt4-4.11.4-cp27-cp27m-win_amd64.whl
```
> **Note**: The PyQt4 wheel file is included in the repository root. It must be installed manually as it is not available on PyPI.

**Python 3.11 environment** (for CNN training and inference):
```
conda env create -f environment_py311.yml
conda activate rapid_py311
```
> **Note**: The default CUDA version is 12.8. If your GPU requires a different version, edit `environment_py311.yml` and change `cu128` to `cu118`, `cu121`, or `cu124` before creating the environment. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for details.

**Verify GPU availability:**
```
conda activate rapid_py311
python -c "import torch; print(torch.cuda.is_available())"
```

**Extract AutoFP:**

Extract `autofp-1.3.5.zip` in the repository root so that the folder structure is `autofp-1.3.5/` at the top level. See [AutoFP Setup](#autofp-setup) for details.

### Useful Environment Management Commands
```
conda env list                              # List all environments
conda activate rapid_py27                   # Activate Python 2.7 environment
conda activate rapid_py311                  # Activate Python 3.11 environment
conda deactivate                            # Deactivate current environment
conda remove --name rapid_py27 --all        # Delete the rapid_py27 environment
conda remove --name rapid_py311 --all       # Delete the rapid_py311 environment
```

## Manual Installation

If you prefer to set up the environments manually instead of using the yml files:

### rapid_py27 (Python 2.7)

1. Open Anaconda Prompt (on Windows, search for it in the Start menu)

2. Create the environment:
```
conda create -n rapid_py27 -c conda-forge python=2.7
conda activate rapid_py27
```

3. Install packages:
```
conda install numpy scipy pandas matplotlib
pip install sobol_seq
```

4. Install PyQt4 from the included wheel file:
```
pip install PyQt4-4.11.4-cp27-cp27m-win_amd64.whl
```

> **Note**: pip for Python 2.7 may need to be installed manually. Download the get-pip.py script from: [https://bootstrap.pypa.io/pip/2.7/get-pip.py](https://bootstrap.pypa.io/pip/2.7/get-pip.py) and run `python get-pip.py`.

### rapid_py311 (Python 3.11)

1. Create the environment:
```
conda create -n rapid_py311 python=3.11
conda activate rapid_py311
```

2. Install PyTorch with CUDA support:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
> For other CUDA versions, replace `cu128` with `cu118`, `cu121`, or `cu124`. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

3. Install remaining dependencies:
```
pip install numpy scipy pandas matplotlib shap sobol_seq
```

4. Verify GPU availability:
```
python -c "import torch; print(torch.cuda.is_available())"
```

## AutoFP Setup

The repository includes `autofp-1.3.5.zip` which must be extracted before running the pipeline.

1. Locate `autofp-1.3.5.zip` in the repository root directory
2. Extract (unzip) it in place so that the folder `autofp-1.3.5/` exists at the top level of the repository
3. The resulting structure should be:
```
RAPID/
├── autofp-1.3.5/
│   ├── autofp_fs_unselect_GUI_suppressed.py
│   ├── autofp_fs_unselect_GUI_notsuppressed.py
│   └── ...
├── CNN/
├── scripts/
└── ...
```

> **Note**: Running scripts from outside the `autofp-1.3.5` directory can cause `.pyc` file errors.

## Setting up PowerShell for Conda

1. Open Anaconda Prompt and run:
```
conda init powershell
```

2. Open PowerShell as administrator and run:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```
Enter 'A' when prompted.

3. Close both Anaconda Prompt and PowerShell, then reopen PowerShell.

4. Verify the installation by typing:
```
conda
```
You should see the conda help information displayed.

## Adding Conda to PATH

This step is necessary for executing .bat files properly in Windows:

1. Open the Start Menu and search for "Environment Variables"
2. Click on "Edit the system environment variables"
3. In the System Properties window, click on "Environment Variables"
4. In the Environment Variables window, select the 'Path' variable in the 'System variables' section and click 'Edit'
5. Click "New" and add the path to the Scripts directory of your Anaconda installation
(typically `C:\Users\YourUsername\Anaconda3\Scripts`)
6. Click OK on all dialogs to apply the changes

## FullProf Suite Installation

Installing FullProf is essential for running the automated data augmentation pipeline as it relies heavily on PRF files.

1. Download FullProf Suite from: [https://www.ill.eu/sites/fullprof/php/downloads.html](https://www.ill.eu/sites/fullprof/php/downloads.html)

2. Follow the installation instructions, using the default settings
- It's recommended to install in the Local Disk (C:)
- The default installation directory should be `C:\FullProf_Suite`

## PCR/PRF Application Settings

### Setting PCR files to open with edpcr

1. Right-click any PCR file and select 'Properties'
2. Click on 'Change', then select 'Look for another app on this PC'
3. Navigate to the FullProf installation directory (`C:\FullProf_Suite`) and select `edpcr`

### Setting PRF files to open with winplotr

1. Right-click any PRF file and select 'Properties'
2. Click on 'Change', then select 'Look for another app on this PC'
3. Navigate to the FullProf installation directory (`C:\FullProf_Suite`) and select `winplotr`

After completing these settings, PCR files will open with edpcr and PRF files will open with winplotr.

## Package Descriptions

- **PyTorch**: Open-source deep learning framework for CPU/GPU
- **torchvision**: Computer vision utilities, pre-trained models
- **torchaudio**: Audio/speech processing
- **matplotlib**: Visualization library
- **numpy**: Numerical computing library
- **scipy**: Scientific computing library
- **pandas**: Data analysis library
- **shap**: For model interpretability
- **sobol_seq**: For Sobol sequence generation used in parameter sampling
- **PyQt4**: GUI toolkit required by AutoFP (Python 2.7 only)
