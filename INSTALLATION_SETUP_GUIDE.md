# RAPID Pipeline - Installation & Setup Guide

This document provides comprehensive setup instructions for the XRD RAPID Pipeline, covering environment setup, dependencies, and configuration requirements. Please note that this pipeline is designed for Windows environments only.
## Table of Contents

1. [Python 2.x Environment Setup](#python-2x-environment-setup)
2. [Installing Anaconda](#installing-anaconda)
3. [Creating a Python 2.7 Environment](#creating-a-python-27-environment)
4. [Activating and Deactivating Environments](#activating-and-deactivating-environments)
5. [Installing Required Packages](#installing-required-packages)
6. [Setting up PowerShell for Conda](#setting-up-powershell-for-conda)
7. [Adding Conda to PATH](#adding-conda-to-path)
8. [FullProf Suite Installation](#fullprof-suite-installation)
9. [PCR/PRF Application Settings](#pcrprf-application-settings)
10. [PyTorch Installation](#pytorch-installation)
11. [Additional Dependencies](#additional-dependencies)

---

## Python 2.x Environment Setup

### Installing Anaconda

1. Go to (https://www.anaconda.com/download) and download the installer for your operating system. You can skip the email registration by clicking the 'skip' button.

2. Follow the installation instructions for your operating system.

### Creating a Python 2.7 Environment

1. Open Anaconda Prompt (on Windows, search for it in the Start menu)

2. Create a virtual environment with the following command:
conda create -n py27 -c conda-forge python=2.7
> **Note**: The project uses the environment name 'py27', so it's recommended to use this name for compatibility.

3. Type 'y' when prompted to install the necessary packages for the environment.

### Activating and Deactivating Environments

1. To activate the virtual environment:
```
conda activate py27
```
You'll see the environment name in parentheses (py27) at the beginning of your command prompt.

3. To deactivate the current environment:
```
conda deactivate
```

### Useful Environment Management Commands
```
conda env list         # List all environments
conda remove --name py27 --all  # Delete the py27 environment
```

## Installing Required Packages

1. Activate the Python 2.7 environment:
conda activate py27

2. Install pip for Python 2.7:
- Download the get-pip.py script from: [https://bootstrap.pypa.io/pip/2.7/get-pip.py](https://bootstrap.pypa.io/pip/2.7/get-pip.py)
- Navigate to the directory containing the downloaded file:
  ```
  cd PATH\TO\FILE\DIRECTORY\
  python get-pip.py
  ```

3. Install PyQt4:
- Download the PyQt4 wheel file: `PyQt4-4.11.4-cp27-cp27m-win_amd64.whl` 
- Navigate to the directory containing the wheel file:
  ```
  cd PATH\TO\FILE\DIRECTORY\
  pip install PyQt4-4.11.4-cp27-cp27m-win_amd64.whl
  ```

4. Install basic packages using Anaconda Navigator:
- Open Anaconda Navigator
- Click on "Environments"
- Select your Python 2.7 environment (py27)
- Select "Not installed" in the dropdown
- Search for and install the following packages:
  - numpy
  - scipy
  - pandas

5. Install matplotlib and sobol_seq using command line:
conda install matplotlib
pip install sobol_seq

## Setting up PowerShell for Conda

1. Open Anaconda Prompt and run:
conda init powershell

2. Open PowerShell as administrator and run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
- Enter 'A' when prompted

3. Close both Anaconda Prompt and PowerShell, then reopen PowerShell

4. Verify the installation by typing:
conda
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

## PyTorch Installation

### Installing PyTorch with CUDA Support
If your Anaconda has Python version higher than 3.11, downgrade it first with `conda install python=3.11` as PyTorch doesn't support newer versions yet.
Install PyTorch + CUDA 11.8 from PyTorch & NVIDIA channels:
  ```
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Check GPU availability:
  ```
  python -c "import torch; print(torch.cuda.is_available())"
```

For specific versions with CUDA 11.7:
  ```
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

For more options, refer to PyTorch's previous versions page: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)

## Additional Dependencies
Install matplotlib, numpy, pandas:
conda install matplotlib numpy pandas
Install SHAP and Sobol sequence generator:
pip install shap sobol_seq

### Package Descriptions:
- **PyTorch**: Open-source deep learning framework for CPU/GPU
- **torchvision**: Computer vision utilities, pre-trained models
- **torchaudio**: Audio/speech processing
- **pytorch-cuda**: NVIDIA CUDA toolkit for GPU acceleration in PyTorch
- **matplotlib**: Visualization library
- **numpy**: Numerical computing library
- **pandas**: Data analysis library
- **shap**: For model interpretability 
- **sobol_seq**: For Sobol sequence generation used in parameter sampling
