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
<img src="https://github.com/user-attachments/assets/1dd7eee3-f848-460e-94d5-90711bff84c0" alt="github_figure1" height="300">
<img src="https://github.com/user-attachments/assets/591445b0-bff5-4255-a791-003d44d7919d" alt="github_figure2" height="300">
</div>

# RAPID

RAPID (Rietveld Analysis Pipeline with Intelligent Deep-learning) automates Rietveld refinement of X-ray diffraction data using Convolutional Neural Networks. The pipeline generates synthetic XRD datasets through parameter sweeps, trains CNNs to predict crystallographic parameters from diffraction patterns, and ultimately enables phase identification with automated refinement. RAPID replaces manual parameter optimization with deep learning to reduce analysis time from hours to seconds while preserving crystallographic precision. As a version 1.0.0 release, though some components require further refinement for production use, the system demonstrates the potential of ML-driven structural characterization.

## System Requirements

- **Operating System**: Windows only
- **Installation**: Extract to a SHORT path (e.g., `C:\username` or Desktop) to avoid Windows path length errors
  - ⚠️ DO NOT extract to deeply nested folders

This repository provides scripts and batch files for **Rietveld Refinement**, **Data Augmentation**, and **Machine Learning** (CNN-based) analysis of XRD data. The workflow is split into three main parts:

1. **Data Augmentation Pipeline**: Handles all the steps of generating or modifying XRD data (from `.cif` to `.pcr` to `.dat`) and performing parameter sweeps/refinements, culminating in a large dataset suitable for machine learning.  
2. **`CNN` Folder**: Contains the scripts and batch processes for training a CNN on the generated XRD datasets (`*_simulated_data_row_param.dat`), refining new experimental data, and orchestrating multiple training sessions using a macro approach.
3. **`single_phase_identification` Folder**: Provides tools for identifying unknown materials from XRD patterns, predicting parameters, and performing automated Rietveld refinement using the XRD Analysis Tool.



## Table of Contents

1. [Differentiating Data Augmentation vs. CNN vs. XRD Analysis](#1-differentiating-data-augmentation-vs-cnn-vs-xrd-analysis)  
2. [Data Augmentation Pipeline](#2-data-augmentation-pipeline)  
   - [Manual for Running Batch Files](#manual-for-running-batch-files)  
   - [To Clean Up the Data Directory](#to-clean-up-the-data-directory)  
   - [Reference PCR Format Scripts](#reference-pcr-format-scripts)  
   - [PCR Format Fix Scripts](#pcr-format-fix-scripts)  
3. [CNN Pipeline](#3-cnn-pipeline)  
   - [Notable Folders & Files in `CNN/`](#notable-folders--files-in-cnn)  
   - [Configuring ML_inputs.txt for CNN Training](#configuring-ml_inputstxt-for-cnn-training)
   - [Example: Running the CNN Macro](#example-running-the-cnn-macro)  
   - [Data Files for CNN](#data-files-for-cnn)  
   - [Parameter Scaling Options](#parameter-scaling-options)
   - [Correlation Analysis](#correlation-analysis)
   - [Managing CNN Models](#managing-cnn-models)
   - [CNN Output Files](#cnn-output-files)
4. [End-to-End Processing with macro_DA_ML.bat](#4-end-to-end-processing-with-macro_da_ml-bat)
   - [Overview and Purpose](#overview-and-purpose)
   - [Processing Modes](#processing-modes)
   - [Using the Multi-Dataset Processing Mode](#using-the-multi-dataset-processing-mode)
   - [Using the Fine-Tuning Mode](#using-the-fine-tuning-mode)
   - [Workflow and Output Organization](#workflow-and-output-organization)
5. [XRD Analysis Pipeline](#5-xrd-analysis-pipeline)
   - [Setting Up the Material Database](#setting-up-the-material-database)
   - [Analyzing Unknown Materials](#analyzing-unknown-materials)
   - [Components of the XRD Analysis System](#components-of-the-xrd-analysis-system)
6. [Directory Structure Examples](#6-directory-structure-examples)  
   - [Directory Structure - Data Augmentation](#directory-structuredata-augmentation)  
   - [Directory Structure - CNN](#directory-structurecnn)  
   - [Directory Structure - single_phase_identification](#directory-structuresingle_phase_identification)
7. [Scripts in the `scripts` Folder](#7-scripts-in-the-scripts-folder)  
8. [Output Figure Organization & AutoFP](#8-output-figure-organization--autofp)  
   - [AutoFP-1.3.5 Folder](#autofp-135-folder)  
9. [Contents of the `data` Folder](#9-contents-of-the-data-folder)  
10. [PyTorch Installation & Dependencies](#10-pytorch-installation--dependencies)
11. [License](#11-License)
12. [Credits](#12-Credits)




## 1. Differentiating Data Augmentation vs. CNN vs. XRD Analysis

- **Data Augmentation Pipeline**  
  - Located in the root directory with `scripts/` subfolder
  - Contains scripts (`step1` through `step6`) for **data generation** and parameter sweeps using Rietveld refinements  
  - Produces large `.dat` or `.bin` files later used in CNN training
  - Includes reference PCR format handlers and automatic fixes for specific materials

- **`CNN`**  
  - Focuses on **machine learning**: CNN training, inference, refining experimental data  
  - Contains scripts like `train_xrd.py`, `utils.py`, `train_CNN_macro.bat` for orchestrating multiple training sessions from `macro_inputs/`  
  - Includes correlation analysis for model interpretation
  - Provides comprehensive Rietveld refinement workflow using predicted parameters

- **`single_phase_identification`**
  - Provides end-to-end **material identification** from unknown XRD patterns through the XRD Analysis Tool
  - Contains a database of material fingerprints and trained models
  - Features automated Rietveld refinement and comprehensive reports
  - Designed for analyzing real-world experimental XRD data



## 2. Data Augmentation Pipeline

### Manual for Running Batch Files

1. **Prepare Your Files**  
   - Place your `_vesta.cif` and `.dat` files into the `dat_vestacif_files` folder
   - Name them appropriately, e.g. `CeO2_vesta.cif`, `CeO2.dat`

2. **Input Parameters in `inputs.txt`**  
   - **Lines 1-5**: Basic file names and subfolder configuration
   - **Lines 6-13**: Parameter ranges for zero, background, and lattice parameters
   - **Line 14**: Two-theta sampling range (Y/N with format `N;initial,step,final`)
   - **Lines 15-22**: Processing/output configs and CNN training option
   - **Lines 23-27**: Percentage ranges for Biso, scale factor, and U/V/W parameters
   - **Lines 29-33**: Distribution classification and re-run options
   - **Line 34**: Reference PCR file to extract parameters from
   - **Lines 35-37**: Zoomed inset configuration for classification plots
   - **Lines 38-39**: Parameters for copying from reference PCR
   - **Lines 40-44**: Specific folder and plot generation options
   - **Line 46**: Atom data format in PCR file (4-line or 2-line)

_For details, refer to **RAPID_manual.pdf**._


3. **Run `data_augmentation.bat`**  
   - Executes the complete pipeline: converting CIF→PCR, creating modified parameter files, running AutoFP refinements, and generating combined data files
   - Double-click `data_augmentation.bat` to start

### Reference PCR Format Scripts

The system includes several specialized scripts to handle different crystal structures and parameter formats:

- **`reference_pcr_CeO2.py`**: Handles CeO2 cubic structure parameters
- **`reference_pcr_pbso4.py`**: Handles PbSO4 orthorhombic structure parameters
- **`reference_pcr_tbbaco.py`**: Handles TbBaCo structure parameters

Each reference script can:
- Extract critical parameters from an existing PCR file
- Apply those parameters to a new PCR file
- Fix atom label spacing and format for proper parsing
- Handle lattice parameters specific to the crystal structure
- Apply extended parameters (asymmetry, scan range) when requested

To use these references, specify the reference PCR filename in line 34 of `inputs.txt`. Use lines 38-39 to control copying additional parameters.

### PCR Format Fix Scripts

The system also includes specialized fix scripts for specific formats:

- **`fix_ceo2_pcr.py`**: Applies comprehensive transformations for CeO2.pcr format:
  - Applies exact formatting from the solution PCR

These fix scripts are automatically applied when the corresponding reference PCR is selected in `inputs.txt`.

### To Clean Up the Data Directory

- **Run `refresh_datadirectory.bat`**  
  1. Cleans up the `data` directory by removing every subfolder except `backup`
  2. Double-click `refresh_datadirectory.bat` to refresh the data directory



## 3. CNN Pipeline

The CNN pipeline focuses on training convolutional neural networks for XRD pattern analysis and parameter prediction, with enhanced features for model management, visualization, and automated refinement.

### Notable Folders & Files in `CNN`

1. **`macro_inputs/`**  
   - Stores configuration files for CNN training (`ML_inputs_1.txt`, `ML_inputs_2.txt`, etc.)  
   - Defines training parameters, data sources, refinement settings, and visualization options

2. **`train_CNN_macro.bat`**  
   - Orchestrates training by running multiple configurations sequentially
   - Automatically processes all configs in `macro_inputs/` folder
   - Handles environment switching for Rietveld refinement steps

3. **`clean_ml_inputs.bat` & `create_ml_inputs.bat`**
   - Utilities to manage CNN training configurations
   - `clean_ml_inputs.bat`: Removes all configurations except template file
   - `create_ml_inputs.bat`: Creates multiple training configs from a template

4. **`select_best_models.py`**
   - Analyzes training results to identify the best-performing models
   - Evaluates Rwp values from refinement results
   - Keeps top-performing models and deletes the rest

### Configuring ML_inputs.txt for CNN Training

The ML_inputs.txt file controls all aspects of CNN training and refinement. Here's a detailed breakdown of each line:

- **Line 1**: Train from scratch?
  - Format: "y;model_name" or "n"
  - Example: `y;model_CeO2_1` to create a new model called "model_CeO2_1"
  - If "n", will try to load an existing model (specified in line 2)

- **Line 2**: Existing model to load
  - Format: Model folder name or "n"
  - Example: `model_CeO2_20250510` to load this specific model
  - Use "n" if training from scratch (line 1 is "y")

- **Line 3**: Dataset configuration 
  - Format: "N;dataset_folder" or "Y;folder1,folder2,folder3"
  - Example: `N;CeO2_20250515_102736` for training on a single dataset
  - Example: `Y;CeO2_10K,CeO2_15K` for progressive training on multiple datasets

- **Line 4**: Experimental files for refinement
  - Format: "filename.dat" or comma-separated files or "n"
  - Example: `CeO2.dat` to refine this experimental file
  - Example: `n` to skip refinement

- **Line 5**: Do Rietveld Refinement?
  - Format: "y" or "n"
  - Controls whether to run Rietveld refinement step after training

- **Line 6**: Run feature importance test?
  - Format: "y" or "n"
  - Enables correlation matrix visualization between features

- **Line 7**: Omit background parameter?
  - Format: "y" or "n"
  - Removes background from training/prediction for better stability

- **Line 8**: Bragg reflection peak number to zoom
  - Format: Number or "default"
  - Selects which Bragg peak to focus on in refinement visualizations

- **Line 9**: Zoom width
  - Format: Number (in degrees 2θ) or "auto"
  - Controls the width of zoomed regions in refinement plots

- **Line 10**: Use digit-based parameter scaling?
  - Format: "y" or "n"
  - Normalizes parameters to similar digit places

- **Line 11**: Use adaptive scaling?
  - Format: "y" or "n"
  - Scales parameters relative to reference values

- **Line 12**: DAT file format structure
  - Format: "CeO2.dat", "pbso4.dat", or "tbbaco.dat"
  - Specifies the format structure for parsing experimental files

_For details, refer to **RAPID_manual.pdf**._


### Example: Running the CNN Macro

1. **Prepare configuration files** in `macro_inputs/` folder
   - Use `ML_inputs_1.txt` as a template
   - Optionally run `create_ml_inputs.bat` to generate multiple configs

2. **Configure training parameters**
   - Edit ML_inputs files to set desired options following the format above

3. **Run `train_CNN_macro.bat`**
   - Processes all configurations sequentially
   - Handles environment switching for refinement
   - Creates output in `saved_models/` directory

### Data Files for CNN

The CNN training pipeline is designed to work with the datasets generated by the Data Augmentation Pipeline. Key data files include:

- `*_simulated_data_row_param.dat`: Primary training file with intensity and parameter data
- `*_simulated_data_column_for_plot.dat`: Formatted for visualization
- `*_simulated_data_2theta_param_info.dat`: Metadata about parameters and sampling

### Parameter Scaling Options

The CNN training pipeline offers three parameter scaling strategies configurable via ML_inputs.txt. It is recommended to use digit-based scaling:

1. **Digit-based scaling** (line 10: "y")
   - Normalizes each parameter's first significant digit to ones place
   - Ensures consistent magnitude ranges across parameters
   - Example: 0.00034 → 340, 5430 → 5.43
   - Recommended for most circumstances

2. **Adaptive scaling** (line 11: "y")
   - Uses reference values from first data row
   - Creates customized scaling factors for each parameter
   - Preserves relative importance within parameter categories

3. **Default scaling** (fallback)
   - Pre-configured scaling factors optimized by parameter type:
     - Zero: 100
     - Background: 0.1
     - Lattice: 1.0
     - Biso: 10.0
     - Scale: 100000
     - U/V/W: 1000

These scaling options significantly improve training stability across diverse crystal structures and parameter distributions.

### Correlation Analysis

The feature importance analysis provides tools to understand which parameters are most important for model predictions:

- **Permutation Importance**
  - Measures increase in prediction error when a parameter is permuted
  - Identifies which parameters have strongest impact on model output
  
- **Correlation Analysis**
  - Calculates correlation coefficients between all feature pairs
  - Creates correlation matrix heatmap visualization
  - Identifies strongly correlated parameters
  
- **Visualization Types**
  - Bar charts of overall importance
  - Correlation heatmaps showing parameter relationships
  - Beeswarm plots showing value distribution and impact
  - Dependence plots showing how parameter values affect predictions*

*⚠️ **Note:** Although SHAP figures are generated, they are considered deprecated as they do not contain any physical significance to the XRD analysis. For now, these visualizations should not be used for interpretations.

To run correlation analysis, set line 6 to "y" in your ML_inputs.txt file.

### Managing CNN Models

The system includes utilities for managing multiple CNN models:

- **`create_ml_inputs.bat`**: Creates multiple training configurations based on a template
- **`clean_ml_inputs.bat`**: Cleans up configuration files to maintain organization
- **`select_best_models.py`**: Analyzes refinement results to identify top-performing models

Model selection is based on Rietveld refinement quality metrics (Rwp values), allowing automatic identification of the most effective CNN models for specific materials.

### CNN Output Files

The CNN pipeline generates numerous output files organized in a structured directory:

1. **Model Files**
   - `model_name.pth`: Trained PyTorch model file
   - `train_data_log.dat`: Training log with configuration details

2. **Training Results**
   - **Metrics Visualizations:**
     - `cnn_training_loss_vs_epoch.png`: Loss curves for training/validation 
     - `cnn_training_mae_vs_epoch.png`: Mean Absolute Error progression
     - `cnn_training_rmse_vs_epoch.png`: Root Mean Squared Error progression
   - **Training Logs:**
     - `cnn_training_results_[timestamp].dat`: Detailed epoch-by-epoch training log
     - `parameter_info_[timestamp].dat`: Parameter structure information

3. **Correlation Analysis**
   - **Visualizations:**
     - `parameter_correlation_[timestamp].png`: Parameter correlation heatmap
   - **Data Files:**
     - `parameter_correlation_data_[timestamp].dat`: Correlation matrices

4. **Refinement Results**
   - **PCR Files:**
     - `cnn_refined.pcr`: PCR file with CNN-predicted parameters
   - **Refinement Output:**
     - `.prf` files: Pattern calculation results
     - `.out` files: Refinement output logs
   - **Analysis Reports:**
     - `analysis_report.dat`: Comprehensive refinement quality metrics
     - `refinement_comparison.png`: Plot comparing CNN vs. solution refinement
     - `cnn_refinement.png`: CNN-only refinement visualization

5. **Experimental Data Files**
   - `*_refined_parameters.dat`: CNN predicted parameters for experimental data
   - `*_experimental_data_column_for_plot.dat`: Processed experimental data
   - `*_readable_experimental_data.dat`: Human-readable data format



## 4. End-to-End Processing with macro_DA_ML.bat

### Overview and Purpose

The `macro_DA_ML.bat` file provides an end-to-end automated pipeline that combines the data augmentation and CNN training processes into a single workflow. This powerful tool streamlines the entire process from generating synthetic XRD datasets to training CNN models and performing refinements.

Key features:
- Automatically runs data augmentation to generate datasets
- Moves generated datasets to the CNN/data/train_data directory
- Trains CNN models on each dataset
- Organizes the models and results with proper folder structure
- Selects the best-performing models based on refinement quality metrics

### Processing Modes

The `macro_DA_ML.bat` script offers two distinct processing modes:

1. **Multi-Dataset Processing Mode**
   - Processes multiple datasets defined in input files
   - Suitable for batch processing several materials or parameter configurations
   - Uses input files in the `macro_inputs` folder (e.g., `inputs_1.txt`, `inputs_2.txt`)

2. **Fine-Tuning Mode**
   - Focuses on a single compound with specified parameter adjustments
   - Allows precise control over which parameters to fine-tune (zero shift, lattice parameters, Biso, scale factor, U/V/W)
   - Generates multiple fine-tuned variations of the input configuration
   - Ideal for optimizing model performance for a specific material

### Using the Multi-Dataset Processing Mode

1. **Prepare Input Files**
   - Create `inputs_1.txt` (and optionally more numbered files) in the `macro_inputs` folder
   - Follow the standard format described in the Data Augmentation Pipeline section

2. **Run `macro_DA_ML.bat` and Select Option [1]**
   - The script will automatically:
     - Process each input file to generate datasets
     - Move them to the CNN training folder
     - Configure and run CNN training for each dataset
     - Organize and evaluate the resulting models

3. **Output Organization**
   - Datasets are generated in the `data/` directory with timestamps
   - CNN models are saved in `CNN/saved_models/` with dataset-specific subfolders
   - Best-performing models are identified and preserved

### Using the Fine-Tuning Mode

1. **Run `macro_DA_ML.bat` and Select Option [2]**

2. **Select Parameters to Fine-Tune**
   - Zero shift: Controls the 2θ offset correction
   - Lattice parameters: Fine-tunes crystal structure dimensions
   - Biso parameter: Adjusts atomic displacement factors
   - Scale factor: Modifies overall intensity scaling
   - U/V/W parameters: Tunes peak shape and width

3. **Specify Fine-Tuning Details**
   - Number of iterations: Controls how many variations to generate
   - Parameter shifts: Define the magnitude of adjustments for each parameter
     - For zero parameter: Use absolute shift values
     - For other parameters: Use percentage values
   - Note: when fine tuning, classification re-runs option will be automatically toggled off.

4. **Automated Generation and Training**
   - The system automatically creates fine-tuned input variations
   - Processes each variation to generate specialized datasets
   - Trains models on these datasets with appropriate configurations
   - Evaluates and identifies optimal parameter combinations

### Workflow and Output Organization

The workflow follows three main phases:

1. **Phase 1: Data Augmentation**
   - Runs the selected input files through the data augmentation pipeline
   - Generates synthetic XRD patterns with parameter variations
   - Creates combined dataset files ready for CNN training

2. **Phase 2: Dataset Organization**
   - Moves generated datasets to the CNN/data/train_data directory
   - Maintains proper file organization and naming conventions
   - Creates a list of datasets to process

3. **Phase 3: CNN Training and Refinement**
   - For each dataset:
     - Configures ML_inputs files with appropriate settings
     - Runs CNN training with optimal hyperparameters
     - Performs Rietveld refinement with the trained model
     - Evaluates model quality based on refinement metrics
     - Organizes models in a structured directory hierarchy

The final output includes:
- Trained CNN models organized by dataset
- Refinement results with quality metrics
- Visual comparisons between CNN predictions and actual refinements
- Selected best-performing models for each material/configuration



## 5. XRD Analysis Pipeline

The XRD Analysis Tool provides an enhanced system for identifying unknown materials from XRD patterns, predicting parameters, and performing automated Rietveld refinement.

### Setting Up the Material Database

1. **Run `xrd_analysis.bat` and select option [1]**
   - Creates/updates the single phase material database from saved models
   - Generates XRD pattern fingerprints with peak shape analysis
   - Locates PCR templates for refinement
   - Results are stored in `single_phase_identification/database/`

### Analyzing Unknown Materials

1. **Prepare Your Files**
   - Place unknown XRD pattern files (`.dat`) in the `single_phase_identification/unknown_materials/` folder

2. **Run `xrd_analysis.bat` and select option [2]**
   - Lists available `.dat` files for analysis
   - Identifies the most likely material match
   - Predicts parameters using the appropriate CNN model
   - Performs automatic Rietveld refinement with advanced peak shape analysis
   - Generates comprehensive reports with results in an identified materials folder

### Components of the XRD Analysis System

- **Material Database**: Contains fingerprints with peak shape metrics and model metadata organized by material
- **Fingerprint Matching**: Enhanced algorithm using position, intensity correlation, and peak shape (FWHM and asymmetry)
- **Parameter Prediction**: Uses appropriate CNN model for the identified material with proper scaling
- **Automated Refinement Pipeline**: Three-step process that creates PCR files, runs AutoFP refinement, and analyzes results
- **Advanced Visualization**: Generates enhanced plots with Bragg peak markers, difference curves and zoomed insets
- **Comprehensive Reports**: Detailed identification and refinement results with quality metrics (R-factors)
- **Support for All Crystal Systems**: Handles cubic, tetragonal, orthorhombic, monoclinic, triclinic, hexagonal, and trigonal structures



## 6. Directory Structure Examples

### Directory Structure - Data Augmentation
```bash
# Before running data_augmentation.bat (managing CeO2 data):
root_directory/
├── inputs.txt                          # Parameter configuration file
├── refresh_datadirectory.bat           # Utility to clean data directory
├── data_augmentation.bat               # Main batch file for the pipeline
├── data/
│   └── backup/                         # Backup folder preserved during cleanup
├── dat_vestacif_files/
│   ├── CeO2.dat                        # XRD pattern data file
│   └── CeO2_vesta.cif                  # Crystal structure file
├── scripts/
│   ├── step1_cif2pcr.py                # Convert CIF → PCR
│   ├── step2_pcrfolders_modifiedparameters.py  # Create parameter variations
│   ├── step3_runautofp_fixall.py       # Run AutoFP refinements
│   ├── step3p5_distribution_classification.py  # Classify results
│   ├── step4_overplots_3datfiles.py    # Generate combined data files
│   ├── refresh_datadirectory.py        # Directory cleanup script
│   ├── algorithm_uniformsampling.py    # Parameter sampling utilities
│   ├── reference_pcr_CeO2.py           # CeO2 reference handler
│   ├── reference_pcr_pbso4.py          # PbSO4 reference handler
│   ├── reference_pcr_tbbaco.py         # TbBaCo reference handler
│   └── fix_ceo2_pcr.py                 # CeO2 format fix script
└── autofp-1.3.5/
    ├── autofp_fs_unselect_GUI_suppressed.py    # Non-interactive AutoFP
    ├── autofp_fs_unselect_GUI_notsuppressed.py # Interactive AutoFP     
    └── other python modules for AutoFP
```
```bash
# After running data_augmentation.bat (CeO2 data):
root_directory/
├── inputs.txt
├── refresh_datadirectory.bat
├── data_augmentation.bat
├── convert_bin_to_dat.bat        
├── data/
│   ├── backup/
│   └── CeO2_YYYYMMDD_HHMMSS/            # Timestamp-based subfolder
│       ├── inputs_YYYYMMDD_HHMMSS.txt   # Copy of inputs used
│       ├── classification_report.dat    # Classification results and metrics
│       ├── source_files/                # Original input files
│       │   ├── CeO2.dat
│       │   ├── CeO2.pcr
│       │   └── CeO2_vesta.cif
│       ├── output_figures/              # Various visualization outputs
│       │   ├── rietveld_refinement_plots/
│       │   │   └── sample*_rietveld_fit.png
│       │   ├── ycal_original_versus_interpolated_plots/
│       │   │   └── comparison_sample*_2thetaYcal_plot.png
│       │   ├── classification_profile_plots/
│       │   │   └── classification_profile_overplot_run*_zoomed.png
│       │   └── parameter_shift_distribution_plots/
│       │       ├── selectedparametershifts_uniformdistribution.png
│       │       └── selectedparametershifts_randomdistribution.png
│       ├── generated_samples/           # Individual refinement results
│       │   ├── sample1/
│       │   │   ├── sample1.prf
│       │   │   ├── sample1.pcr
│       │   │   ├── sample1.out
│       │   │   └── comparison_sample1_2thetaYcal_plot.png
│       │   ├── sample2/
│       │   │   └── ... 
│       │   └── ...
│       ├── CeO2_simulated_data_row_param.dat         # Combined data for ML
│       ├── CeO2_simulated_data_column_for_plot.dat   # Data for plotting
│       └── CeO2_simulated_data_2theta_param_info.dat # Parameter information
├── dat_vestacif_files/
│   ├── CeO2.dat
│   └── CeO2_vesta.cif
├── scripts/
│   └── ... (same as before)
└── autofp-1.3.5/
    └── ... (same as before)
```
```bash
### Directory Structure - CNN

CNN/
├── data/
│   └── ...  # (Optional) If you store CNN-specific datasets/logs here
├── macro_inputs/
│   ├── ML_inputs_1.txt
│   ├── ML_inputs_2.txt
│   └── ...
├── saved_models/
│   ├── model_CeO2_1/
│   │   ├── model_CeO2_1.pth
│   │   ├── train_data_log.dat
│   │   ├── refinement_result/
│   │   │   └── CeO2_20250515_102736/
│   │   │       └── CeO2/
│   │   │           ├── CeO2_refined_parameters.dat
│   │   │           ├── CeO2_experimental_data_column_for_plot.dat
│   │   │           └── Rietveld_Refinement/
│   │   │               ├── CNN_ML_refinement/
│   │   │               │   ├── cnn_refined.pcr
│   │   │               │   ├── CeO2.dat
│   │   │               │   ├── CeO2.prf
│   │   │               │   └── CeO2.out
│   │   │               ├── solution_refinement/
│   │   │               │   └── ... 
│   │   │               └── output_analysis/
│   │   │                   ├── refinement_comparison.png
│   │   │                   ├── cnn_refinement.png
│   │   │                   └── analysis_report.dat
│   │   ├── training_result/
│   │   │   └── CeO2_20250515_102736/
│   │   │       ├── cnn_training_results_20250515_140523.dat
│   │   │       ├── cnn_training_loss_vs_epoch.png
│   │   │       ├── cnn_training_mae_vs_epoch.png
│   │   │       ├── cnn_training_rmse_vs_epoch.png
│   │   │       └── parameter_info_20250515_140523.dat
│   │   └── correlation_analysis/
│   │       └── parameter_correlation_20250515_140523.png
│   └── ...
├── scripts/
│   ├── train_xrd.py
│   ├── utils.py
│   ├── feature_importance.py
│   ├── Rietveld_Refinement_step1.py
│   ├── Rietveld_Refinement_step2.py
│   ├── Rietveld_Refinement_step3.py
│   └── select_best_models.py
├── clean_ml_inputs.bat
├── create_ml_inputs.bat
└── train_CNN_macro.bat
```
```bash
### Directory Structure - single_phase_identification

single_phase_identification/
├── database/
│   ├── material_catalog.json            # Catalog of materials with model info
│   ├── fingerprints/                    # XRD fingerprints for pattern matching
│   │   ├── CeO2_fingerprint.json
│   │   ├── pbso4_fingerprint.json
│   │   └── visualizations/              # Enhanced plots of fingerprinted patterns
│   │       └── CeO2_fingerprint.png
├── identified_materials/                # Results organized by material and timestamp
│   ├── unknown_CeO2_20250515_140523/x
│   │   ├── identification_report.dat    # Comprehensive identification report
│   │   ├── pattern_comparison.png       # Visual comparison with fingerprint
│   │   ├── parameter_refinement/        # CNN prediction results
│   │   │   ├── CeO2_refined_parameters.dat
│   │   │   └── CeO2_experimental_data_column_for_plot.dat
│   │   ├── Rietveld_Refinement/         # Organized refinement results
│   │   │   ├── CNN_ML_refinement/       # AutoFP refinement with CNN parameters
│   │   │   │   ├── cnn_refined.pcr
│   │   │   │   ├── CeO2.dat
│   │   │   │   ├── CeO2.prf
│   │   │   │   └── CeO2.out
│   │   │   └── analysis_output/         # Result analysis and plots
│   │   │       ├── CeO2_refinement.png
│   │   │       └── CeO2_refinement_report.dat
│   │   └── source_files/                # Original input files
│   │       └── unknown.dat
├── unknown_materials/                   # Place to put unknown XRD patterns
│   └── unknown.dat
│
scripts/                             # Python scripts for the pipeline
│   ├── database_manager.py
│   ├── material_identifier.py
│   ├── Rietveld_Refinement_step1_MI.py
│   ├── Rietveld_Refinement_step2_MI.py
│   └── Rietveld_Refinement_step3_MI.py
xrd_analysis.bat                     # Main batch file for XRD analysis

```
## 7. Scripts in the `scripts` Folder

1. **step1_cif2pcr.py**  
   - **Purpose**: Convert CIF → PCR.  
   - **Order**: First.  
   - **Output**: PCR file, copying `.dat` & `.cif` to the new subfolder.  
   - **Details**: Calculates Biso from Uani (`B_iso = U_ani * 8π²`), sets atomic params, etc.

2. **step2_pcrfolders_modifiedparameters.py**  
   - **Purpose**: Creates multiple PCR files/folders with random param modifications (zero, background, lattice).  
   - **Order**: Second.  
   - **Output**: Subfolders, e.g. `sample1`, `sample2`, etc.
   - **Details**: Handles multiple crystal structures and atom-specific Biso variations.

3. **step3_runautofp_fixall.py**  
   - **Purpose**: Runs AutoFP refinement on each PCR.  
   - **Order**: Third.  
   - **Output**: PRF files in each subfolder.  
   - **Details**: Chooses GUI-suppressed/nonsuppressed based on `inputs.txt`.

4. **step3p5_distribution_classification.py**  
   - **Purpose**: Classify refinement results as CLOSE, BOUNDARY, or INVALID based on Rwp.
   - **Order**: Between steps 3 and 4.
   - **Output**: `classification_report.dat` and overplot of profiles with zoomed inset.
   - **Details**: Implements 80:20 ratio re-run logic to adaptively adjust parameter ranges.

5. **step4_overplots_3datfiles.py**  
   - **Purpose**: Processes PRF → combined `.dat` files, comparison plots.  
   - **Order**: Fourth.  
   - **Output**:  
     - `*_simulated_data_row_param.dat`  
     - `*_simulated_data_column_for_plot.dat`  
     - `*_simulated_data_2theta_param_info.dat`  
     - Comparison plots in each subfolder.

6. **Reference PCR scripts** (`reference_pcr_*.py`)
   - **Purpose**: Extract and apply parameters from reference PCR files for specific materials.
   - **Order**: Used by step1 when specified in inputs.txt.
   - **Details**: Support different crystal structures with proper parameter handling.

7. **PCR Format Fix scripts** (`fix_*.py`)
   - **Purpose**: Apply specific transformations to ensure exact format matching with solution PCR files.
   - **Order**: Used after reference PCR processing.
   - **Details**: Complete format overhaul for specific material types.



## 8. Output Figure Organization & AutoFP

### Rietveld Refinement & Pattern Comparison

- **Rietveld Refinement Plots** (`output_figures/rietveld_refinement_plots/`)  
  - `*_rietveld_fit.png`: Observed vs. calculated patterns, difference curves.

- **Pattern Comparison Plots** (`output_figures/ycal_original_versus_interpolated_plots/`)  
  - Compare original vs. interpolated patterns with dashed lines, legends, axis labels.

- **Classification Profile Plots** (`output_figures/classification_profile_plots/`)
  - Overplots of profiles classified as CLOSE, BOUNDARY, INVALID.
  - Includes zoomed inset focused on the first Bragg reflection.

- **Parameter Distribution Plots** (`output_figures/parameter_shift_distribution_plots/`)
  - 3D visualizations of uniform and random parameter distributions.
  - Shows the sampling strategy for zero, background, and lattice parameters.

### AutoFP-1.3.5 Folder

- **autofp_fs_unselect_GUI_suppressed.py**  
  - GUI-suppressed AutoFP for faster batch refinement.  
  - Clears parameters, checks log for "1 ok!".

- **autofp_fs_unselect_GUI_notsuppressed.py**  
  - Interactive AutoFP with full GUI.  

> **Note**: Running scripts outside `autofp-1.3.5` can cause `.pyc` file errors.



## 9. Contents of the `data` Folder

- **`backup/`**  
  - Stores backups before new batch runs; never deleted by `refresh_datadirectory.bat`.

- **Timestamped Subfolders**  
  - Named like `CeO2_20250515_140523/`.  
  - Contains `generated_samples/`, `output_figures/`, `source_files/`, and `.dat` files (`_simulated_data_*`).
  - Contains `classification_report.dat` with detailed Rwp analysis.

Inside each timestamped subfolder:

- **`generated_samples/`**  
  - `sample1/`, `sample2/` subfolders with `.prf`, `.pcr`, `.out`, comparison plots.

- **`output_figures/`**  
  - Distribution plots, R-factor interpolation data (`1D`, `2D`), original vs. interpolated patterns.
  - Classification profile plots with zoomed insets.
  - Parameter shift distribution plots showing the sampling strategy.

- **`source_files/`**  
  - The original `.dat`, `.cif`, and `.pcr`.

- **Main `.dat` files**  
  - E.g. `CeO2_simulated_data_row_param.dat`, `CeO2_simulated_data_column_for_plot.dat`.

## 10. PyTorch Installation & Dependencies
**Quick install (CUDA).**

- Use **Python 3.11**:  
  `conda install python=3.11`
- **PyTorch + CUDA 11.8**:  
  `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
- **Check GPU**:  
  `python -c "import torch; print(torch.cuda.is_available())"`
- **Alt (CUDA 11.7 / specific versions)**:  
  `conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia`

**Additional dependencies**  
`conda install matplotlib numpy pandas`  
`pip install shap sobol_seq`

_For details, refer to **Installation Setup Guide.md**._

  
## 11. License
RAPID is open‑source software distributed under the GNU General Public License version 3 (GPLv3) licence. It is free to use, modify and redistribute under the terms of the GPLv3.

 
## 12. Credits
* Suk Jin Mun (SukjinMun) - Main author (1.0.0)
* Yoonsoo Nam (1.0.0)
* Sungkyun Choi (1.0.0)

### Acknowledgments
This project makes use of [AutoFP](https://github.com/xpclove/autofp), an open-source tool for automated Rietveld refinement, and the [FullProf Suite](https://www.ill.eu/sites/fullprof/), which serves as the Rietveld refinement engine interfaced by RAPID. We thank the authors for providing their implementation, which inspired and supported our work.

