#!/usr/bin/env python
"""
Material configurations for Rwp Uncertainty Quantification
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.dirname(SCRIPT_DIR)

MATERIALS = {
    'CeO2': {
        'model_folder': 'model_CeO2_100epochs_digit_zero0p0075_lp0p005_May13_2025',
        'model_file': 'model_CeO2_111.pth',
        'train_data_folder': 'CeO2_20250511_111201',
        'train_data_file': 'CeO2_simulated_data_row_param.dat',
        'template_pcr': 'source_files/CeO2.pcr',
        'dat_filename': 'CeO2_simulated.dat',
        'two_theta_start': 25.00,
        'two_theta_step': 0.20,
        'two_theta_end': 100.00,
        'n_intensity_points': 376,
        'n_params': 8,
        'param_names': ['Zero', 'Lattice_a', 'Biso_Ce', 'Biso_O', 'Scale', 'U', 'V', 'W'],
        'scaling_factors': {
            'Zero': 1000.0, 'Lattice_a': 10.0, 'Biso_Ce': 1000.0, 'Biso_O': 100.0,
            'Scale': 100000.0, 'U': 1000.0, 'V': 1000.0, 'W': 1000.0
        },
        'param_info': {'param_counts': {'lattice parameter': 1, 'biso': 2}, 'has_background': False, 'total_params': 8},
        'crystal_system': 'cubic',
    },
    'pbso4': {
        'model_folder': 'model_pbso4_100epochs_digit_finetune_zero0p05_May07_2025',
        'model_file': 'model_pbso4_64.pth',
        'train_data_folder': 'pbso4_20250514_173501',
        'train_data_file': 'pbso4_simulated_data_row_param.dat',
        'template_pcr': 'source_files/pbso4.pcr',
        'dat_filename': 'pbso4_simulated.dat',
        'two_theta_start': 10.00,
        'two_theta_step': 0.40,
        'two_theta_end': 154.00,
        'n_intensity_points': 361,
        'n_params': 13,
        'param_names': ['Zero', 'Lattice_a', 'Lattice_b', 'Lattice_c', 'Biso_Pb', 'Biso_S', 'Biso_O1', 'Biso_O2', 'Biso_O3', 'Scale', 'U', 'V', 'W'],
        'scaling_factors': {
            'Zero': 100.0, 'Lattice_a': 1.0, 'Lattice_b': 1.0, 'Lattice_c': 1.0,
            'Biso_Pb': 1.0, 'Biso_S': 1.0, 'Biso_O1': 1.0, 'Biso_O2': 1.0, 'Biso_O3': 1.0,
            'Scale': 10.0, 'U': 10.0, 'V': 10.0, 'W': 10.0
        },
        'param_info': {'param_counts': {'lattice parameter': 3, 'biso': 5}, 'has_background': False, 'total_params': 13},
        'crystal_system': 'orthorhombic',
    },
    'tbbaco': {
        'model_folder': 'model_tbbaco_100epochs_digit_June29_2025',
        'model_file': 'model_tbbaco_10K_185.pth',
        'train_data_folder': 'tbbaco_20250627_033355',
        'train_data_file': 'tbbaco_simulated_data_row_param.dat',
        'template_pcr': 'source_files/tbbaco.pcr',
        'dat_filename': 'tbbaco_simulated.dat',
        'two_theta_start': 15.00,
        'two_theta_step': 0.30,
        'two_theta_end': 125.00,
        'n_intensity_points': 368,
        'n_params': 13,
        'param_names': ['Zero', 'Lattice_a', 'Lattice_b', 'Lattice_c', 'Biso_Ba', 'Biso_Tb', 'Biso_Co', 'Biso_O1', 'Biso_O2', 'Scale', 'U', 'V', 'W'],
        'scaling_factors': {
            'Zero': 100.0, 'Lattice_a': 1.0, 'Lattice_b': 1.0, 'Lattice_c': 0.1,
            'Biso_Ba': 100.0, 'Biso_Tb': 100.0, 'Biso_Co': 100.0, 'Biso_O1': 100.0, 'Biso_O2': 100.0,
            'Scale': 10000.0, 'U': 100.0, 'V': 100.0, 'W': 100.0
        },
        'param_info': {'param_counts': {'lattice parameter': 3, 'biso': 5}, 'has_background': False, 'total_params': 13},
        'crystal_system': 'orthorhombic',
    },
}


def get_material_config(material_name):
    if material_name not in MATERIALS:
        raise ValueError(f"Unknown material: {material_name}. Available: {list(MATERIALS.keys())}")

    config = MATERIALS[material_name].copy()

    # Build full paths
    config['model_path'] = os.path.join(CNN_DIR, 'saved_models', 'backup',
                                        config['model_folder'], config['model_file'])
    config['train_data_path'] = os.path.join(CNN_DIR, 'data', 'train_data',
                                             config['train_data_folder'], config['train_data_file'])
    config['template_pcr_path'] = os.path.join(CNN_DIR, 'data', 'train_data',
                                               config['train_data_folder'], config['template_pcr'])
    config['output_dir'] = os.path.join(CNN_DIR, 'saved_models', 'backup',
                                        config['model_folder'], 'uncertainty_results', 'rwp_uncertainty')

    return config


def list_materials():
    """List all available materials."""
    return list(MATERIALS.keys())
