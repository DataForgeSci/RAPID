#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import torch

from utils import CNN
from rwp_uncertainty_config import get_material_config, list_materials


def load_model(config, device):
    param_info = config['param_info']
    n_params = config['n_params']
    model = CNN(output_dim=n_params, param_info=param_info).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device, weights_only=True), strict=False)
    model.eval()
    return model


def load_pattern(config, idx):
    n_intensity = config['n_intensity_points']
    with open(config['train_data_path'], 'r') as f:
        for i, line in enumerate(f):
            if i == idx:
                parts = line.strip().split()
                intensities = np.array([float(parts[j]) for j in range(n_intensity)], dtype=np.float32)
                return intensities
    return None


def create_dat_file(config, intensities, output_path):
    two_theta_start = config['two_theta_start']
    two_theta_step = config['two_theta_step']
    two_theta = np.array([two_theta_start + i * two_theta_step for i in range(len(intensities))])
    with open(output_path, 'w') as f:
        for tt, intensity in zip(two_theta, intensities):
            f.write(f"{tt:15.5f}\t{intensity:15.5f}\n")


def run_inference(config, model, intensities, device):
    max_val = np.max(intensities)
    normalized = intensities / max_val if max_val > 0 else intensities
    normalized = normalized - np.mean(normalized)
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    predictions = output.cpu().numpy()[0]
    params = {}
    param_names = config['param_names']
    scaling_factors = config['scaling_factors']
    for i, name in enumerate(param_names):
        params[name] = predictions[i] / scaling_factors[name]
    return params


def create_pcr_with_params(config, template_pcr, output_pcr, pred_params):
    with open(template_pcr, 'r') as f:
        content = f.read()

    dat_filename = config['dat_filename']

    # Update data file reference
    lines = content.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if '!File names of data(patterns) files' in line:
            new_lines.append(line)
            # Skip the next line (old dat filename) and add new one
            if i + 1 < len(lines):
                new_lines.append(dat_filename)
                continue
        elif i > 0 and '!File names of data(patterns) files' in lines[i-1]:
            continue  # Skip old dat filename
        else:
            new_lines.append(line)

    content = '\n'.join(new_lines)

    # Update 2theta range
    two_theta_start = config['two_theta_start']
    two_theta_step = config['two_theta_step']
    two_theta_end = config['two_theta_end']
    content = re.sub(
        r'(!     Thmin       Step       Thmax.*\n)\s*[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+',
        f'\\g<1>    {two_theta_start:.4f}   {two_theta_step:.6f}   {two_theta_end:.4f}   0.000   0.000',
        content
    )

    with open(output_pcr, 'w') as f:
        f.write(content)


def main():
    if len(sys.argv) < 3:
        print("Usage: python rwp_uncertainty_step1.py <material> <start_idx> [end_idx]")
        print(f"Available materials: {', '.join(list_materials())}")
        sys.exit(1)

    material = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3]) if len(sys.argv) > 3 else start_idx

    print("=" * 60)
    print(f"RWP UNCERTAINTY STEP 1 - {material}")
    print("=" * 60)

    try:
        config = get_material_config(material)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {config['model_path']}")
    print(f"Training data: {config['train_data_path']}")

    os.makedirs(config['output_dir'], exist_ok=True)
    model = load_model(config, device)

    for idx in range(start_idx, end_idx + 1):
        print(f"\nProcessing pattern {idx}...")

        pattern_dir = os.path.join(config['output_dir'], f'pattern_{idx:04d}')
        cnn_dir = os.path.join(pattern_dir, 'CNN_ML_refinement')
        os.makedirs(cnn_dir, exist_ok=True)

        intensities = load_pattern(config, idx)
        if intensities is None:
            print(f"  Error: Could not load pattern {idx}")
            continue

        pred_params = run_inference(config, model, intensities, device)

        create_dat_file(config, intensities, os.path.join(cnn_dir, config['dat_filename']))
        create_pcr_with_params(config, config['template_pcr_path'], os.path.join(cnn_dir, 'cnn_refined.pcr'), pred_params)

        print(f"  Created files in {cnn_dir}")

    print("\n" + "=" * 60)
    print("Step 1 complete. Now activate conda py27 and run step 2:")
    print("  conda activate py27")
    print(f"  python rwp_uncertainty_step2.py {material} {start_idx} {end_idx}")
    print("=" * 60)


if __name__ == "__main__":
    main()
