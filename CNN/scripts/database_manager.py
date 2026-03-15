#!/usr/bin/env python


import os
import sys
import json
import glob
import re
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

DATABASE_DIR = os.path.join('single_phase_identification', 'database')
FINGERPRINTS_DIR = os.path.join(DATABASE_DIR, 'fingerprints')
VISUALIZATIONS_DIR = os.path.join(FINGERPRINTS_DIR, 'visualizations')
CATALOG_PATH = os.path.join(DATABASE_DIR, 'material_catalog.json')
SAVED_MODELS_DIR = 'saved_models'
TRAIN_DATA_DIR = os.path.join('data', 'train_data')

PARAM_TYPE_ZERO = 'zero'
PARAM_TYPE_BACKGROUND = 'background'
PARAM_TYPE_LATTICE = 'lattice parameter'
PARAM_TYPE_BISO = 'biso'
PARAM_TYPE_SCALE = 'scale factor'
PARAM_TYPE_U = 'u parameter'
PARAM_TYPE_V = 'v parameter'
PARAM_TYPE_W = 'w parameter'

def setup_directories():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(FINGERPRINTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    print(f"Created necessary directories: {DATABASE_DIR}")

def extract_material_name(folder_name):
    match = re.match(r'(?:model_)?([a-zA-Z0-9]+)_', folder_name)
    if match:
        return match.group(1)
    return folder_name  # Fallback to the full name

def scan_models():

    catalog = {}
    
    print(f"Scanning for trained models in {SAVED_MODELS_DIR}...")
    
    model_folders = [f for f in os.listdir(SAVED_MODELS_DIR) 
                    if os.path.isdir(os.path.join(SAVED_MODELS_DIR, f)) and f != "backup"]
    
    for folder in model_folders:
        model_path = os.path.join(SAVED_MODELS_DIR, folder)
        
        pth_files = glob.glob(os.path.join(model_path, "*.pth"))
        
        if not pth_files:
            print(f"  Skipping {folder}: No .pth model file found")
            continue
            
        pth_file = pth_files[0]
        
        material_name = extract_material_name(folder)
        print(f"  Found model for {material_name}: {os.path.basename(pth_file)}")
        
        if material_name in catalog:
            print(f"  Updating existing entry for {material_name}")
        
        training_info = extract_training_info(model_path)
        
        catalog[material_name] = {
            "model_path": pth_file,
            "model_folder": folder,
            "crystal_system": training_info.get("crystal_system", "cubic"),
            "scaling_factors": training_info.get("scaling_factors"),
            "parameter_count": training_info.get("param_count", 8),
            "omit_background": training_info.get("omit_background", False),
            "lattice_count": training_info.get("lattice_count", 1),
            "biso_count": training_info.get("biso_count", 1)
        }
    
    if not catalog:
        print("Warning: No trained models found in saved_models directory")
    else:
        print(f"Found models for {len(catalog)} materials: {', '.join(catalog.keys())}")
    
    return catalog

def extract_training_info(model_path):
    info = {
        "crystal_system": "cubic",  # Default
        "param_count": 8,
        "omit_background": False,
        "scaling_factors": [100, 0.1, 1.0, 10.0, 100000, 1000, 1000, 1000], 
        "lattice_count": 1,
        "biso_count": 1
    }
    
    training_result_folders = glob.glob(os.path.join(model_path, 'training_result', '*'))
    
    for folder in training_result_folders:
        training_files = glob.glob(os.path.join(folder, 'cnn_training_results_*.dat'))
        for log_file in training_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                    bg_match = re.search(r'Omit background parameter:\s*([yYtT][a-zA-Z]*)', content)
                    if bg_match:
                        info["omit_background"] = True
                    
                    crystal_match = re.search(r'crystal structure:\s*(\w+)', content, re.IGNORECASE)
                    if crystal_match:
                        info["crystal_system"] = crystal_match.group(1).lower()
                    
                    param_match = re.search(r'total parameters:\s*(\d+)', content, re.IGNORECASE)
                    if param_match:
                        info["param_count"] = int(param_match.group(1))
                    
                    lattice_match = re.search(r'Lattice parameters:\s*(\d+)', content, re.IGNORECASE)
                    if lattice_match:
                        info["lattice_count"] = int(lattice_match.group(1))
                    
                    biso_match = re.search(r'Biso parameters:\s*(\d+)', content, re.IGNORECASE)
                    if biso_match:
                        info["biso_count"] = int(biso_match.group(1))
                    
                    scaling_match = re.search(r'Scaling factors:\s*\[([\d\., ]+)\]', content)
                    if scaling_match:
                        factors_str = scaling_match.group(1)
                        try:
                            info["scaling_factors"] = [float(x.strip()) for x in factors_str.split(',')]
                        except:
                            pass
            except Exception as e:
                print(f"  Error reading training log {log_file}: {e}")
    
    return info

def find_pcr_templates(catalog):
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"Warning: Training data directory {TRAIN_DATA_DIR} not found")
        return catalog
    
    print(f"Searching for PCR templates...")
    
    templates_found = []
    
    for material_name, info in catalog.items():
        if "model_folder" in info:
            model_dir = os.path.join(SAVED_MODELS_DIR, info["model_folder"])
            
            refinement_paths = glob.glob(os.path.join(model_dir, 'refinement_result', '*', '*'))
            
            for refine_path in refinement_paths:
                solution_dir = os.path.join(refine_path, 'Rietveld_Refinement', 'solution_refinement')
                if os.path.exists(solution_dir):
                    pcr_files = glob.glob(os.path.join(solution_dir, '*.pcr'))
                    if pcr_files:
                        pcr_template = pcr_files[0]
                        catalog[material_name]['pcr_template'] = pcr_template
                        templates_found.append(material_name)
                        print(f"  Found solution PCR template for {material_name}")
                        break  # Found a template, move to next material
            
            if material_name not in templates_found:
                for refine_path in refinement_paths:
                    cnn_dir = os.path.join(refine_path, 'Rietveld_Refinement', 'CNN_ML_refinement')
                    if os.path.exists(cnn_dir):
                        pcr_files = glob.glob(os.path.join(cnn_dir, '*.pcr'))
                        if pcr_files:
                            pcr_template = pcr_files[0]
                            catalog[material_name]['pcr_template'] = pcr_template
                            templates_found.append(material_name)
                            print(f"  Found CNN PCR template for {material_name}")
                            break
    
    for material_name, info in catalog.items():
        if material_name in templates_found:
            continue  # Already found in refinement folders
        
        material_folders = [f for f in os.listdir(TRAIN_DATA_DIR) 
                           if os.path.isdir(os.path.join(TRAIN_DATA_DIR, f)) and f != "backup" 
                           and extract_material_name(f) == material_name]
        
        for folder in material_folders:
            source_files_path = os.path.join(TRAIN_DATA_DIR, folder, 'source_files')
            if not os.path.exists(source_files_path):
                continue
            
            pcr_files = glob.glob(os.path.join(source_files_path, '*.pcr'))
            if not pcr_files:
                continue
            
            pcr_template = pcr_files[0]
            
            catalog[material_name]['pcr_template'] = pcr_template
            templates_found.append(material_name)
            print(f"  Found training data PCR template for {material_name}")
            break  # Found a template, move to next material
    
    if templates_found:
        print(f"Added PCR templates for {len(templates_found)} materials")
    else:
        print("No PCR templates found")
    
    return catalog

def calculate_peak_shape_metrics(two_theta_values, intensity_values, peak_positions, window_size=10):

    fwhm_values = []
    asymmetry_values = []
    peak_heights = []
    
    interp_func = interp1d(two_theta_values, intensity_values, kind='cubic', 
                         bounds_error=False, fill_value=0)
    
    for peak_pos in peak_positions:
        peak_idx = np.argmin(np.abs(two_theta_values - peak_pos))
        
        window_start = max(0, peak_idx - window_size)
        window_end = min(len(two_theta_values), peak_idx + window_size + 1)
        
        window_x = two_theta_values[window_start:window_end]
        window_y = intensity_values[window_start:window_end]
        
        if len(window_x) < 5:  # Not enough points for reliable analysis
            fwhm_values.append(None)
            asymmetry_values.append(None)
            peak_heights.append(None)
            continue
        
        x_fine = np.linspace(window_x[0], window_x[-1], 1000)
        y_fine = interp_func(x_fine)
        max_idx = np.argmax(y_fine)
        peak_height = y_fine[max_idx]
        refined_peak_pos = x_fine[max_idx]
        
        half_max = peak_height / 2
        
        try:
            above_half_max = y_fine >= half_max
            transitions = np.diff(above_half_max.astype(int))
            rising_edges = np.where(transitions == 1)[0]
            falling_edges = np.where(transitions == -1)[0]
            
            if len(rising_edges) > 0 and len(falling_edges) > 0:
                rising_before_peak = rising_edges[rising_edges < max_idx]
                if len(rising_before_peak) > 0:
                    left_idx = rising_before_peak[-1]
                else:
                    left_idx = 0
                
                falling_after_peak = falling_edges[falling_edges > max_idx]
                if len(falling_after_peak) > 0:
                    right_idx = falling_after_peak[0]
                else:
                    right_idx = len(x_fine) - 1
                
                fwhm = x_fine[right_idx] - x_fine[left_idx]
                
                left_width = refined_peak_pos - x_fine[left_idx]
                right_width = x_fine[right_idx] - refined_peak_pos
                
                if left_width > 0:
                    asymmetry = right_width / left_width
                else:
                    asymmetry = 1.0
                
                fwhm_values.append(fwhm)
                asymmetry_values.append(asymmetry)
                peak_heights.append(peak_height)
            else:
                fwhm_values.append(None)
                asymmetry_values.append(None)
                peak_heights.append(peak_height)
        except Exception as e:
            print(f"  Error calculating peak metrics at 2θ={peak_pos:.2f}: {e}")
            fwhm_values.append(None)
            asymmetry_values.append(None)
            peak_heights.append(None)
    
    valid_fwhm = [v for v in fwhm_values if v is not None]
    valid_asymmetry = [v for v in asymmetry_values if v is not None]
    
    result = {
        "fwhm": fwhm_values,
        "asymmetry": asymmetry_values,
        "peak_heights": peak_heights,
        "stats": {
            "fwhm_mean": np.mean(valid_fwhm) if valid_fwhm else None,
            "fwhm_std": np.std(valid_fwhm) if valid_fwhm else None,
            "asymmetry_mean": np.mean(valid_asymmetry) if valid_asymmetry else None,
            "asymmetry_std": np.std(valid_asymmetry) if valid_asymmetry else None
        }
    }
    
    return result

def extract_peaks_from_prf(prf_file):

    import os
    import re
    import numpy as np
    
    if not os.path.isfile(prf_file):
        print(f"Error: PRF file not found: {prf_file}")
        return None
    
    try:
        with open(prf_file, 'r') as f:
            content = f.readlines()
        
        profile_data = []
        profile_section = False
        
        for line in content:
            line = line.strip()
            
            if "2Theta" in line and "Yobs" in line and "Ycal" in line:
                profile_section = True
                continue
            
            if profile_section:
                if not line or "(" in line and ")" in line and ":" in line:
                    profile_section = False
                    continue
                
                parts = line.split()
                if len(parts) >= 3:  # Need at least 2theta, Yobs, Ycal
                    try:
                        two_theta = float(parts[0])
                        yobs = float(parts[1])
                        ycal = float(parts[2])
                        
                        profile_data.append((two_theta, yobs, ycal))
                    except (ValueError, IndexError):
                        continue
        
        if profile_data:
            profile_data = np.array(profile_data)
            two_theta_values = profile_data[:, 0]
            yobs_values = profile_data[:, 1]
            ycal_values = profile_data[:, 2]
        else:
            print(f"Warning: No profile data found in {prf_file}")
            return None
        
        peak_positions = []
        hkl_values = []
        
        reflection_markers = [
            "Bragg R-factor",
            "BRAGG R-FACTOR", 
            "h   k   l",
            "Reflection data"
        ]
        
        for i, line in enumerate(content):
            line = line.strip()
            
            section_start = False
            for marker in reflection_markers:
                if marker in line:
                    section_start = True
                    break
            
            if section_start:
                for j in range(i+1, min(i+100, len(content))):
                    ref_line = content[j].strip()
                    
                    if ("(" in ref_line and ")" in ref_line and 
                        re.search(r'\(\s*[-+]?\d+\s+[-+]?\d+\s+[-+]?\d+\s*\)', ref_line)):
                        
                        parts = ref_line.split()
                        
                        two_theta_val = None
                        for part in parts:
                            try:
                                val = float(part)
                                if 0 < val < 180:
                                    two_theta_val = val
                                    break
                            except ValueError:
                                continue
                        
                        if two_theta_val:
                            peak_positions.append(two_theta_val)
                            
                            hkl_match = re.search(r'\(\s*([-+]?\d+)\s+([-+]?\d+)\s+([-+]?\d+)\s*\)', ref_line)
                            if hkl_match:
                                h, k, l = hkl_match.groups()
                                hkl_values.append((h, k, l))
                            else:
                                hkl_values.append(None)
        
        if not peak_positions:
            for line in content:
                line = line.strip()
                if "(" in line and ")" in line and ":" in line:
                    parts = line.split()
                    
                    try:
                        pos = float(parts[0])
                        peak_positions.append(pos)
                        
                        hkl_match = re.search(r'\(\s*([-+]?\d+)\s+([-+]?\d+)\s+([-+]?\d+)\s*\)', line)
                        if hkl_match:
                            h, k, l = hkl_match.groups()
                            hkl_values.append((h, k, l))
                        else:
                            hkl_values.append(None)
                    except (ValueError, IndexError):
                        continue
        
        if not peak_positions and len(yobs_values) > 5:
            from scipy.signal import find_peaks
            
            min_height = max(yobs_values) * 0.1
            peak_indices, _ = find_peaks(yobs_values, height=min_height, distance=10)
            
            for idx in peak_indices:
                peak_positions.append(two_theta_values[idx])
                hkl_values.append(None)  # No HKL info available from this method
        
        peak_intensities = []
        d_spacings = []
        
        for pos in peak_positions:
            idx = np.argmin(np.abs(two_theta_values - pos))
            
            intensity = yobs_values[idx]
            peak_intensities.append(intensity)
            
            d_spacing = 1.5406 / (2 * np.sin(np.radians(pos/2)))
            d_spacings.append(d_spacing)
        
        print(f"  Calculating peak shape metrics for {len(peak_positions)} peaks...")
        shape_metrics = calculate_peak_shape_metrics(
            two_theta_values, yobs_values, peak_positions
        )
        
        if peak_positions:
            result = {
                "positions": peak_positions,
                "intensities": peak_intensities,
                "d_spacings": d_spacings,
                "hkl_values": hkl_values,
                "fwhm_values": shape_metrics["fwhm"],
                "asymmetry_values": shape_metrics["asymmetry"],
                "shape_stats": shape_metrics["stats"],
                "profile_data": {
                    "two_theta": two_theta_values.tolist(),
                    "yobs": yobs_values.tolist(),
                    "ycalc": ycal_values.tolist()
                }
            }
            
            print(f"Successfully extracted {len(peak_positions)} peaks with shape analysis from {prf_file}")
            if peak_intensities:
                print(f"  Intensity range: {min(peak_intensities):.1f} - {max(peak_intensities):.1f}")
            
            if shape_metrics["stats"]["fwhm_mean"]:
                print(f"  Mean FWHM: {shape_metrics['stats']['fwhm_mean']:.4f}°")
                print(f"  Mean asymmetry: {shape_metrics['stats']['asymmetry_mean']:.4f}")
            
            return result
        else:
            print(f"Warning: No peak positions found in {prf_file}")
            return None
            
    except Exception as e:
        print(f"Error extracting peaks from {prf_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_fingerprint_visualization(fingerprint, material_name, output_path):

    try:
        positions = fingerprint.get('peak_positions', [])
        intensities = fingerprint.get('peak_intensities', [])
        fwhm_values = fingerprint.get('fwhm_values', [])
        asymmetry_values = fingerprint.get('asymmetry_values', [])
        
        if not positions or not intensities:
            print(f"Warning: No peak data for visualization of {material_name}")
            return False
        
        print(f"  Creating visualization with {len(positions)} peaks for {material_name}")
        print(f"  Intensity range in visualization: {min(intensities):.1f} - {max(intensities):.1f}")
        
        display_name = material_name
        if material_name.lower() == 'pbso4':
            display_name = r'$\mathrm{PbSO_4}$'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), height_ratios=[3, 1], sharex=True)
        
        plt.subplots_adjust(bottom=0.15)
        
        for pos, intensity in zip(positions, intensities):
            ax1.vlines(pos, 0, intensity, linewidth=1.5)
            
        ax1.scatter(positions, intensities, color='red', s=20, zorder=3)
        
        ax1.set_title(f"XRD Pattern Fingerprint Match", fontsize=35)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=30)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        ax1.set_xlim(10, 120)
        ax2.set_xlim(10, 120)
        ax1.set_xticks([20, 40, 60, 80, 100])
        
        if intensities:
            ax1.set_ylim(0, max(intensities) * 1.3)
        
        info_text = f"Material: {display_name}\n"
        info_text += f"Peaks: {len(positions)}\n"
        
        shape_stats = fingerprint.get('shape_stats', {})
        if shape_stats and shape_stats.get('fwhm_mean') is not None:
            info_text += f"Mean FWHM: {shape_stats['fwhm_mean']:.4f}°\n"
            info_text += f"Mean Asymmetry: {shape_stats['asymmetry_mean']:.4f}\n"
        
        if intensities:
            info_text += f"Max Intensity: {max(intensities):.1f}"
        
        ax1.text(0.98, 0.97, info_text, 
                transform=ax1.transAxes,
                fontsize=25,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        valid_indices = []
        valid_fwhm = []
        valid_asymmetry = []
        
        for i, (fwhm, asym) in enumerate(zip(fwhm_values, asymmetry_values)):
            if fwhm is not None and asym is not None:
                valid_indices.append(i)
                valid_fwhm.append(fwhm)
                valid_asymmetry.append(asym)
        
        if valid_indices:
            valid_positions = [positions[i] for i in valid_indices]
            
            bar1 = ax2.bar(valid_positions, valid_fwhm, width=0.5, alpha=0.6, label='FWHM (deg.)')
            
            ax3 = ax2.twinx()
            line1 = ax3.plot(valid_positions, valid_asymmetry, 's-', color='red', label='Asymmetry', linewidth=2, markersize=8)[0]
            ax3.set_ylabel('Asymmetry (right/left ratio)', fontsize=30)
            ax3.tick_params(axis='y', labelsize=30)
            
            ax3.set_ylim(0, max(2.0, max(valid_asymmetry) * 1.1))
            
            ax2.set_xlabel('2θ (deg.)', fontsize=30)
            ax2.set_ylabel('FWHM (deg.)', fontsize=30)
            ax2.tick_params(axis='y', labelsize=30)
            ax2.grid(True, linestyle='--', alpha=0.3)
            
            ax2.set_ylim(0, max(0.5, max(valid_fwhm) * 1.5))
            
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            fig.legend(lines1 + lines2, labels1 + labels2, loc='lower right', 
                      bbox_to_anchor=(0.98, -0.06), fontsize=30, ncol=2,
                      frameon=True, fancybox=True, shadow=True)
        else:
            ax2.text(0.5, 0.5, "No valid FWHM/asymmetry data available", 
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_xlabel('2θ (deg.)', fontsize=30)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"  Created enhanced fingerprint visualization for {material_name}")
        return True
        
    except Exception as e:
        print(f"Error creating visualization for {material_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_fingerprints(catalog):

    print("Generating XRD pattern fingerprints with peak shape analysis...")
    
    fingerprints_created = []
    
    for material_name, info in catalog.items():
        pattern_found = False
        
        if "model_folder" in info:
            model_dir = os.path.join(SAVED_MODELS_DIR, info["model_folder"])
            refinement_paths = glob.glob(os.path.join(model_dir, 'refinement_result', '*', '*'))
            
            for refine_path in refinement_paths:
                solution_dir = os.path.join(refine_path, 'Rietveld_Refinement', 'solution_refinement')
                if os.path.exists(solution_dir):
                    prf_files = glob.glob(os.path.join(solution_dir, '*.prf'))
                    if prf_files:
                        prf_file = prf_files[0]
                        print(f"  Processing solution PRF file for {material_name}: {os.path.basename(prf_file)}")
                        
                        peaks = extract_peaks_from_prf(prf_file)
                        if peaks:
                            fingerprint = {
                                "peak_positions": peaks["positions"],
                                "peak_intensities": peaks["intensities"],
                                "d_spacings": peaks["d_spacings"],
                                "fwhm_values": peaks["fwhm_values"],
                                "asymmetry_values": peaks["asymmetry_values"],
                                "shape_stats": peaks["shape_stats"],
                                "peak_count": len(peaks["positions"]),
                                "source": "solution_prf_file",
                                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            fingerprint_path = os.path.join(FINGERPRINTS_DIR, f"{material_name}_fingerprint.json")
                            with open(fingerprint_path, 'w') as f:
                                json.dump(fingerprint, f, indent=2)
                            
                            viz_path = os.path.join(VISUALIZATIONS_DIR, f"{material_name}_fingerprint.pdf")
                            create_fingerprint_visualization(fingerprint, material_name, viz_path)
                            
                            catalog[material_name]["fingerprint_path"] = fingerprint_path
                            catalog[material_name]["visualization_path"] = viz_path
                            fingerprints_created.append(material_name)
                            pattern_found = True
                            break
                
                if pattern_found:
                    break
        
        if not pattern_found:
            for folder in os.listdir(TRAIN_DATA_DIR):
                folder_material = extract_material_name(folder)
                if folder_material != material_name:
                    continue
                    
                prf_paths = [
                    os.path.join(TRAIN_DATA_DIR, folder, 'output_figures', 'PRF_files'),
                    os.path.join(TRAIN_DATA_DIR, folder, 'generated_samples', 'sample1')
                ]
                
                for prf_path in prf_paths:
                    if not os.path.exists(prf_path):
                        continue
                        
                    prf_files = glob.glob(os.path.join(prf_path, '*.prf'))
                    if not prf_files:
                        continue
                    
                    prf_file = prf_files[0]
                    print(f"  Processing training data PRF file for {material_name}: {os.path.basename(prf_file)}")
                    
                    peaks = extract_peaks_from_prf(prf_file)
                    if not peaks:
                        continue
                    
                    fingerprint = {
                        "peak_positions": peaks["positions"],
                        "peak_intensities": peaks["intensities"],
                        "d_spacings": peaks["d_spacings"],
                        "fwhm_values": peaks["fwhm_values"],
                        "asymmetry_values": peaks["asymmetry_values"],
                        "shape_stats": peaks["shape_stats"],
                        "peak_count": len(peaks["positions"]),
                        "source": "training_prf_file",
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    fingerprint_path = os.path.join(FINGERPRINTS_DIR, f"{material_name}_fingerprint.json")
                    with open(fingerprint_path, 'w') as f:
                        json.dump(fingerprint, f, indent=2)
                    
                    viz_path = os.path.join(VISUALIZATIONS_DIR, f"{material_name}_fingerprint.pdf")
                    create_fingerprint_visualization(fingerprint, material_name, viz_path)
                    
                    catalog[material_name]["fingerprint_path"] = fingerprint_path
                    catalog[material_name]["visualization_path"] = viz_path
                    fingerprints_created.append(material_name)
                    pattern_found = True
                    break
                
                if pattern_found:
                    break
    
    if fingerprints_created:
        print(f"Created enhanced fingerprints with peak shape analysis for {len(fingerprints_created)} materials")
    else:
        print("No fingerprints created")
    
    return catalog

def save_catalog(catalog):
    with open(CATALOG_PATH, 'w') as f:
        json.dump(catalog, f, indent=2)
    print(f"Saved material catalog with {len(catalog)} entries to {CATALOG_PATH}")

def check_database():
    if not os.path.exists(CATALOG_PATH):
        print("Material catalog not found")
        return False
    
    try:
        with open(CATALOG_PATH, 'r') as f:
            catalog = json.load(f)
        
        print(f"Material catalog exists with {len(catalog)} entries")
        
        valid_models = 0
        for material, info in catalog.items():
            if "model_path" in info and os.path.exists(info["model_path"]):
                valid_models += 1
        
        if valid_models == 0:
            print("Warning: No valid model paths found in catalog")
            return False
        
        print(f"Found {valid_models} valid model references")
        return True
    
    except Exception as e:
        print(f"Error checking database: {e}")
        return False

def update_database():
    setup_directories()
    
    existing_catalog = {}
    if os.path.exists(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, 'r') as f:
                existing_catalog = json.load(f)
            print(f"Loaded existing catalog with {len(existing_catalog)} entries")
        except Exception as e:
            print(f"Error loading existing catalog: {e}")
    
    catalog = scan_models()
    
    for material, info in existing_catalog.items():
        if material not in catalog:
            print(f"Preserving existing entry for {material}")
            catalog[material] = info
    
    catalog = find_pcr_templates(catalog)
    
    catalog = process_fingerprints(catalog)
    
    save_catalog(catalog)
    
    return len(catalog) > 0

def main():
    parser = argparse.ArgumentParser(description="XRD Database Manager")
    parser.add_argument("--update", action="store_true", help="Create/update the database")
    parser.add_argument("--check", action="store_true", help="Check if database exists and is valid")
    
    args = parser.parse_args()
    
    if args.check:
        return check_database()
    elif args.update:
        return update_database()
    else:
        parser.print_help()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)