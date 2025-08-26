#!/usr/bin/env python


import os
import sys
import json
import glob
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import subprocess
from pathlib import Path
import traceback
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

DATABASE_DIR = os.path.join('single_phase_identification', 'database')
CATALOG_PATH = os.path.join(DATABASE_DIR, 'material_catalog.json')
FINGERPRINTS_DIR = os.path.join(DATABASE_DIR, 'fingerprints')
IDENTIFIED_DIR = os.path.join('single_phase_identification', 'identified_materials')
SAVED_MODELS_DIR = 'saved_models'

DAT_FORMATS = ["Si.dat", "CeO2.dat", "pbso4.dat", "tbbaco.dat"]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Identify unknown material from XRD pattern")
    parser.add_argument("--file", required=True, help="Path to unknown XRD pattern file")
    return parser.parse_args()

def load_material_catalog():
    if not os.path.exists(CATALOG_PATH):
        print("Error: Material catalog not found. Please run database update first.")
        sys.exit(1)
    
    with open(CATALOG_PATH, 'r') as f:
        return json.load(f)

def detect_dat_format(file_path):

    try:
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read(5000)  
        
        if re.search(r'^\s*1\s+\d+', content, re.MULTILINE) or re.search(r'^\s*10\s+\d+10', content, re.MULTILINE):
            print(f"Detected pbso4.dat format based on line prefixes")
            return "pbso4.dat"
        
        first_line = content.strip().split('\n')[0]
        parts = first_line.strip().split()
        if len(parts) >= 3 and all(re.match(r'^\d+\.?\d*$', p) for p in parts[:3]):
            print(f"Detected tbbaco.dat format based on header structure")
            return "tbbaco.dat"
        
        two_col_count = 0
        first_few_lines = content.strip().split('\n')[:10]
        for line in first_few_lines:
            parts = line.strip().split()
            if len(parts) == 2 and all(re.match(r'^[-+]?\d*\.?\d+$', p) for p in parts):
                two_col_count += 1
        
        if two_col_count >= 3:
            print(f"Detected standard 2-column format (Si.dat)")
            return "Si.dat"
            
        print(f"Could not determine format, defaulting to Si.dat")
        return "Si.dat"
        
    except Exception as e:
        print(f"Error detecting file format: {e}")
        print(f"Defaulting to Si.dat format")
        return "Si.dat"

def read_xrd_pattern(file_path, dat_format=None):

    from utils import parse_dat_file
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None, None
    
    if dat_format is None:
        dat_format = detect_dat_format(file_path)
    
    print(f"Reading XRD pattern from {file_path} using {dat_format} format")
    
    ttheta_values, intensity_values = parse_dat_file(file_path, dat_format)
    
    if not ttheta_values or len(ttheta_values) == 0:
        print("Error: Failed to extract valid data from XRD file")
        return None, None
    
    print(f"Successfully read {len(ttheta_values)} data points")
    return ttheta_values, intensity_values

def load_fingerprint(material_name):
    fingerprint_path = os.path.join(FINGERPRINTS_DIR, f"{material_name}_fingerprint.json")
    if not os.path.exists(fingerprint_path):
        return None
    
    with open(fingerprint_path, 'r') as f:
        return json.load(f)

def calculate_peak_shape_metrics(two_theta_values, intensity_values, peak_positions, window_size=10):

    tth_arr = np.array(two_theta_values)
    intensity_arr = np.array(intensity_values)
    
    fwhm_values = []
    asymmetry_values = []
    peak_heights = []
    
    interp_func = interp1d(tth_arr, intensity_arr, kind='cubic', 
                         bounds_error=False, fill_value=0)
    
    for peak_pos in peak_positions:
        peak_idx = np.argmin(np.abs(tth_arr - peak_pos))
        
        window_start = max(0, peak_idx - window_size)
        window_end = min(len(tth_arr), peak_idx + window_size + 1)
        
        window_x = tth_arr[window_start:window_end]
        window_y = intensity_arr[window_start:window_end]
        
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

def find_peaks_in_xrd_pattern(ttheta_values, intensity_values, min_height=0.05, min_distance=0.5, min_prominence=0.02, width=None):

    from scipy.signal import find_peaks
    
    ttheta = np.array(ttheta_values)
    intensity = np.array(intensity_values)
    
    max_intensity = np.max(intensity)
    if max_intensity > 0:
        normalized = intensity / max_intensity
    else:
        return [], []
    
    if len(ttheta) > 1:
        step_size = np.median(np.diff(ttheta))
    else:
        step_size = 0.01  # Default value
    
    distance_points = int(min_distance / step_size) if step_size > 0 else 10
    
    peak_indices, properties = find_peaks(
        normalized,
        height=min_height,
        distance=distance_points,
        prominence=min_prominence,
        width=width
    )
    
    if len(peak_indices) == 0:
        peak_indices, _ = find_peaks(
            normalized,
            height=min_height/2,
            distance=max(3, distance_points//2),
            prominence=min_prominence/2
        )
        
        if len(peak_indices) == 0:
            return [], []
    
    peak_positions = ttheta[peak_indices].tolist()
    peak_intensities = intensity[peak_indices].tolist()
    
    return peak_positions, peak_intensities

def calculate_shape_match(unknown_fwhm, unknown_asymmetry, ref_fwhm, ref_asymmetry):

    valid_unknown_fwhm = [v for v in unknown_fwhm if v is not None]
    valid_unknown_asymmetry = [v for v in unknown_asymmetry if v is not None]
    valid_ref_fwhm = [v for v in ref_fwhm if v is not None]
    valid_ref_asymmetry = [v for v in ref_asymmetry if v is not None]
    
    if not valid_unknown_fwhm or not valid_ref_fwhm:
        return 0.0
    
    unknown_fwhm_mean = np.mean(valid_unknown_fwhm)
    ref_fwhm_mean = np.mean(valid_ref_fwhm)
    
    norm_unknown_fwhm = [v / unknown_fwhm_mean if v is not None else 1.0 for v in unknown_fwhm]
    norm_ref_fwhm = [v / ref_fwhm_mean if v is not None else 1.0 for v in ref_fwhm]
    
    
    fwhm_diffs = []
    for u_fwhm, r_fwhm in zip(norm_unknown_fwhm, norm_ref_fwhm):
        if u_fwhm is not None and r_fwhm is not None:
            diff = abs(u_fwhm - r_fwhm) / max(u_fwhm, r_fwhm)
            fwhm_diffs.append(diff)
    
    asym_diffs = []
    for u_asym, r_asym in zip(valid_unknown_asymmetry, valid_ref_asymmetry):
        if u_asym is not None and r_asym is not None:
            diff = abs(u_asym - r_asym) / max(u_asym, r_asym)
            asym_diffs.append(diff)
    
    avg_fwhm_diff = np.mean(fwhm_diffs) if fwhm_diffs else 1.0
    avg_asym_diff = np.mean(asym_diffs) if asym_diffs else 1.0
    
    fwhm_score = max(0, 1 - avg_fwhm_diff)
    asym_score = max(0, 1 - avg_asym_diff)
    
    shape_score = fwhm_score * 0.6 + asym_score * 0.4
    
    return shape_score

def match_pattern_to_materials(ttheta_values, intensity_values, catalog):

    match_results = []
    
    tth_arr = np.array(ttheta_values)
    intensity_arr = np.array(intensity_values)
    
    max_intensity = np.max(intensity_arr)
    normalized_intensities = intensity_arr / max_intensity * 100
    
    unknown_peak_positions, unknown_peak_intensities = find_peaks_in_xrd_pattern(
        tth_arr, 
        normalized_intensities,
        min_height=0.05,
        min_distance=0.3,
        min_prominence=0.02
    )
    
    if unknown_peak_positions:
        shape_metrics = calculate_peak_shape_metrics(
            ttheta_values, normalized_intensities, unknown_peak_positions
        )
        unknown_fwhm = shape_metrics["fwhm"]
        unknown_asymmetry = shape_metrics["asymmetry"]
        
        print(f"Extracted {len(unknown_peak_positions)} peaks from unknown pattern")
        if shape_metrics["stats"]["fwhm_mean"]:
            print(f"Unknown pattern mean FWHM: {shape_metrics['stats']['fwhm_mean']:.4f}°")
            print(f"Unknown pattern mean asymmetry: {shape_metrics['stats']['asymmetry_mean']:.4f}")
    else:
        print("Warning: No clear peaks detected in unknown pattern")
        unknown_fwhm = []
        unknown_asymmetry = []
    
    print("Matching pattern against material database with peak shape analysis...")
    
    for material_name in catalog:
        fingerprint = load_fingerprint(material_name)
        if not fingerprint:
            print(f"Skipping {material_name} - No fingerprint available")
            continue
        
        ref_positions = fingerprint.get("peak_positions", [])
        ref_intensities = fingerprint.get("peak_intensities", [])
        
        if not ref_positions or not ref_intensities:
            continue
        
        ref_fwhm = fingerprint.get("fwhm_values", [])
        ref_asymmetry = fingerprint.get("asymmetry_values", [])
        
        max_ref_intensity = max(ref_intensities)
        normalized_ref_intensities = [i / max_ref_intensity * 100 for i in ref_intensities]
        
        position_score = calculate_position_match(ttheta_values, ref_positions)
        
        intensity_score = calculate_intensity_correlation(
            ttheta_values, normalized_intensities,
            ref_positions, normalized_ref_intensities
        )
        
        shape_score = 0.0
        if unknown_fwhm and ref_fwhm and unknown_asymmetry and ref_asymmetry:
            shape_score = calculate_shape_match(
                unknown_fwhm, unknown_asymmetry,
                ref_fwhm, ref_asymmetry
            )
            
            print(f"  {material_name}: Position={position_score:.3f}, Intensity={intensity_score:.3f}, Shape={shape_score:.3f}")
        else:
            print(f"  {material_name}: Position={position_score:.3f}, Intensity={intensity_score:.3f}, Shape=N/A")
        
        if shape_score > 0:
            overall_score = (position_score * 0.5) + (intensity_score * 0.2) + (shape_score * 0.3)
        else:
            overall_score = position_score * 0.7 + intensity_score * 0.3
        
        match_results.append({
            "material_name": material_name,
            "position_score": position_score,
            "intensity_score": intensity_score,
            "shape_score": shape_score,
            "overall_score": overall_score,
            "peak_count": len(ref_positions),
            "fingerprint": fingerprint,  # Store the fingerprint for later use
            "unknown_shape_metrics": {
                "fwhm": unknown_fwhm,
                "asymmetry": unknown_asymmetry
            } if unknown_fwhm else None
        })
    
    match_results.sort(key=lambda x: x["overall_score"], reverse=True)
    
    print("\nTop matches:")
    for i, match in enumerate(match_results[:3]):
        shape_info = f", Shape={match['shape_score']:.3f}" if match['shape_score'] > 0 else ""
        print(f"{i+1}. {match['material_name']}: Overall={match['overall_score']:.3f} (Position={match['position_score']:.3f}, Intensity={match['intensity_score']:.3f}{shape_info})")
    
    return match_results

def calculate_position_match(unknown_ttheta, reference_positions, tolerance=0.3):

    if not reference_positions:
        return 0.0
    
    from scipy.signal import find_peaks
    unknown_y = np.interp(np.array(reference_positions), np.array(unknown_ttheta), np.array([1] * len(unknown_ttheta)))
    
    matching_peaks = 0
    for ref_pos in reference_positions:
        closest_idx = np.argmin(np.abs(np.array(unknown_ttheta) - ref_pos))
        closest_pos = unknown_ttheta[closest_idx]
        
        if abs(closest_pos - ref_pos) <= tolerance:
            matching_peaks += 1
    
    return matching_peaks / len(reference_positions)

def calculate_intensity_correlation(unknown_x, unknown_y, ref_x, ref_y):

    min_x = max(min(unknown_x), min(ref_x))
    max_x = min(max(unknown_x), max(ref_x))
    
    if min_x >= max_x:
        return 0.0
    
    grid_x = np.linspace(min_x, max_x, 500)
    
    unknown_interp = np.interp(grid_x, unknown_x, unknown_y)
    ref_interp = np.interp(grid_x, ref_x, ref_y)
    
    if np.all(unknown_interp == unknown_interp[0]) or np.all(ref_interp == ref_interp[0]):
        print("Warning: Constant array detected, correlation set to 0")
        return 0.0
    
    from scipy.stats import pearsonr
    try:
        corr, _ = pearsonr(unknown_interp, ref_interp)
        return max(0, corr)  # Ensure non-negative
    except Exception as e:
        print(f"Warning: Error calculating correlation: {e}")
        return 0.0

def create_fingerprint_comparison_plot(ttheta_values, intensity_values, match_results, output_path):

    if not match_results:
        print("No match results available for comparison plot")
        return False
    
    tth_arr = np.array(ttheta_values)
    intensity_arr = np.array(intensity_values)
    
    top_match = match_results[0]
    material_name = top_match["material_name"]
    fingerprint = top_match.get("fingerprint", {})
    
    display_name = material_name
    if material_name.lower() == 'pbso4':
        display_name = r'$\mathrm{PbSO_4}$'
    
    ref_positions = fingerprint.get("peak_positions", [])
    ref_intensities = fingerprint.get("peak_intensities", [])
    ref_fwhm = fingerprint.get("fwhm_values", [])
    ref_asymmetry = fingerprint.get("asymmetry_values", [])
    
    unknown_shape_metrics = top_match.get("unknown_shape_metrics", {})
    unknown_fwhm = unknown_shape_metrics.get("fwhm", []) if unknown_shape_metrics else []
    unknown_asymmetry = unknown_shape_metrics.get("asymmetry", []) if unknown_shape_metrics else []
    
    if not ref_positions or not ref_intensities:
        print(f"No fingerprint data available for {material_name}")
        return False
    
    max_intensity = np.max(intensity_arr)
    normalized_intensities = intensity_arr / max_intensity * 100
    
    max_ref_intensity = max(ref_intensities)
    normalized_ref_intensities = [i / max_ref_intensity * 100 for i in ref_intensities]
    
    unknown_peak_positions, unknown_peak_intensities = find_peaks_in_xrd_pattern(
        ttheta_values, 
        normalized_intensities,
        min_height=0.05,      
        min_distance=0.3,     
        min_prominence=0.02   
    )
    
    if unknown_peak_intensities:
        max_unknown_peak = max(unknown_peak_intensities)
        if max_unknown_peak > 0:
            unknown_peak_intensities = [i / max_unknown_peak * 100 for i in unknown_peak_intensities]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), 
                                       gridspec_kw={'height_ratios': [3, 1, 2]})
    
    plt.subplots_adjust(hspace=0.8, bottom=0.15)
    
    line1 = ax1.plot(ttheta_values, normalized_intensities, 'b-', linewidth=1.5, 
             label='Unknown Pattern (line)', alpha=0.7)[0]
    
    scatter1 = None
    if unknown_peak_positions:
        peak_intensities_plot = []
        for pos in unknown_peak_positions:
            idx = np.searchsorted(ttheta_values, pos)
            if idx >= len(ttheta_values):
                idx = len(ttheta_values) - 1
            elif idx > 0 and (abs(ttheta_values[idx-1] - pos) < abs(ttheta_values[idx] - pos)):
                idx = idx - 1
            
            peak_intensities_plot.append(normalized_intensities[idx])
        
        scatter1 = ax1.scatter(unknown_peak_positions, peak_intensities_plot, 
                   color='blue', s=50, marker='^', zorder=3,
                   edgecolors='darkblue', linewidths=1.5, 
                   label='Unknown Pattern Peaks')
    
    for pos, intensity in zip(ref_positions, normalized_ref_intensities):
        ax1.vlines(pos, 0, intensity, colors='r', linewidth=1.0, alpha=0.7)
    
    scatter2 = ax1.scatter(ref_positions, normalized_ref_intensities, color='r', s=30, marker='o',
                label=f'Reference: {display_name}')
    
    matching_positions = []
    for ref_pos in ref_positions:
        closest_idx = np.argmin(np.abs(np.array(ttheta_values) - ref_pos))
        closest_pos = ttheta_values[closest_idx]
        
        if abs(closest_pos - ref_pos) <= 0.3:
            matching_positions.append((ref_pos, closest_pos))
    
    for ref_pos, close_pos in matching_positions:
        ref_idx = ref_positions.index(ref_pos)
        ref_intensity = normalized_ref_intensities[ref_idx]
        
        close_idx = np.argmin(np.abs(np.array(ttheta_values) - close_pos))
        close_intensity = normalized_intensities[close_idx]
        
        ax1.plot([ref_pos, close_pos], 
                 [ref_intensity, close_intensity], 
                 'g--', linewidth=0.8, alpha=0.6)
    
    position_score = top_match["position_score"]
    intensity_score = top_match["intensity_score"]
    shape_score = top_match.get("shape_score", 0.0)
    overall_score = top_match["overall_score"]
    
    score_text = f"Position Score: {position_score:.3f}\n"
    score_text += f"Intensity Score: {intensity_score:.3f}\n"
    if shape_score > 0:
        score_text += f"Shape Score: {shape_score:.3f}\n"
    score_text += f"Overall Score: {overall_score:.3f}\n"
    score_text += f"Matching Peaks: {len(matching_positions)}/{len(ref_positions)}"
    
    # ax1.text(0.02, -1.97, score_text, transform=ax1.transAxes,
    #          fontsize=30, verticalalignment='top',
    #          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    if unknown_peak_positions:
        ax2.eventplot([unknown_peak_positions], colors=['b'], lineoffsets=[0.7], linelengths=[0.5])
    else:
        ax2.text(0.5, 0.7, "No clear peaks detected in unknown pattern", 
                horizontalalignment='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.eventplot([ref_positions], colors=['r'], lineoffsets=[0.3], linelengths=[0.5])
    
    if unknown_fwhm and ref_fwhm and len(unknown_peak_positions) > 0:
        bar_width = 0.35
        
        valid_indices = []
        valid_unknown_fwhm = []
        valid_ref_fwhm = []
        valid_unknown_asymmetry = []
        valid_ref_asymmetry = []
        valid_positions = []
        
        for i, unknown_pos in enumerate(unknown_peak_positions):
            if i >= len(unknown_fwhm) or unknown_fwhm[i] is None:
                continue
                
            closest_ref_idx = -1
            min_dist = float('inf')
            for j, ref_pos in enumerate(ref_positions):
                dist = abs(unknown_pos - ref_pos)
                if dist < min_dist and dist <= 0.5:  # Within 0.5 degrees
                    min_dist = dist
                    closest_ref_idx = j
            
            if closest_ref_idx >= 0 and closest_ref_idx < len(ref_fwhm) and ref_fwhm[closest_ref_idx] is not None:
                valid_indices.append(i)
                valid_unknown_fwhm.append(unknown_fwhm[i])
                valid_ref_fwhm.append(ref_fwhm[closest_ref_idx])
                valid_positions.append(unknown_pos)
                
                if i < len(unknown_asymmetry) and unknown_asymmetry[i] is not None:
                    valid_unknown_asymmetry.append(unknown_asymmetry[i])
                else:
                    valid_unknown_asymmetry.append(1.0)  # Default value
                    
                if closest_ref_idx < len(ref_asymmetry) and ref_asymmetry[closest_ref_idx] is not None:
                    valid_ref_asymmetry.append(ref_asymmetry[closest_ref_idx])
                else:
                    valid_ref_asymmetry.append(1.0)  # Default value
        
        if valid_positions:
            x = np.arange(len(valid_positions))
            bar1 = ax3.bar(x - bar_width/2, valid_unknown_fwhm, bar_width, label='Unknown FWHM', alpha=0.7, color='blue')
            bar2 = ax3.bar(x + bar_width/2, valid_ref_fwhm, bar_width, label='Reference FWHM', alpha=0.7, color='red')
            
            tick_positions = [f"{pos:.1f}" for pos in valid_positions]
            if len(x) > 6:
                indices = np.linspace(0, len(x)-1, 6, dtype=int)
                ax3.set_xticks(x[indices])
                ax3.set_xticklabels([tick_positions[i] for i in indices], rotation=0, ha='center')
            else:
                ax3.set_xticks(x)
                ax3.set_xticklabels(tick_positions, rotation=0, ha='center')
                        
            ax4 = ax3.twinx()
            ax4.set_ylabel('Asymmetry (right/left ratio)', color='black', fontsize=30)
            ax4.tick_params(axis='y', labelcolor='black')
            
            line3 = ax4.plot(x - bar_width/2, valid_unknown_asymmetry, 'o-', color='green', label='Unknown Asymmetry', linewidth=2, markersize=8)[0]
            line4 = ax4.plot(x + bar_width/2, valid_ref_asymmetry, 's-', color='orange', label='Reference Asymmetry', linewidth=2, markersize=8)[0]
            
            ax4.set_ylim(0, max(2.0, max(max(valid_unknown_asymmetry), max(valid_ref_asymmetry)) * 1.1))
            
            ax3.set_title('Peak Shape Comparison (FWHM)', fontsize=30)
            ax3.set_xlabel('2θ (deg.)', fontsize=30)
            ax3.set_ylabel('FWHM (deg.)', fontsize=30)
            ax3.grid(True, linestyle='--', alpha=0.3)
            
            max_fwhm = max(max(valid_unknown_fwhm), max(valid_ref_fwhm))
            ax3.set_ylim(0, max(0.5, max_fwhm * 1.5))
            
        else:
            ax3.text(0.5, 0.5, "Insufficient peak shape data for comparison", 
                   ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Peak Shape Comparison', fontsize=12)
            ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, "Peak shape analysis not available", 
               ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Peak Shape Comparison', fontsize=12)
        ax3.axis('off')
    
    title_text = f'XRD Pattern Comparison'
    ax1.set_title(title_text, fontsize=35)
    ax1.set_xlabel('2θ (deg.)', fontsize=30)
    ax1.set_ylabel('Normalized Intensity (%)', fontsize=30)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    y_max = ax1.get_ylim()[1]
    ax1.set_ylim(0, y_max * 1.2)
    
    ax1.legend().set_visible(False)
    
    ax2.set_title('Peak Position Comparison', fontsize=30)
    ax2.set_xlabel('2θ (deg.)', fontsize=30)
    ax2.set_yticks([])  # No y-axis ticks for barcode view
    
    ax2.set_ylim(0, 1.5)  # Extended from default (0, 1) to give more space
    
    ax1.set_xlim(10, 120)
    ax2.set_xlim(10, 120)
    
    handles = []
    labels = []
    
    if scatter1 is not None:
        handles.extend([line1, scatter1, scatter2])
        labels.extend(['Unknown Pattern', 'Unknown Pattern Peaks', f'Reference: {display_name}'])
    else:
        handles.extend([line1, scatter2])
        labels.extend(['Unknown Pattern', f'Reference: {display_name}'])
    
    if unknown_fwhm and ref_fwhm and len(unknown_peak_positions) > 0 and valid_positions:
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        handles.extend(lines3 + lines4)
        labels.extend(labels3 + labels4)
    
    from matplotlib.patches import Rectangle
    invisible = Rectangle((0,0), 0, 0, fill=False, edgecolor='none', visible=False)

    score_handles = [invisible] * 5  # 5 score items
    score_labels = [
        f'Position Score: {position_score:.3f}',
        f'Intensity Score: {intensity_score:.3f}',
        f'Shape Score: {shape_score:.3f}' if shape_score > 0 else None,
        f'Overall Score: {overall_score:.3f}',
        f'Matching Peaks: {len(matching_positions)}/{len(ref_positions)}'
    ]
    score_labels = [label for label in score_labels if label is not None]
    score_handles = score_handles[:len(score_labels)]

    all_handles = handles + score_handles
    all_labels = labels + score_labels

    fig.legend(all_handles, all_labels, loc='lower right', bbox_to_anchor=(0.95, -0.18), 
               fontsize=25, ncol=2, frameon=True, fancybox=True, shadow=True)
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Created enhanced fingerprint comparison plot: {output_path}")
    return True

def create_identification_report(report_path, results):
    with open(report_path, 'w') as f:
        f.write("======================================================\n")
        f.write("            MATERIAL IDENTIFICATION REPORT             \n")
        f.write("======================================================\n\n")
        
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("IDENTIFICATION RESULTS:\n")
        f.write("----------------------\n")
        
        if results:
            top_match = results[0]
            material_name = top_match['material_name']
            overall_score = top_match['overall_score']
            position_score = top_match['position_score']
            intensity_score = top_match['intensity_score']
            shape_score = top_match.get('shape_score', 0.0)
            
            f.write(f"Identified Material: {material_name}\n")
            f.write(f"Overall Match Score: {overall_score:.3f}\n")
            f.write(f"Position Score: {position_score:.3f}\n")
            f.write(f"Intensity Score: {intensity_score:.3f}\n")
            if shape_score > 0:
                f.write(f"Shape Score: {shape_score:.3f}\n")
            f.write("\n")
        else:
            f.write("No matches found in database.\n\n")
        
        f.write("TOP MATCHES:\n")
        for i, match in enumerate(results[:5]):
            f.write(f"{i+1}. {match['material_name']}: Score {match['overall_score']:.3f}\n")
            shape_info = f", Shape Score: {match['shape_score']:.3f}" if match.get('shape_score', 0) > 0 else ""
            f.write(f"   Position Score: {match['position_score']:.3f}, Intensity Score: {match['intensity_score']:.3f}{shape_info}\n")
        f.write("\n")
        
        f.write("NOTES:\n")
        f.write("-----\n")
        f.write("- Identification based on peak position, intensity, and shape analysis\n")
        f.write("- Shape analysis includes FWHM (peak width) and peak asymmetry\n")
        f.write("- Further parameter prediction and refinement will be performed if possible\n")
    
    print(f"Created enhanced identification report: {report_path}")
    return True

def parse_parameter_info_file(model_folder, material_name):

    param_info_files = []
    
    training_result_path = os.path.join(model_folder, 'training_result')
    if os.path.exists(training_result_path):
        for dataset_folder in os.listdir(training_result_path):
            dataset_path = os.path.join(training_result_path, dataset_folder)
            if os.path.isdir(dataset_path):
                for file in os.listdir(dataset_path):
                    if file.startswith('parameter_info_') and file.endswith('.dat'):
                        param_info_files.append(os.path.join(dataset_path, file))
    
    if not param_info_files:
        print(f"No parameter_info files found in model directory: {model_folder}")
        return None
    
    param_info_files.sort(reverse=True)
    param_info_path = param_info_files[0]
    print(f"Using parameter info file: {param_info_path}")
    
    param_info = {
        'param_counts': {},
        'param_types': [],
        'total_params': 0,
        'has_background': False
    }
    
    try:
        with open(param_info_path, 'r') as f:
            content = f.read()
            
            total_match = re.search(r'Total parameters:\s*(\d+)', content)
            if total_match:
                param_info['total_params'] = int(total_match.group(1))
            
            param_counts = {
                'zero': 0,
                'background': 0,
                'lattice parameter': 0,
                'biso': 0,
                'scale factor': 0,
                'u parameter': 0,
                'v parameter': 0,
                'w parameter': 0
            }
            
            for param_type in param_counts.keys():
                count_match = re.search(rf'{param_type}:\s*(\d+)', content)
                if count_match:
                    param_counts[param_type] = int(count_match.group(1))
            
            param_info['param_counts'] = param_counts
            
            param_types = []
            types_section = re.search(r'Parameter types in order:(.*?)(?:\n\n|\Z)', content, re.DOTALL)
            if types_section:
                types_lines = types_section.group(1).strip().split('\n')
                for line in types_lines:
                    if ':' in line:
                        param_type = line.split(':', 1)[1].strip()
                        param_types.append(param_type)
            
            param_info['param_types'] = param_types
            
            bg_line = re.search(r'Background parameter:\s*(.*)', content)
            if bg_line:
                param_info['has_background'] = 'Present' in bg_line.group(1)
                
        print(f"Successfully parsed parameter info with {param_info['total_params']} parameters")
        print(f"Lattice parameters: {param_info['param_counts']['lattice parameter']}")
        print(f"Biso parameters: {param_info['param_counts']['biso']}")
        print(f"Background parameter: {'Present' if param_info['has_background'] else 'Absent/Omitted'}")
        
        return param_info
    
    except Exception as e:
        print(f"Error parsing parameter info file: {e}")
        traceback.print_exc()
        return None

def find_scaling_factors(model_folder, material_name, omit_background=False):

    training_result_path = os.path.join(model_folder, 'training_result')
    
    if not os.path.exists(training_result_path):
        print(f"Training result directory not found: {training_result_path}")
        return None
    
    dataset_folders = [d for d in os.listdir(training_result_path) 
                      if os.path.isdir(os.path.join(training_result_path, d))]
    
    scaling_factors = None
    
    for dataset in dataset_folders:
        dataset_path = os.path.join(training_result_path, dataset)
        
        for file in os.listdir(dataset_path):
            if file.startswith('cnn_training_results_') and file.endswith('.dat'):
                log_path = os.path.join(dataset_path, file)
                
                with open(log_path, 'r') as f:
                    content = f.read()
                    
                    scaling_match = re.search(r'Scaling factors:\s*\[([\d\., ]+)\]', content)
                    if scaling_match:
                        factors_str = scaling_match.group(1)
                        try:
                            scaling_factors = [float(x.strip()) for x in factors_str.split(',')]
                            print(f"Found scaling factors: {scaling_factors}")
                            return scaling_factors
                        except Exception as e:
                            print(f"Error parsing scaling factors: {e}")
    
    print("No scaling factors found in training logs, will use defaults")
    return None

def find_two_theta_info(model_folder, material_name):

    two_theta_info = {
        'min': 5.0,
        'max': 110.0,
        'step': 0.3
    }
    
    try:
        refinement_path = os.path.join(model_folder, 'refinement_result')
        if os.path.exists(refinement_path):
            for dataset_folder in os.listdir(refinement_path):
                dataset_path = os.path.join(refinement_path, dataset_folder)
                if os.path.isdir(dataset_path):
                    for exp_folder in os.listdir(dataset_path):
                        exp_path = os.path.join(dataset_path, exp_folder)
                        if os.path.isdir(exp_path):
                            info_files = glob.glob(os.path.join(exp_path, '*_experimental_data_2theta_info.dat'))
                            for info_file in info_files:
                                with open(info_file, 'r') as f:
                                    line = f.readline().strip()
                                    initial_match = re.search(r'Initial=(\d+\.?\d*)', line)
                                    final_match = re.search(r'Final=(\d+\.?\d*)', line)
                                    step_match = re.search(r'Step=(\d+\.?\d*)', line)
                                    
                                    if initial_match and final_match and step_match:
                                        two_theta_info['min'] = float(initial_match.group(1))
                                        two_theta_info['max'] = float(final_match.group(1))
                                        two_theta_info['step'] = float(step_match.group(1))
                                        print(f"Found two-theta info: {two_theta_info} from {info_file}")
                                        return two_theta_info
        
        training_path = os.path.join(model_folder, 'training_result')
        if os.path.exists(training_path):
            for dataset_folder in os.listdir(training_path):
                dataset_path = os.path.join(training_path, dataset_folder)
                if os.path.isdir(dataset_path):
                    for file in os.listdir(dataset_path):
                        if '_2theta_info.dat' in file or '_2theta_param_info.dat' in file:
                            info_file = os.path.join(dataset_path, file)
                            with open(info_file, 'r') as f:
                                line = f.readline().strip()
                                initial_match = re.search(r'Initial=(\d+\.?\d*)', line)
                                final_match = re.search(r'Final=(\d+\.?\d*)', line)
                                step_match = re.search(r'Step=(\d+\.?\d*)', line)
                                
                                if initial_match and final_match and step_match:
                                    two_theta_info['min'] = float(initial_match.group(1))
                                    two_theta_info['max'] = float(final_match.group(1))
                                    two_theta_info['step'] = float(step_match.group(1))
                                    print(f"Found two-theta info: {two_theta_info} from {info_file}")
                                    return two_theta_info
        
        for root, dirs, files in os.walk(model_folder):
            for file in files:
                if '_2theta_info.dat' in file or '_experimental_data_2theta_info.dat' in file:
                    info_file = os.path.join(root, file)
                    with open(info_file, 'r') as f:
                        line = f.readline().strip()
                        initial_match = re.search(r'Initial=(\d+\.?\d*)', line)
                        final_match = re.search(r'Final=(\d+\.?\d*)', line)
                        step_match = re.search(r'Step=(\d+\.?\d*)', line)
                        
                        if initial_match and final_match and step_match:
                            two_theta_info['min'] = float(initial_match.group(1))
                            two_theta_info['max'] = float(final_match.group(1))
                            two_theta_info['step'] = float(step_match.group(1))
                            print(f"Found two-theta info: {two_theta_info} from {info_file}")
                            return two_theta_info
        
        print(f"Could not find two-theta info, using defaults: {two_theta_info}")
        return two_theta_info
        
    except Exception as e:
        print(f"Error finding two-theta info: {e}")
        traceback.print_exc()
        return two_theta_info

def prepare_data_for_prediction(ttheta_values, intensity_values, material_info, two_theta_info):

    try:
        import torch
        
        training_tth_min = two_theta_info['min']
        training_tth_max = two_theta_info['max']
        training_step = two_theta_info['step']
        
        print(f"Using two-theta range: min={training_tth_min}, max={training_tth_max}, step={training_step}")
        
        new_twotheta = np.arange(training_tth_min, training_tth_max + training_step, training_step)
        
        arr_tth = np.array(ttheta_values)
        arr_yobs = np.array(intensity_values)
        sort_idx = np.argsort(arr_tth)
        arr_tth = arr_tth[sort_idx]
        arr_yobs = arr_yobs[sort_idx]
        
        print(f"Interpolating to {len(new_twotheta)} points using two-theta range: {training_tth_min} to {training_tth_max} with step {training_step}")
        interp_y = np.interp(new_twotheta, arr_tth, arr_yobs)
        
        intens = torch.FloatTensor(interp_y)
        max_val = torch.max(intens)
        intens /= max_val
        intens -= torch.mean(intens)
        intens = intens.unsqueeze(0).unsqueeze(0)
        
        return intens, new_twotheta, interp_y
    
    except Exception as e:
        print(f"Error preparing data for prediction: {e}")
        traceback.print_exc()
        return None, None, None

def predict_parameters(material_name, ttheta_values, intensity_values, catalog, output_dir):

    try:
        import torch
        
        param_dir = os.path.join(output_dir, "parameter_refinement")
        os.makedirs(param_dir, exist_ok=True)
        
        if material_name not in catalog:
            print(f"Error: Material '{material_name}' not found in catalog")
            return None
        
        material_info = catalog[material_name]
        model_path = material_info.get("model_path")
        model_folder = material_info.get("model_folder")
        
        if not model_path or not os.path.exists(model_path):
            print(f"Error: Model path for {material_name} not found: {model_path}")
            return None
        
        model_folder_path = os.path.join(SAVED_MODELS_DIR, model_folder)
        print(f"Found model for {material_name}: {model_path}")
        
        param_info = parse_parameter_info_file(model_folder_path, material_name)
        
        if not param_info:
            print(f"Error: Could not parse parameter info for {material_name}")
            return None
        
        two_theta_info = find_two_theta_info(model_folder_path, material_name)
        
        lattice_count = param_info['param_counts']['lattice parameter']
        biso_count = param_info['param_counts']['biso']
        omit_background = not param_info['has_background']
        total_params = param_info['total_params']
        
        scaling_factors = find_scaling_factors(model_folder_path, material_name, omit_background)
        
        if not scaling_factors:
            scaling_factors = []
            for param_type in param_info['param_types']:
                if 'zero' in param_type:
                    scaling_factors.append(100)
                elif 'background' in param_type:
                    scaling_factors.append(0.1)
                elif 'lattice parameter' in param_type:
                    scaling_factors.append(1.0)
                elif 'biso' in param_type:
                    scaling_factors.append(1.0)  
                elif 'scale factor' in param_type:
                    scaling_factors.append(100000)
                elif 'u parameter' in param_type:
                    scaling_factors.append(10.0)
                elif 'v parameter' in param_type:
                    scaling_factors.append(10.0)
                elif 'w parameter' in param_type:
                    scaling_factors.append(10.0)
                else:
                    scaling_factors.append(1.0)
        
        data_tensor, new_twotheta, interp_y = prepare_data_for_prediction(
            ttheta_values, intensity_values, material_info, two_theta_info
        )
        
        if data_tensor is None:
            print("Error: Failed to prepare data for prediction")
            return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        from utils import CNN
        
        model = CNN(
            output_dim=total_params, 
            param_info=param_info
        ).to(device)
        
        print(f"Created model with output_dim={total_params}, lattice_count={lattice_count}, biso_count={biso_count}")
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        model.eval()
        
        data_tensor = data_tensor.to(device)
        
        with torch.no_grad():
            predictions = model(data_tensor)
            
            if isinstance(scaling_factors, list):
                scaling_factors = torch.FloatTensor(scaling_factors).to(device)
            
            if scaling_factors is not None:
                if scaling_factors.size(0) != predictions.size(1):
                    print(f"Warning: Scaling factors size ({scaling_factors.size(0)}) doesn't match predictions size ({predictions.size(1)})")
                    
                    if scaling_factors.size(0) < predictions.size(1):
                        padding = torch.ones(predictions.size(1) - scaling_factors.size(0), device=device)
                        scaling_factors = torch.cat([scaling_factors, padding])
                    else:
                        scaling_factors = scaling_factors[:predictions.size(1)]
                
                predictions = (predictions / scaling_factors) / 10.0
        
        pred_values = predictions.cpu().numpy()[0]
        
        param_names = []
        for param_type in param_info['param_types']:
            if 'zero' in param_type:
                param_names.append('Zero')
            elif 'background' in param_type:
                param_names.append('Background')
            elif 'lattice parameter a' in param_type:
                param_names.append('Lattice a')
            elif 'lattice parameter b' in param_type:
                param_names.append('Lattice b')
            elif 'lattice parameter c' in param_type:
                param_names.append('Lattice c')
            elif 'biso_' in param_type:
                atom_type = param_type.split('_', 1)[1] if '_' in param_type else ''
                param_names.append(f'Biso {atom_type}')
            elif 'scale factor' in param_type:
                param_names.append('Scale')
            elif 'u parameter' in param_type:
                param_names.append('U')
            elif 'v parameter' in param_type:
                param_names.append('V')
            elif 'w parameter' in param_type:
                param_names.append('W')
            else:
                param_names.append(f'Param{len(param_names)+1}')
        
        while len(param_names) < len(pred_values):
            param_names.append(f'Param{len(param_names)+1}')
        if len(param_names) > len(pred_values):
            param_names = param_names[:len(pred_values)]
        
        predicted_params = {}
        for name, value in zip(param_names, pred_values):
            predicted_params[name] = value
            
        predicted_params["_material_name"] = material_name
        predicted_params["_model_path"] = model_path
        predicted_params["_crystal_structure"] = param_info['param_counts']['lattice parameter']
        predicted_params["_omit_background"] = omit_background
        
        param_file = os.path.join(param_dir, f"{material_name}_refined_parameters.dat")
        with open(param_file, 'w') as f:
            for name, value in zip(param_names, pred_values):
                f.write(f"{name:<12} = {value:12.6f}\n")
            
            f.write("\n# Additional Information\n")
            f.write(f"# Material: {material_name}\n")
            crystal_system = "cubic"
            if param_info['param_counts']['lattice parameter'] == 1:
                crystal_system = "cubic"
            elif param_info['param_counts']['lattice parameter'] == 2:
                crystal_system = "tetragonal"
            elif param_info['param_counts']['lattice parameter'] == 3:
                crystal_system = "orthorhombic"
            elif param_info['param_counts']['lattice parameter'] == 4:
                crystal_system = "monoclinic"
            elif param_info['param_counts']['lattice parameter'] == 6:
                crystal_system = "triclinic"
            
            f.write(f"# Crystal Structure: {crystal_system}\n")
            f.write(f"# Model: {os.path.basename(model_path)}\n")

        print("\n" + "=" * 50)
        print(f"  PREDICTED PARAMETERS FOR {material_name.upper()}")
        print("=" * 50)
        print(f"Crystal System: {crystal_system}")
        print(f"Model: {os.path.basename(model_path)}")
        print("-" * 50)
        print(f"{'Parameter':<15} {'Value':>12}")
        print("-" * 50)
        for name, value in zip(param_names, pred_values):
            print(f"{name:<15} {value:12.6f}")
        print("=" * 50 + "\n")

        print(f"Saved predicted parameters to {param_file}")
        
        column_file = os.path.join(param_dir, f"{material_name}_experimental_data_column_for_plot.dat")
        with open(column_file, 'w') as f:
            for tval, yval in zip(new_twotheta, interp_y):
                f.write(f"{tval:.4f} {yval:.4f}\n")
        
        print(f"Saved interpolated pattern to {column_file}")
        
        return predicted_params
        
    except Exception as e:
        print(f"Error predicting parameters: {e}")
        traceback.print_exc()
        return None

def main():
    args = parse_arguments()
    
    input_file = args.file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    catalog = load_material_catalog()
    print(f"Loaded catalog with {len(catalog)} materials")
    
    dat_format = detect_dat_format(input_file)
    ttheta_values, intensity_values = read_xrd_pattern(input_file, dat_format)
    if not ttheta_values or not intensity_values:
        print("Error: Failed to read pattern data")
        sys.exit(1)
    
    match_results = match_pattern_to_materials(ttheta_values, intensity_values, catalog)
    if not match_results:
        print("Error: No matches found in database")
        sys.exit(1)
    
    top_match = match_results[0]
    material_name = top_match["material_name"]
    print(f"\nTop match: {material_name} (Score: {top_match['overall_score']:.3f})")
    
    print(f"Position Score: {top_match['position_score']:.3f}, Intensity Score: {top_match['intensity_score']:.3f}", end="")
    if top_match.get('shape_score', 0) > 0:
        print(f", Shape Score: {top_match['shape_score']:.3f}")
    else:
        print("")
    
    output_dir = os.path.join(IDENTIFIED_DIR, f"unknown_{material_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_path = os.path.join(output_dir, "pattern_comparison.pdf")
    create_fingerprint_comparison_plot(ttheta_values, intensity_values, match_results, comparison_path)
    
    source_dir = os.path.join(output_dir, "source_files")
    os.makedirs(source_dir, exist_ok=True)
    shutil.copy2(input_file, os.path.join(source_dir, os.path.basename(input_file)))
    
    report_path = os.path.join(output_dir, "identification_report.dat")
    create_identification_report(report_path, match_results)
    
    print("\n=== Starting Parameter Prediction and Refinement ===")
    predicted_params = predict_parameters(material_name, ttheta_values, intensity_values, catalog, output_dir)
        
    print("\nIdentification and refinement completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())