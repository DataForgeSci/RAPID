#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import shutil
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)

def determine_crystal_structure(out_path):

    crystal_structure = "cubic"  # Default
    space_group = None
    
    if not os.path.exists(out_path):
        return crystal_structure, space_group
    
    cubic_patterns = [r'F\s*m\s*-3\s*m', r'P\s*m\s*-3\s*m', r'I\s*m\s*-3\s*m', r'I\s*a\s*-3',
                      r'P\s*n\s*-3', r'F\s*d\s*-3\s*m', r'P\s*4_32']
    tetragonal_patterns = [r'P\s*4/m', r'I\s*4/m', r'P\s*4', r'I\s*4', r'P\s*4/n',
                          r'P\s*42/m', r'I\s*4_1/a', r'P\s*-4']
    hexagonal_patterns = [r'P\s*6/m', r'P\s*6_3/m', r'P\s*-6', r'P\s*6']
    orthorhombic_patterns = [r'P\s*mm', r'P\s*nn', r'C\s*mm', r'I\s*mm', r'F\s*mm',
                            r'P\s*2_12_12_1', r'P\s*nma', r'C\s*mcm', r'P\s*bcn']
    monoclinic_patterns = [r'P\s*2/m', r'P\s*2_1/m', r'C\s*2/m', r'P\s*2_1/c', r'P\s*2/c',
                          r'P\s*2_1', r'C\s*c']
    triclinic_patterns = [r'P\s*-1', r'P\s*1']
    trigonal_patterns = [r'R\s*-3', r'R\s*3', r'P\s*3', r'P\s*-3']
    
    with open(out_path, 'r') as f:
        content = f.read()
        
        system_match = re.search(r'Crystal\s+System\s*[=:]\s*(\w+)', content, re.IGNORECASE)
        if system_match:
            system = system_match.group(1).lower()
            if 'cubic' in system:
                crystal_structure = 'cubic'
            elif 'tetra' in system:
                crystal_structure = 'tetragonal'
            elif 'ortho' in system:
                crystal_structure = 'orthorhombic'
            elif 'mono' in system:
                crystal_structure = 'monoclinic'
            elif 'tri' in system and 'clinic' in system:
                crystal_structure = 'triclinic'
            elif 'tri' in system:
                crystal_structure = 'trigonal'
            elif 'hex' in system:
                crystal_structure = 'hexagonal'
        
        sg_match = re.search(r'Space\s+Group\s*[=:]\s*([A-Za-z0-9_/\-\s]+)', content, re.IGNORECASE)
        if sg_match:
            space_group = sg_match.group(1).strip()
            
            if space_group:
                for pattern in cubic_patterns:
                    if re.search(pattern, space_group):
                        crystal_structure = 'cubic'
                        break
                for pattern in tetragonal_patterns:
                    if re.search(pattern, space_group):
                        crystal_structure = 'tetragonal'
                        break
                for pattern in orthorhombic_patterns:
                    if re.search(pattern, space_group):
                        crystal_structure = 'orthorhombic'
                        break
                for pattern in monoclinic_patterns:
                    if re.search(pattern, space_group):
                        crystal_structure = 'monoclinic'
                        break
                for pattern in triclinic_patterns:
                    if re.search(pattern, space_group):
                        crystal_structure = 'triclinic'
                        break
                for pattern in hexagonal_patterns:
                    if re.search(pattern, space_group):
                        crystal_structure = 'hexagonal'
                        break
                for pattern in trigonal_patterns:
                    if re.search(pattern, space_group):
                        crystal_structure = 'trigonal'
                        break
        
        if crystal_structure == "cubic":
            has_a = re.search(r'\b[aA]\s*=\s*([-\+0-9\.]+)', content) is not None
            has_b = re.search(r'\b[bB]\s*=\s*([-\+0-9\.]+)', content) is not None
            has_c = re.search(r'\b[cC]\s*=\s*([-\+0-9\.]+)', content) is not None
            
            has_non90_alpha = False
            has_non90_beta = False
            has_non90_gamma = False
            alpha_match = re.search(r'\balpha\s*=\s*([-\+0-9\.]+)', content, re.IGNORECASE)
            beta_match = re.search(r'\bbeta\s*=\s*([-\+0-9\.]+)', content, re.IGNORECASE)
            gamma_match = re.search(r'\bgamma\s*=\s*([-\+0-9\.]+)', content, re.IGNORECASE)
            
            if alpha_match:
                try:
                    alpha_val = float(alpha_match.group(1))
                    has_non90_alpha = abs(alpha_val - 90.0) > 0.1
                except:
                    pass
            if beta_match:
                try:
                    beta_val = float(beta_match.group(1))
                    has_non90_beta = abs(beta_val - 90.0) > 0.1
                except:
                    pass
            if gamma_match:
                try:
                    gamma_val = float(gamma_match.group(1))
                    has_non90_gamma = abs(gamma_val - 90.0) > 0.1
                except:
                    pass
            
            if has_a and has_b and has_c:
                a_val = re.search(r'\b[aA]\s*=\s*([-\+0-9\.]+)', content).group(1)
                b_val = re.search(r'\b[bB]\s*=\s*([-\+0-9\.]+)', content).group(1)
                c_val = re.search(r'\b[cC]\s*=\s*([-\+0-9\.]+)', content).group(1)
                
                try:
                    a = float(a_val)
                    b = float(b_val)
                    c = float(c_val)
                    
                    if abs(a - b) < 0.01 and abs(a - c) < 0.01:
                        if has_non90_alpha and has_non90_beta and has_non90_gamma:
                            if abs(alpha_val - beta_val) < 0.01 and abs(alpha_val - gamma_val) < 0.01:
                                crystal_structure = 'trigonal'
                            else:
                                crystal_structure = 'triclinic'
                        elif has_non90_alpha or has_non90_beta or has_non90_gamma:
                            crystal_structure = 'rhombohedral'
                        else:
                            crystal_structure = 'cubic'
                    elif abs(a - b) < 0.01 and abs(a - c) > 0.01:
                        if has_non90_gamma:
                            crystal_structure = 'hexagonal'
                        else:
                            crystal_structure = 'tetragonal'
                    elif abs(a - b) > 0.01 and abs(a - c) > 0.01 and abs(b - c) > 0.01:
                        if not (has_non90_alpha or has_non90_beta or has_non90_gamma):
                            crystal_structure = 'orthorhombic'
                        elif has_non90_beta and not (has_non90_alpha or has_non90_gamma):
                            crystal_structure = 'monoclinic'
                        else:
                            crystal_structure = 'triclinic'
                except:
                    pass
    # Commented out unnecessary lines
    # print(f"Detected crystal structure: {crystal_structure}")
    # if space_group:
    #     print(f"Space group: {space_group}")
    
    return crystal_structure, space_group

def parse_fullprof_out_file(out_path):

    crystal_structure, space_group = determine_crystal_structure(out_path)
    
    result = {
        "rwp": None, "rp": None, "re": None, "chi2": None,
        "zero": None, "background": None, "scale": None,
        "U": None, "V": None, "W": None,
        "biso": None,               
        "status": "OK",
        "crystal_structure": crystal_structure,
        "space_group": space_group,
        
        "atom_biso": {},

        "a": None, "b": None, "c": None,
        "alpha": None, "beta": None, "gamma": None,
    }
    
    if not os.path.isfile(out_path):
        result["status"] = "NO_OUT"
        return result

    found_bragg_section = False
    with open(out_path, "r") as ff:
        lines = ff.readlines()

    for ln in lines:
        if re.search(r"^\s*==>\s*RELIABILITY FACTORS FOR POINTS WITH BRAGG CONTRIBUTIONS FOR PATTERN", ln, re.IGNORECASE):
            found_bragg_section = True
            continue

        if found_bragg_section and "=> Conventional Rietveld Rp,Rwp,Re and Chi2:" in ln:
            parts = ln.split(":")
            if len(parts) >= 2:
                numbers_part = parts[1]
                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?", numbers_part)
                if len(nums) >= 4:
                    try:
                        result["rp"] = float(nums[0])
                        result["rwp"] = float(nums[1])
                        result["re"] = float(nums[2])
                        result["chi2"] = float(nums[3])
                    except:
                        pass

        if "Zero-point:" in ln:
            z_nums = re.findall(r"Zero-point:\s*([-\+0-9\.]+)", ln)
            if z_nums:
                try:
                    result["zero"] = float(z_nums[0])
                except:
                    pass

        if re.search(r"\ba\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE):
            try:
                result["a"] = float(re.search(r"\ba\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE).group(1))
            except:
                pass
        if re.search(r"\bb\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE):
            try:
                result["b"] = float(re.search(r"\bb\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE).group(1))
            except:
                pass
        if re.search(r"\bc\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE):
            try:
                result["c"] = float(re.search(r"\bc\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE).group(1))
            except:
                pass

        if re.search(r"\balpha\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE):
            try:
                result["alpha"] = float(re.search(r"\balpha\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE).group(1))
            except:
                pass
        if re.search(r"\bbeta\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE):
            try:
                result["beta"] = float(re.search(r"\bbeta\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE).group(1))
            except:
                pass
        if re.search(r"\bgamma\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE):
            try:
                result["gamma"] = float(re.search(r"\bgamma\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE).group(1))
            except:
                pass

        atom_biso_match = re.search(r"Atom:\s*(\S+).*Biso\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE)
        if atom_biso_match:
            try:
                atom_name = atom_biso_match.group(1)
                biso_val = float(atom_biso_match.group(2))
                result["atom_biso"][atom_name] = biso_val
                if result["biso"] is None:
                    result["biso"] = biso_val
            except:
                pass

        if "Scale factor =" in ln:
            scnums = re.findall(r"Scale factor\s*=\s*([-\+0-9\.]+)", ln)
            if scnums:
                try:
                    result["scale"] = float(scnums[0])
                except:
                    pass

        if "Background coefficients:" in ln:
            pass

        if re.search(r"\bU\s*\=\s*", ln):
            unums = re.findall(r"\bU\s*\=\s*([-\+0-9\.]+)", ln)
            if unums:
                try:
                    result["U"] = float(unums[0])
                except:
                    pass
        if re.search(r"\bV\s*\=\s*", ln):
            vnums = re.findall(r"\bV\s*\=\s*([-\+0-9\.]+)", ln)
            if vnums:
                try:
                    result["V"] = float(vnums[0])
                except:
                    pass
        if re.search(r"\bW\s*\=\s*", ln):
            wnums = re.findall(r"\bW\s*\=\s*([-\+0-9\.]+)", ln)
            if wnums:
                try:
                    result["W"] = float(wnums[0])
                except:
                    pass

    has_negative_biso = any(b < 0 for b in result["atom_biso"].values())
    if has_negative_biso or (result["biso"] is not None and result["biso"] < 0):
        result["status"] = "NEG_BISO"

    if result["rwp"] is None:
        result["status"] = "NO_RWP"

    return result


def parse_prf_bragg_positions(prf_file):
    if not os.path.isfile(prf_file):
        return [], None

    ref_positions = []
    with open(prf_file, "r") as ff:
        lines = ff.readlines()

    for line in lines:
        ln = line.strip()
        if "(" in ln:  
            parts = ln.split()
            try:
                val = float(parts[0])
                ref_positions.append(val)
            except:
                pass

    if not ref_positions:
        return [], None
    return ref_positions, ref_positions[0]

def read_prf_data(prf_file):
    if not os.path.isfile(prf_file):
        return None
        
    data = []
    start_idx = -1
    with open(prf_file, 'r') as f:
        lines = f.readlines()

    for i, raw in enumerate(lines):
        if re.search(r'^\s*2Theta\s+Yobs\s+Ycal', raw):
            start_idx = i + 1
            break
            
    if start_idx < 0:
        return None
        
    for ln in lines[start_idx:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) >= 3:
            try:
                x = float(parts[0])
                yobs = float(parts[1])
                ycal = float(parts[2])
                data.append([x, yobs, ycal])
            except:
                pass
                
    if not data:
        return None
    return np.array(data)

def find_zoomed_region(cnn_data, soln_data, peak_number="default", crystal_structure="cubic", zoom_width="auto"):

    avg_fp = 28.4
    
    peaks = []
    intensities = []
    
    if cnn_data is not None:
        for i in range(1, len(cnn_data) - 1):
            if (cnn_data[i, 1] > cnn_data[i-1, 1] and 
                cnn_data[i, 1] > cnn_data[i+1, 1] and 
                cnn_data[i, 1] > 50):  # Threshold to avoid noise
                peaks.append(cnn_data[i, 0])  # 2theta position
                intensities.append(cnn_data[i, 1])  # Intensity
    
    if not peaks and soln_data is not None:
        for i in range(1, len(soln_data) - 1):
            if (soln_data[i, 1] > soln_data[i-1, 1] and 
                soln_data[i, 1] > soln_data[i+1, 1] and 
                soln_data[i, 1] > 50):  # Threshold to avoid noise
                peaks.append(soln_data[i, 0])
                intensities.append(soln_data[i, 1])
    
    selected_peak_index = 0
    if peaks and intensities:
        sorted_peaks = [x for _, x in sorted(zip(intensities, peaks), reverse=True)]
        
        if peak_number == "default" or not peaks:
            avg_fp = sorted_peaks[0]
            selected_peak_index = 0
        else:
            try:
                peak_idx = int(peak_number) - 1
                if 0 <= peak_idx < len(sorted_peaks):
                    avg_fp = sorted_peaks[peak_idx]
                    selected_peak_index = peak_idx
                else:
                    print(f"Warning: Peak number {peak_number} out of range (1-{len(sorted_peaks)}). Using first peak.")
                    avg_fp = sorted_peaks[0]
                    selected_peak_index = 0
            except ValueError:
                print(f"Warning: Invalid peak number '{peak_number}'. Using first peak.")
                avg_fp = sorted_peaks[0]
                selected_peak_index = 0
    
    if zoom_width.lower() != "auto":
        try:
            x_half_width = float(zoom_width) / 2.0
            print(f"Using custom zoom width: {zoom_width} degrees")
            return avg_fp, x_half_width, selected_peak_index
        except:
            print(f"Invalid zoom width '{zoom_width}'. Using structure-dependent defaults.")
    
    if crystal_structure == "cubic":
        x_half_width = 0.1  # Narrow for cubic (sharp peaks)
    elif crystal_structure in ["tetragonal", "hexagonal", "trigonal"]:
        x_half_width = 0.15  # Slightly wider
    elif crystal_structure == "orthorhombic":
        x_half_width = 0.2  # Medium width (possible peak splitting)
    elif crystal_structure in ["monoclinic", "triclinic"]:
        x_half_width = 0.25  # Wider for low symmetry (more complex patterns)
    else:
        x_half_width = 0.15  # Default
    
    return avg_fp, x_half_width, selected_peak_index

def create_comparison_plot(cnn_data, soln_data, output_path, material_name=None, 
                          peak_number="default", crystal_structure="cubic", zoom_width="auto"):
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    if cnn_data is not None:
        ax.plot(cnn_data[:,0], cnn_data[:,1], 'ro', markersize=3, mfc='none', label='Observed Data', linestyle='')
        ax.plot(cnn_data[:,0], cnn_data[:,2], 'k--', label='CNN ML Refinement', marker='o',
               markersize=3, markerfacecolor='none', linewidth=1)
        
    if soln_data is not None:
        ax.plot(soln_data[:,0], soln_data[:,2], 'b--', label='Solution Refinement', marker='o',
               markersize=3, markerfacecolor='none', linewidth=1)
    
    avg_fp, x_half_width, selected_peak_index = find_zoomed_region(
        cnn_data, soln_data, peak_number, crystal_structure, zoom_width)
    x_low = avg_fp - x_half_width
    x_high = avg_fp + x_half_width
    
    y_max = 0
    y_min = float('inf')
    
    if cnn_data is not None:
        mask = (cnn_data[:,0] >= x_low) & (cnn_data[:,0] <= x_high)
        if np.any(mask):
            y_max = max(y_max, np.max(cnn_data[mask,1]), np.max(cnn_data[mask,2]))
            y_min = min(y_min, np.min(cnn_data[mask,1]), np.min(cnn_data[mask,2]))
    
    if soln_data is not None:
        mask = (soln_data[:,0] >= x_low) & (soln_data[:,0] <= x_high)
        if np.any(mask):
            y_max = max(y_max, np.max(soln_data[mask,2]))
            y_min = min(y_min, np.min(soln_data[mask,2]))
    
    if y_min == float('inf'):
        y_min = 0
        
    y_max *= 1.05
    y_min *= 0.95
    
    axins = inset_axes(ax, width="30%", height="30%", loc="upper right")
    
    if cnn_data is not None:
        axins.plot(cnn_data[:,0], cnn_data[:,1], 'ro', markersize=3, mfc='none', linestyle='')
        axins.plot(cnn_data[:,0], cnn_data[:,2], 'k--', marker='o',
                  markersize=3, markerfacecolor='none', linewidth=1)
    
    if soln_data is not None:
        axins.plot(soln_data[:,0], soln_data[:,2], 'b--', marker='o',
                  markersize=3, markerfacecolor='none', linewidth=1)
    
    axins.set_xlim(x_low, x_high)
    axins.set_ylim(y_min, y_max)
    
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="none")
    
    if peak_number == "default":
        inset_label = f"<Bragg reflection #1>"
    else:
        inset_label = f"<Bragg reflection #{selected_peak_index + 1}>"
    
    axins.text(
        0.5, 0.95,
        inset_label,
        transform=axins.transAxes,
        ha='center',  # Centered text
        va='top',
        fontsize=8,  # Smaller font size
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', pad=2.0),
        zorder=5  # Ensure text appears above the data
    )
    
    ax.set_xlabel(u"2θ (deg.)")
    ax.set_ylabel("Intensity")
    
    if material_name:
        if soln_data is not None:
            ax.set_title(f"Rietveld Refinement Comparison Overplot ({material_name})")
        else:
            ax.set_title(f"CNN Rietveld Refinement ({material_name})")
    else:
        if soln_data is not None:
            ax.set_title(f"Rietveld Refinement Comparison Overplot ({crystal_structure} structure)")
        else:
            ax.set_title(f"CNN Rietveld Refinement ({crystal_structure} structure)")
    
    ax.legend(loc='upper left')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def format_cell_params_for_report(result, crystal_structure):
    cell_params = {}
    
    if result["a"] is not None:
        cell_params["a"] = "{:10.6f}".format(result["a"])
    else:
        cell_params["a"] = "    NA"
    
    if crystal_structure in ["orthorhombic", "monoclinic", "triclinic"]:
        if result["b"] is not None:
            cell_params["b"] = "{:10.6f}".format(result["b"])
        else:
            cell_params["b"] = "    NA"
    else:
        cell_params["b"] = cell_params["a"]
    
    if crystal_structure in ["tetragonal", "hexagonal", "orthorhombic", "monoclinic", "triclinic"]:
        if result["c"] is not None:
            cell_params["c"] = "{:10.6f}".format(result["c"])
        else:
            cell_params["c"] = "    NA"
    else:
        cell_params["c"] = cell_params["a"]
    
    if crystal_structure in ["triclinic", "trigonal"]:
        if result["alpha"] is not None:
            cell_params["alpha"] = "{:8.4f}".format(result["alpha"])
        else:
            cell_params["alpha"] = "  NA"
    else:
        cell_params["alpha"] = "90.0000"
    
    if crystal_structure in ["monoclinic", "triclinic"]:
        if result["beta"] is not None:
            cell_params["beta"] = "{:8.4f}".format(result["beta"])
        else:
            cell_params["beta"] = "  NA"
    else:
        cell_params["beta"] = "90.0000"
    
    if crystal_structure in ["hexagonal", "triclinic"]:
        if result["gamma"] is not None:
            cell_params["gamma"] = "{:8.4f}".format(result["gamma"])
        else:
            cell_params["gamma"] = "  NA"
    elif crystal_structure == "hexagonal":
        cell_params["gamma"] = "120.0000"
    else:
        cell_params["gamma"] = "90.0000"
    
    return cell_params

def create_atom_biso_report(result, all_atoms):
    biso_report = {}
    
    for atom in all_atoms:
        if atom in result["atom_biso"] and result["atom_biso"][atom] is not None:
            biso_report[atom] = "{:8.5f}".format(result["atom_biso"][atom])
        else:
            biso_report[atom] = "   NA"
    
    return biso_report

def create_analysis_report(refine_root, output_file, cnn_results, soln_results, omit_background=False):

    all_atoms_raw = set()
    if cnn_results and "atom_biso" in cnn_results:
        all_atoms_raw.update(cnn_results["atom_biso"].keys())
    if soln_results and "atom_biso" in soln_results:
        all_atoms_raw.update(soln_results["atom_biso"].keys())

    def unify_atom_name(name):
        if len(name) == 0:
            return name
        return name[0].upper() + name[1:].lower()

    atom_map = {}  
    for raw_name in all_atoms_raw:
        lower_key = raw_name.lower()
        if lower_key not in atom_map:
            atom_map[lower_key] = unify_atom_name(raw_name)

    all_atoms = sorted(atom_map.values())

    def create_atom_biso_report_normalized(results):

        if (not results) or ("atom_biso" not in results):
            return {}
        raw_dict = results["atom_biso"]
        out_report = {}
        for lc_key, disp_key in atom_map.items():

            found_val = None
            for real_atom_name, val in raw_dict.items():
                if real_atom_name.lower() == lc_key:
                    found_val = val
                    break
            if found_val is not None:
                out_report[disp_key] = f"{found_val:8.5f}"
            else:
                out_report[disp_key] = "   NA"
        return out_report

    cnn_biso_report = create_atom_biso_report_normalized(cnn_results)
    soln_biso_report = create_atom_biso_report_normalized(soln_results)

    crystal_structure = (
        soln_results.get("crystal_structure", "") if soln_results else ""
    ) or (
        cnn_results.get("crystal_structure", "") if cnn_results else ""
    ) or "cubic"


    def format_cell_params_for_report(result, cryst_sys):
        cp = {}
        if not result:
            cp["a"] = "    NA"
            cp["b"] = "    NA"
            cp["c"] = "    NA"
            cp["alpha"] = "  NA"
            cp["beta"] = "  NA"
            cp["gamma"] = "  NA"
            return cp

        def safe_fmt(v):
            return "{:10.6f}".format(v) if (v is not None) else "    NA"

        cp["a"] = safe_fmt(result.get("a"))
        if cryst_sys in ["orthorhombic", "monoclinic", "triclinic"]:
            cp["b"] = safe_fmt(result.get("b"))
        else:
            cp["b"] = cp["a"]
        if cryst_sys in ["tetragonal", "hexagonal", "orthorhombic", "monoclinic", "triclinic"]:
            cp["c"] = safe_fmt(result.get("c"))
        else:
            cp["c"] = cp["a"]
        if cryst_sys in ["triclinic", "trigonal"]:
            cp["alpha"] = "{:8.4f}".format(result["alpha"]) if result.get("alpha") is not None else "  NA"
        else:
            cp["alpha"] = "90.0000"

        if cryst_sys in ["monoclinic", "triclinic"]:
            cp["beta"] = "{:8.4f}".format(result["beta"]) if result.get("beta") is not None else "  NA"
        else:
            cp["beta"] = "90.0000"

        if cryst_sys in ["hexagonal", "triclinic"]:
            if result.get("gamma") is not None:
                cp["gamma"] = "{:8.4f}".format(result["gamma"])
            else:
                cp["gamma"] = "  NA"
        elif cryst_sys == "hexagonal":
            cp["gamma"] = "120.0000"
        else:
            cp["gamma"] = "90.0000"
        return cp

    cnn_cell_params = format_cell_params_for_report(cnn_results, crystal_structure)
    soln_cell_params = format_cell_params_for_report(soln_results, crystal_structure)

    with open(output_file, "w") as f:
        f.write("# Rietveld Refinement Analysis Report\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")
        f.write(f"# Crystal Structure: {crystal_structure}\n\n")

        if soln_results and "space_group" in soln_results and soln_results["space_group"]:
            f.write(f"# Space Group: {soln_results['space_group']}\n\n")
        elif cnn_results and "space_group" in cnn_results and cnn_results["space_group"]:
            f.write(f"# Space Group: {cnn_results['space_group']}\n\n")

        f.write("# Explanation:\n")
        f.write("#   RefinementType: CNN ML or Solution refinement\n")
        f.write("#   Rp(%)         : Conventional Rietveld Rp in percent (Bragg-Only)\n")
        f.write("#   Rwp(%)        : Conventional Rietveld Rwp in percent (Bragg-Only)\n")
        f.write("#   Re(%)         : Expected Rwp in percent (Bragg-Only)\n")
        f.write("#   Chi²          : Goodness of fit (Rwp/Re)² (Bragg-Only)\n")
        f.write("#   ZeroShift     : final zero shift (degrees)\n")
        f.write("#   Background    : first background coefficient\n")

        for atom in all_atoms:
            f.write(f"#   Biso_{atom}      : isotropic displacement parameter for {atom}\n")

        f.write("#   Scale         : final scale factor\n")

        if crystal_structure == "cubic":
            f.write("#   a             : cubic lattice parameter (Å)\n")
        elif crystal_structure in ["tetragonal", "hexagonal"]:
            f.write("#   a, c          : tetragonal/hexagonal lattice parameters (Å)\n")
        elif crystal_structure == "orthorhombic":
            f.write("#   a, b, c       : orthorhombic lattice parameters (Å)\n")
        elif crystal_structure == "monoclinic":
            f.write("#   a, b, c, beta : monoclinic lattice parameters (Å, degrees)\n")
        elif crystal_structure == "triclinic":
            f.write("#   a, b, c, alpha, beta, gamma : triclinic lattice parameters (Å, degrees)\n")
        elif crystal_structure == "trigonal":
            f.write("#   a, alpha      : trigonal lattice parameters (Å, degrees)\n")

        f.write("#   U, V, W       : final peak shape parameters\n")
        f.write("#   Status        : parse result from .out (OK, NO_OUT, etc.)\n")
        f.write("#   FirstBragg    : first bragg peak 2theta from .prf\n\n")

        f.write("# IMPORTANT: All R-factors (Rp, Rwp, Re, Chi²) are from 'Points with Bragg Contributions' section:\n")
        f.write("#   * These values are calculated only over the data points overlapping actual Bragg peaks\n")
        f.write("#   * Conventional Rietveld Rp: Basic ratio of total absolute difference to total intensity\n")
        f.write("#   * Rwp: Weighted difference ratio, typically your main index\n")
        f.write("#   * Re: Ideal Rwp if only noise remains\n")
        f.write("#   * Chi²: (Rwp/Re)², checking if you match the data within random error\n\n")

        if omit_background:
            f.write("# NOTE: Background parameter was omitted from CNN training/prediction\n")
            f.write("#       Background values for CNN results are taken from original PCR files\n\n")

        header = "# RefinementType        Rp(%)      Rwp(%)     Re(%)      Chi²      ZeroShift     Background    "
        for atom in all_atoms:
            header += f"Biso_{atom:<6} "
        header += "Scale      "

        if crystal_structure == "cubic":
            header += "a          "
        elif crystal_structure in ["tetragonal", "hexagonal"]:
            header += "a          c          "
        elif crystal_structure == "orthorhombic":
            header += "a          b          c          "
        elif crystal_structure == "monoclinic":
            header += "a          b          c          beta      "
        elif crystal_structure == "triclinic":
            header += "a          b          c          alpha     beta      gamma     "
        elif crystal_structure == "trigonal":
            header += "a          alpha      "

        header += "U        V        W     FirstBragg   Status"
        f.write(header + "\n")

        def format_line(label, res, biso_dict, cell_params):

            if not res:
                return None

            if res.get("rp") is not None:
                rp_str = f"{res['rp']:8.3f}"
            else:
                rp_str = "   NA"

            if res.get("rwp") is not None:
                rwp_str = f"{res['rwp']:8.3f}"
            else:
                rwp_str = "   NA"

            if res.get("re") is not None:
                re_str = f"{res['re']:8.3f}"
            else:
                re_str = "   NA"

            if res.get("chi2") is not None:
                chi2_str = f"{res['chi2']:8.3f}"
            else:
                chi2_str = "   NA"

            zero_val = res.get("zero")
            if zero_val is not None:
                zero_str = f"{zero_val:11.5f}"
            else:
                zero_str = "     NA"

            bg_val = res.get("background")
            if bg_val is not None:
                bg_str = f"{bg_val:11.5f}"
            else:
                bg_str = "     NA"

            biso_cols = ""
            for atom in all_atoms:
                biso_cols += f"{biso_dict.get(atom,'   NA')}  "

            scale_val = res.get("scale")
            if scale_val is not None:
                scale_str = f"{scale_val:10.6f}"
            else:
                scale_str = "    NA"

            if crystal_structure == "cubic":
                lattice_str = f"{cell_params['a']}  "
            elif crystal_structure in ["tetragonal", "hexagonal"]:
                lattice_str = f"{cell_params['a']}  {cell_params['c']}  "
            elif crystal_structure == "orthorhombic":
                lattice_str = f"{cell_params['a']}  {cell_params['b']}  {cell_params['c']}  "
            elif crystal_structure == "monoclinic":
                lattice_str = f"{cell_params['a']}  {cell_params['b']}  {cell_params['c']}  {cell_params['beta']}  "
            elif crystal_structure == "triclinic":
                lattice_str = (f"{cell_params['a']}  {cell_params['b']}  {cell_params['c']}  "
                               f"{cell_params['alpha']}  {cell_params['beta']}  {cell_params['gamma']}  ")
            elif crystal_structure == "trigonal":
                lattice_str = f"{cell_params['a']}  {cell_params['alpha']}  "
            else:
                lattice_str = ""

            def safe_par(k):
                v = res.get(k)
                return f"{v:8.5f}" if (v is not None) else "   NA"

            U_str = safe_par("U")
            V_str = safe_par("V")
            W_str = safe_par("W")

            if "first_bragg_2theta" in res and res["first_bragg_2theta"] is not None:
                fb_str = f"{res['first_bragg_2theta']:10.5f}"
            else:
                fb_str = "    NA"

            st = res.get("status", "UNKNOWN")

            line = (f"{label:<20s}  {rp_str}  {rwp_str}  {re_str}  {chi2_str}  {zero_str}  "
                    f"{bg_str}  {biso_cols}{scale_str}  {lattice_str}"
                    f"{U_str}  {V_str}  {W_str}  {fb_str}  {st}")
            return line


        cnn_line = format_line(
            "CNN_ML",
            cnn_results,
            cnn_biso_report,
            cnn_cell_params
        )
        if cnn_line is not None:
            f.write(cnn_line + "\n")


        soln_line = format_line(
            "Solution",
            soln_results,
            soln_biso_report,
            soln_cell_params
        )
        if soln_line is not None:
            f.write(soln_line + "\n")


def parse_pcr_parameters(pcr_path):
    params = {
        "zero": None, "biso": None, "scale": None, 
        "U": None, "V": None, "W": None, "background": None,
        "a": None, "b": None, "c": None,
        "alpha": None, "beta": None, "gamma": None,
        "atom_biso": {},  
    }
    
    if not os.path.isfile(pcr_path):
        return params
        
    with open(pcr_path, 'r') as f:
        lines = f.readlines()
        
    crystal_structure = "cubic"  
    
    for i, line in enumerate(lines):
        if "Zero" in line and "Code" in line and "SyCos" in line:
            if i + 1 < len(lines):
                parts = lines[i+1].split()
                if parts:
                    try:
                        params["zero"] = float(parts[0])
                    except:
                        pass
        
        if "Background coefficients/codes" in line:
            if i + 1 < len(lines):
                bg_line = lines[i+1].strip()
                parts = bg_line.split()
                if parts and len(parts) > 0:
                    try:
                        params["background"] = float(parts[0])  
                    except:
                        pass
        
        if line.strip().startswith("!     a") or "!" in line and "a" in line and "b" in line and "c" in line:
            if i + 1 < len(lines):
                data = lines[i+1].split()
                if len(data) >= 1:
                    try:
                        params["a"] = float(data[0])
                        
                        if len(data) >= 2:
                            params["b"] = float(data[1])
                        if len(data) >= 3:
                            params["c"] = float(data[2])
                        
                        if len(data) >= 4:
                            params["alpha"] = float(data[3])
                        if len(data) >= 5:
                            params["beta"] = float(data[4])
                        if len(data) >= 6:
                            params["gamma"] = float(data[5])
                            
                        a, b, c = params["a"], params["b"], params["c"]
                        alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
                        
                        if a and b and c:
                            if abs(a-b) < 0.001 and abs(a-c) < 0.001:
                                if alpha and beta and gamma and (
                                    abs(alpha-90) > 0.1 or abs(beta-90) > 0.1 or abs(gamma-90) > 0.1):
                                    if abs(alpha-beta) < 0.001 and abs(alpha-gamma) < 0.001:
                                        crystal_structure = "trigonal"
                                    else:
                                        crystal_structure = "triclinic"
                                else:
                                    crystal_structure = "cubic"
                            elif abs(a-b) < 0.001 and abs(a-c) > 0.001:
                                if gamma and abs(gamma-120) < 0.1:
                                    crystal_structure = "hexagonal"
                                else:
                                    crystal_structure = "tetragonal"
                            elif abs(a-b) > 0.001 and abs(a-c) > 0.001 and abs(b-c) > 0.001:
                                if beta and abs(beta-90) > 0.1 and (
                                    not alpha or not gamma or (abs(alpha-90) < 0.1 and abs(gamma-90) < 0.1)):
                                    crystal_structure = "monoclinic"
                                elif (alpha and abs(alpha-90) > 0.1) or (
                                    gamma and abs(gamma-90) > 0.1):
                                    crystal_structure = "triclinic"
                                else:
                                    crystal_structure = "orthorhombic"
                    except:
                        pass
                        
        if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+', line):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    atom_name = parts[0]
                    biso_val = float(parts[5])
                    
                    params["atom_biso"][atom_name] = biso_val
                    
                    if params["biso"] is None:
                        params["biso"] = biso_val
                except:
                    pass
                    
        if "Scale" in line and "Shape1" in line:
            if i + 1 < len(lines):
                data = lines[i+1].split()
                if data:
                    try:
                        scale_str = data[0]
                        if 'E' in scale_str or 'e' in scale_str:
                            params["scale"] = float(scale_str)
                        else:
                            params["scale"] = float(scale_str)
                    except:
                        pass
        
        if "U" in line and "V" in line and "W" in line and "X" in line:
            if i + 1 < len(lines):
                data = lines[i+1].split()
                if len(data) >= 3:
                    try:
                        params["U"] = float(data[0])
                        params["V"] = float(data[1])
                        params["W"] = float(data[2])
                    except:
                        pass
    
    params["crystal_structure"] = crystal_structure
    
    return params

def read_ml_inputs(ml_file):
    if not os.path.isfile(ml_file):
        print(f"Error: {ml_file} not found. Exiting.")
        sys.exit(1)

    lines = []
    with open(ml_file, 'r') as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            lines.append(t)
            if len(lines) == 9:  
                break

    if len(lines) < 5:
        print("Error: ML_inputs file must have at least 5 non-comment lines.")
        sys.exit(1)

    line1 = lines[0].strip()
    model_name = "default_model"
    if ";" in line1:
        parts = line1.split(";", 1)
        model_name = parts[1].strip() if len(parts) > 1 else "default_model"

    dataset_line = lines[2].strip()
    dataset_folders = []
    if ";" in dataset_line:
        parts = dataset_line.split(";", 1)
        leftover = parts[1].strip()
        splitted = leftover.split(',')
        for ds in splitted:
            ds_clean = ds.strip()
            if ds_clean:
                dataset_folders.append(ds_clean)
    else:
        dataset_folders.append(dataset_line)

    if not dataset_folders:
        print("No dataset folder name found on line3.")
        sys.exit(1)
    
    experimental_line = lines[3].strip()
    experiment_base = os.path.splitext(experimental_line)[0]
    
    material_name = None
    if experimental_line and experimental_line.lower() != "n":
        if ',' in experimental_line:
            first_dat = experimental_line.split(',')[0].strip()
            material_name = os.path.splitext(first_dat)[0]
        else:
            material_name = experiment_base
    
    omit_background = False
    if len(lines) >= 7:
        omit_background = lines[6].lower().strip() == 'y'
    
    peak_number = "default"
    if len(lines) >= 8:
        peak_number = lines[7].lower().strip()
    
    zoom_width = "auto"
    if len(lines) >= 9:
        zoom_width_val = lines[8].lower().strip()
        if zoom_width_val == 'auto':
            zoom_width = zoom_width_val
        else:
            try:
                width = float(zoom_width_val)
                if width > 0:  # Must be positive
                    zoom_width = str(width)
            except:
                pass

    return model_name, dataset_folders, experiment_base, omit_background, material_name, peak_number, zoom_width

def main():
    if len(sys.argv) != 2:
        print("Usage: python Rietveld_Refinement_step3.py <ML_inputs_file>")
        sys.exit(1)

    ml_inputs = sys.argv[1]
    model_name, dataset_folders, experiment_base, omit_background, material_name, peak_number, zoom_width = read_ml_inputs(ml_inputs)

    for dataset_name in dataset_folders:
        print(f"\nProcessing refinement analysis for dataset: {dataset_name}")
        print(f"Background parameter omitted: {omit_background}")
        print(f"Material name: {material_name or 'Not specified'}")
        print(f"Peak number for zooming: {peak_number}")
        print(f"Zoom width: {zoom_width}")
        
        refine_root = os.path.join(
            "saved_models",
            model_name,
            "refinement_result",
            dataset_name,
            experiment_base,
            "Rietveld_Refinement"
        )

        if not os.path.exists(refine_root):
            print(f"Error: Refinement directory not found: {refine_root}")
            continue  
            
        cnn_dir = os.path.join(refine_root, "CNN_ML_refinement")
        soln_dir = os.path.join(refine_root, "solution_refinement")
        
        if not os.path.exists(cnn_dir) or not os.path.exists(soln_dir):
            print(f"Error: Refinement result subdirectories missing for {dataset_name}")
            continue  

        output_dir = os.path.join(refine_root, "output_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        refined_params_file = os.path.join(refine_root, "..", f"{experiment_base}_refined_parameters.dat")
        refined_params = {}
        if os.path.exists(refined_params_file):
            with open(refined_params_file, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, val = line.strip().split('=')
                        refined_params[key.strip()] = float(val.strip())

        cnn_results = None
        soln_results = None

        cnn_out_files = [f for f in os.listdir(cnn_dir) if f.endswith('.out')]
        if cnn_out_files:
            cnn_out_path = os.path.join(cnn_dir, cnn_out_files[0])
            cnn_results = parse_fullprof_out_file(cnn_out_path)
            if cnn_results and refined_params:
                cnn_results["zero"] = refined_params.get("Zero")
                if not omit_background:
                    cnn_results["background"] = refined_params.get("Background")
                
                for key in refined_params:
                    if key.startswith("Biso "):
                        atom_name = key.split("Biso ", 1)[1]
                        if atom_name and "atom_biso" in cnn_results:
                            cnn_results["atom_biso"][atom_name] = refined_params[key]
                
                cnn_results["scale"] = refined_params.get("Scale")
                cnn_results["U"] = refined_params.get("U")
                cnn_results["V"] = refined_params.get("V")
                cnn_results["W"] = refined_params.get("W")
                
                if "Lattice a" in refined_params:
                    cnn_results["a"] = refined_params["Lattice a"]
                if "Lattice b" in refined_params:
                    cnn_results["b"] = refined_params["Lattice b"]
                if "Lattice c" in refined_params:
                    cnn_results["c"] = refined_params["Lattice c"]
                if "Alpha" in refined_params:
                    cnn_results["alpha"] = refined_params["Alpha"]
                if "Beta" in refined_params:
                    cnn_results["beta"] = refined_params["Beta"]
                if "Gamma" in refined_params:
                    cnn_results["gamma"] = refined_params["Gamma"]
            
            cnn_prf_files = [f for f in os.listdir(cnn_dir) if f.endswith('.prf')]
            if cnn_prf_files:
                prf_path = os.path.join(cnn_dir, cnn_prf_files[0])
                bragg_pos, first_bragg = parse_prf_bragg_positions(prf_path)
                cnn_results["bragg_positions"] = bragg_pos
                cnn_results["first_bragg_2theta"] = first_bragg
        
        soln_out_files = [f for f in os.listdir(soln_dir) if f.endswith('.out')]
        soln_pcr_files = [f for f in os.listdir(soln_dir) if f.endswith('.pcr')]
        
        if soln_out_files:
            soln_out_path = os.path.join(soln_dir, soln_out_files[0])
            soln_results = parse_fullprof_out_file(soln_out_path)
            
            if soln_pcr_files:
                soln_pcr_path = os.path.join(soln_dir, soln_pcr_files[0])
                pcr_params = parse_pcr_parameters(soln_pcr_path)
                if soln_results:
                    soln_results["zero"] = pcr_params["zero"]
                    soln_results["background"] = pcr_params["background"]
                    soln_results["scale"] = pcr_params["scale"]
                    soln_results["U"] = pcr_params["U"]
                    soln_results["V"] = pcr_params["V"]
                    soln_results["W"] = pcr_params["W"]
                    
                    soln_results["a"] = pcr_params["a"]
                    soln_results["b"] = pcr_params["b"]
                    soln_results["c"] = pcr_params["c"]
                    soln_results["alpha"] = pcr_params["alpha"]
                    soln_results["beta"] = pcr_params["beta"]
                    soln_results["gamma"] = pcr_params["gamma"]
                    
                    for atom, biso in pcr_params["atom_biso"].items():
                        if "atom_biso" not in soln_results:
                            soln_results["atom_biso"] = {}
                        soln_results["atom_biso"][atom] = biso
                    
                    if omit_background and cnn_results and pcr_params["background"] is not None:
                        cnn_results["background"] = pcr_params["background"]
                        print(f"Using PCR background value ({pcr_params['background']}) for CNN results (background was omitted from training)")
                    
                    if "crystal_structure" in pcr_params and "crystal_structure" not in soln_results:
                        soln_results["crystal_structure"] = pcr_params["crystal_structure"]
                        
            soln_prf_files = [f for f in os.listdir(soln_dir) if f.endswith('.prf')]
            if soln_prf_files:
                prf_path = os.path.join(soln_dir, soln_prf_files[0])
                bragg_pos, first_bragg = parse_prf_bragg_positions(prf_path)
                soln_results["bragg_positions"] = bragg_pos
                soln_results["first_bragg_2theta"] = first_bragg
        
        crystal_structure = (
            soln_results.get("crystal_structure", "") or 
            cnn_results.get("crystal_structure", "") or 
            "cubic"  
        )
        
        if soln_results:
            soln_results["crystal_structure"] = crystal_structure
        if cnn_results:
            cnn_results["crystal_structure"] = crystal_structure
        
        cnn_data = None
        soln_data = None
        
        if cnn_prf_files:
            cnn_prf_path = os.path.join(cnn_dir, cnn_prf_files[0])
            cnn_data = read_prf_data(cnn_prf_path)
        
        if soln_prf_files:
            soln_prf_path = os.path.join(soln_dir, soln_prf_files[0])
            soln_data = read_prf_data(soln_prf_path)
        
        report_path = os.path.join(output_dir, "analysis_report.dat")
        create_analysis_report(refine_root, report_path, cnn_results, soln_results, omit_background)
        print(f"Created analysis report => {report_path}")
        
        plot_path = os.path.join(output_dir, "refinement_comparison.png")
        create_comparison_plot(cnn_data, soln_data, plot_path, material_name, peak_number, crystal_structure, zoom_width)
        print(f"Created comparison plot => {plot_path}")

        cnn_plot_path = os.path.join(output_dir, "cnn_refinement.png")
        create_comparison_plot(cnn_data, None, cnn_plot_path, material_name, peak_number, crystal_structure, zoom_width)
        print(f"Created CNN-only plot => {cnn_plot_path}")

if __name__ == "__main__":
    main()