#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
                normalized_atom = normalize_atom_name(atom_name)
                result["atom_biso"][normalized_atom] = biso_val
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

        # background
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

def normalize_atom_name(atom_name):

    if not atom_name:
        return atom_name
    
    normalized = atom_name[0].upper() + atom_name[1:].lower()
    return normalized

def get_unified_atom_names(atom_biso_dict):

    normalized_dict = {}
    
    for atom_name, biso_val in atom_biso_dict.items():
        normalized_name = normalize_atom_name(atom_name)
        normalized_dict[normalized_name] = biso_val
    
    return normalized_dict

def parse_pcr_parameters(pcr_path):
    params = {
        "zero": None, "biso": None, "scale": None, 
        "U": None, "V": None, "W": None, "background": None,
        "a": None, "b": None, "c": None,
        "alpha": None, "beta": None, "gamma": None,
        "atom_biso": {},  # Store atom-specific Biso values
    }
    
    if not os.path.isfile(pcr_path):
        return params
        
    with open(pcr_path, 'r') as f:
        lines = f.readlines()
        
    crystal_structure = "cubic"  # Default
    
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
                        params["background"] = float(parts[0])  # First coefficient
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
                            # a=b=c for cubic, a=b≠c for tetragonal, a≠b≠c for orthorhombic
                            if abs(a-b) < 0.001 and abs(a-c) < 0.001:
                                # a=b=c
                                if alpha and beta and gamma and (
                                    abs(alpha-90) > 0.1 or abs(beta-90) > 0.1 or abs(gamma-90) > 0.1):
                                    # Non-90 angles
                                    if abs(alpha-beta) < 0.001 and abs(alpha-gamma) < 0.001:
                                        crystal_structure = "trigonal"
                                    else:
                                        crystal_structure = "triclinic"
                                else:
                                    crystal_structure = "cubic"
                            elif abs(a-b) < 0.001 and abs(a-c) > 0.001:
                                # a=b≠c
                                if gamma and abs(gamma-120) < 0.1:
                                    crystal_structure = "hexagonal"
                                else:
                                    crystal_structure = "tetragonal"
                            elif abs(a-b) > 0.001 and abs(a-c) > 0.001 and abs(b-c) > 0.001:
                                # a≠b≠c
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
                    
                    normalized_atom = normalize_atom_name(atom_name)
                    
                    params["atom_biso"][normalized_atom] = biso_val
                    
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

def parse_prf_bragg_positions(prf_file):
    if not os.path.isfile(prf_file):
        return [], None

    ref_positions = []
    with open(prf_file, "r") as ff:
        lines = ff.readlines()

    for line in lines:
        ln = line.strip()
        if "(" in ln:  # reflection line
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

def find_zoomed_region(data, peak_number="default", crystal_structure="cubic", zoom_width="auto"):

    avg_fp = 28.4
    
    peaks = []
    intensities = []
    
    if data is not None:
        for i in range(1, len(data) - 1):
            if (data[i, 1] > data[i-1, 1] and 
                data[i, 1] > data[i+1, 1] and 
                data[i, 1] > 50):  # Threshold to avoid noise
                peaks.append(data[i, 0])  # 2theta position
                intensities.append(data[i, 1])  # Intensity
    
    selected_peak_index = 0
    if peaks and intensities:
        sorted_indices = np.argsort(intensities)[::-1]  # sort indices in descending order
        sorted_peaks = [peaks[i] for i in sorted_indices]
        
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
    
    if zoom_width != "auto":
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

def create_refinement_plot(data, output_path, material_name, crystal_structure="cubic", peak_number="default", zoom_width="auto"):

    if data is None:
        print("Error: No data for plotting")
        return False
    
    twotheta = data[:,0]
    yobs = data[:,1]
    ycalc = data[:,2]
    if data.shape[1] > 3:
        ydiff = data[:,3]
    else:
        ydiff = yobs - ycalc
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    yobs_max = np.max(yobs)
    ydiff_min = np.min(ydiff)
    
    diff_offset = -1500  # Place diff curve below the main plot
    diff_adjusted = ydiff + diff_offset
    
    ax.plot(twotheta, ycalc, 'k--', linewidth=1.0, marker='o', markersize=3, markerfacecolor='none', label='Ycalc')
    ax.scatter(twotheta, yobs, edgecolors='red', facecolors='none', marker='o', s=15, linewidth=0.5, label='Yobs')
    ax.plot(twotheta, diff_adjusted, 'b-', linewidth=1.0, label='Yobs-Ycalc')
    
    bragg_positions = []
    bragg_dir = os.path.dirname(output_path)
    cnn_dir = os.path.join(os.path.dirname(bragg_dir), "CNN_ML_refinement")
    cnn_prf_files = glob.glob(os.path.join(cnn_dir, "*.prf"))
    
    if cnn_prf_files:
        prf_file = cnn_prf_files[0]
        bragg_positions, _ = parse_prf_bragg_positions(prf_file)
    
    bragg_line_bottom = -500
    bragg_line_height = 200
    
    if bragg_positions:
        for pos in bragg_positions:
            if pos >= min(twotheta) and pos <= max(twotheta):
                ax.vlines(pos, 
                          ymin=bragg_line_bottom, 
                          ymax=bragg_line_bottom + bragg_line_height, 
                          colors='magenta', 
                          linestyles='-', 
                          linewidth=0.7)
        
        ax.vlines([], [], [], color='magenta', linestyle='-', linewidth=0.7, label='Bragg Positions')
    
    rwp = None
    cnn_out_files = glob.glob(os.path.join(cnn_dir, "*.out"))
    if cnn_out_files:
        results = parse_fullprof_out_file(cnn_out_files[0])
        if results and "rwp" in results and results["rwp"] is not None:
            rwp = results["rwp"]
            ax.plot([], [], ' ', label=f"Rwp = {rwp:.2f}%")
    
    max_idx = np.argmax(yobs)
    max_intensity_pos = twotheta[max_idx]
    
    closest_bragg = max_intensity_pos
    if bragg_positions:
        closest_bragg = bragg_positions[np.argmin(np.abs(np.array(bragg_positions) - max_intensity_pos))]
    
    axins = inset_axes(ax, width="23.3%", height="23.3%", loc="upper right")
    
    zoom_width = 1.5
    x_low = closest_bragg - zoom_width/2
    x_high = closest_bragg + zoom_width/2
    
    mask = (twotheta >= x_low) & (twotheta <= x_high)
    if np.any(mask):
        axins.plot(twotheta[mask], ycalc[mask], 'k--', marker='o', markersize=3, markerfacecolor='none', linewidth=0.7)
        axins.scatter(twotheta[mask], yobs[mask], edgecolors='red', facecolors='none', marker='o', s=15, linewidth=0.5)
        
        y_min = 0  # Start at zero for better aesthetics
        y_max = max(np.max(yobs[mask]), np.max(ycalc[mask])) * 1.1  # 10% margin
        
        axins.set_xlim(x_low, x_high)
        axins.set_ylim(y_min, y_max)
        
        axins.grid(True, linestyle='--', alpha=0.3)
        
        axins.set_xlabel('2-theta (deg.)')
        axins.set_ylabel('Intensity (arb. units)')
        
        axins.text(0.5, 0.95, "<strongest Bragg peak>", 
                   transform=axins.transAxes, ha='center', va='top',
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', pad=2.0))
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    legend = ax.legend(loc='upper left', fontsize=10)
    
    ax.set_xlabel('2-theta (deg.)')
    ax.set_ylabel('Intensity (arb. units)')
    
    title = f"Rietveld Refinement Results"
    if material_name:
        title += f"\n{material_name} - {crystal_structure.capitalize()} Structure"
    ax.set_title(title, fontsize=14)
    
    diff_lowest = diff_offset + np.min(ydiff)
    ax.set_ylim(diff_lowest - 200, yobs_max * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created refinement plot: {output_path}")
    return True

def create_analysis_report(output_file, results, material_name):
    crystal_structure = results.get("crystal_structure", "cubic")
    space_group = results.get("space_group", "")
    
    if not output_file.endswith(".dat"):
        output_file = output_file.replace(".txt", ".dat")
        if not output_file.endswith(".dat"):
            output_file += ".dat"
    
    unified_atoms = get_unified_atom_names(results.get("atom_biso", {}))
    
    all_atoms = sorted(unified_atoms.keys())
    
    cell_params = {}
    if results["a"] is not None:
        cell_params["a"] = "{:10.6f}".format(results["a"])
    else:
        cell_params["a"] = "    NA"
    
    if crystal_structure in ["orthorhombic", "monoclinic", "triclinic"]:
        if results["b"] is not None:
            cell_params["b"] = "{:10.6f}".format(results["b"])
        else:
            cell_params["b"] = "    NA"
    else:
        cell_params["b"] = cell_params["a"]
    
    if crystal_structure in ["tetragonal", "hexagonal", "orthorhombic", "monoclinic", "triclinic"]:
        if results["c"] is not None:
            cell_params["c"] = "{:10.6f}".format(results["c"])
        else:
            cell_params["c"] = "    NA"
    else:
        cell_params["c"] = cell_params["a"]
    
    if crystal_structure in ["triclinic", "trigonal"]:
        if results["alpha"] is not None:
            cell_params["alpha"] = "{:8.4f}".format(results["alpha"])
        else:
            cell_params["alpha"] = "  NA"
    else:
        cell_params["alpha"] = "90.0000"
    
    if crystal_structure in ["monoclinic", "triclinic"]:
        if results["beta"] is not None:
            cell_params["beta"] = "{:8.4f}".format(results["beta"])
        else:
            cell_params["beta"] = "  NA"
    else:
        cell_params["beta"] = "90.0000"
    
    if crystal_structure in ["hexagonal", "triclinic"]:
        if results["gamma"] is not None:
            cell_params["gamma"] = "{:8.4f}".format(results["gamma"])
        else:
            cell_params["gamma"] = "  NA"
    elif crystal_structure == "hexagonal":
        cell_params["gamma"] = "120.0000"
    else:
        cell_params["gamma"] = "90.0000"
    
    biso_report = {}
    for atom in all_atoms:
        if atom in unified_atoms:
            biso_report[atom] = "{:8.5f}".format(unified_atoms[atom])
        else:
            biso_report[atom] = "   NA"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header and explanation
        f.write("# Rietveld Refinement Analysis Report\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")
        f.write(f"# Crystal Structure: {crystal_structure}\n\n")

        if space_group:
            f.write(f"# Space Group: {space_group}\n\n")

        f.write("# Explanation:\n")
        f.write("#   RefinementType: CNN ML refinement\n")
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

        if results.get("rp") is not None:
            rp_str = f"{results['rp']:8.3f}"
        else:
            rp_str = "   NA"

        if results.get("rwp") is not None:
            rwp_str = f"{results['rwp']:8.3f}"
        else:
            rwp_str = "   NA"

        if results.get("re") is not None:
            re_str = f"{results['re']:8.3f}"
        else:
            re_str = "   NA"

        if results.get("chi2") is not None:
            chi2_str = f"{results['chi2']:8.3f}"
        else:
            chi2_str = "   NA"

        zero_val = results.get("zero")
        if zero_val is not None:
            zero_str = f"{zero_val:11.5f}"
        else:
            zero_str = "     NA"

        bg_val = results.get("background")
        if bg_val is not None:
            bg_str = f"{bg_val:11.5f}"
        else:
            bg_str = "     NA"

        biso_cols = ""
        for atom in all_atoms:
            biso_cols += f"{biso_report.get(atom,'   NA')}  "

        scale_val = results.get("scale")
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

        U_str = f"{results['U']:8.5f}" if results.get("U") is not None else "   NA"
        V_str = f"{results['V']:8.5f}" if results.get("V") is not None else "   NA"
        W_str = f"{results['W']:8.5f}" if results.get("W") is not None else "   NA"

        if "first_bragg_2theta" in results and results["first_bragg_2theta"] is not None:
            fb_str = f"{results['first_bragg_2theta']:10.5f}"
        else:
            fb_str = "    NA"

        st = results.get("status", "UNKNOWN")

        cnn_line = f"CNN_ML{material_name:>15s}  {rp_str}  {rwp_str}  {re_str}  {chi2_str}  {zero_str}  {bg_str}  {biso_cols}{scale_str}  {lattice_str}{U_str}  {V_str}  {W_str}  {fb_str}  {st}"
        f.write(cnn_line + "\n")
    
    print(f"Created analysis report: {output_file}")
    return True

def extract_material_name(identified_dir):
    folder_name = os.path.basename(identified_dir)
    parts = folder_name.split('_')
    if len(parts) < 2:
        return "Unknown"
    
    return parts[1]

def get_refined_parameters(refined_params_file, param_info_file=None):

    refined_params = {}
    
    if os.path.exists(refined_params_file):
        with open(refined_params_file, 'r') as f:
            for line in f:
                if '=' in line:
                    if line.strip().startswith('#'):
                        continue
                    key, val = line.strip().split('=')
                    key = key.strip()
                    val = val.strip()
                    try:
                        refined_params[key] = float(val)
                    except:
                        refined_params[key] = val
    
    if param_info_file and os.path.exists(param_info_file):
        with open(param_info_file, 'r') as f:
            for line in f:
                if ':' in line:
                    if line.strip().startswith('#'):
                        continue
                    key, val = line.strip().split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    if key not in refined_params:
                        try:
                            refined_params[key] = float(val)
                        except:
                            refined_params[key] = val
    
    return refined_params

def main():
    if len(sys.argv) < 2:
        print("Usage: python Rietveld_Refinement_step3_MI.py <identified_material_folder>")
        sys.exit(1)

    identified_dir = sys.argv[1]
    if not os.path.exists(identified_dir):
        print(f"Error: Identified material folder not found: {identified_dir}")
        sys.exit(1)

    material_name = extract_material_name(identified_dir)
    print(f"Processing material: {material_name}")

    refine_root = os.path.join(identified_dir, "Rietveld_Refinement")
    if not os.path.exists(refine_root):
        print(f"Error: Refinement directory not found: {refine_root}")
        sys.exit(1)

    output_dir = os.path.join(refine_root, "analysis_output")
    os.makedirs(output_dir, exist_ok=True)

    cnn_dir = os.path.join(refine_root, "CNN_ML_refinement")
    if not os.path.exists(cnn_dir):
        print(f"Error: CNN refinement directory not found: {cnn_dir}")
        sys.exit(1)

    out_files = glob.glob(os.path.join(cnn_dir, "*.out"))
    if not out_files:
        print(f"Error: No .out file found in {cnn_dir}")
        sys.exit(1)
    
    out_file = out_files[0]
    print(f"Found .out file: {os.path.basename(out_file)}")

    results = parse_fullprof_out_file(out_file)
    
    crystal_structure = results.get("crystal_structure", "cubic")
    print(f"Detected crystal structure: {crystal_structure}")

    prf_files = glob.glob(os.path.join(cnn_dir, "*.prf"))
    if not prf_files:
        print(f"Error: No .prf file found in {cnn_dir}")
        sys.exit(1)
    
    prf_file = prf_files[0]
    print(f"Found .prf file: {os.path.basename(prf_file)}")

    bragg_positions, first_bragg = parse_prf_bragg_positions(prf_file)
    results["bragg_positions"] = bragg_positions
    results["first_bragg_2theta"] = first_bragg
    
    pcr_files = glob.glob(os.path.join(cnn_dir, "*.pcr"))
    if pcr_files:
        pcr_file = pcr_files[0]
        print(f"Found .pcr file: {os.path.basename(pcr_file)}")
        pcr_params = parse_pcr_parameters(pcr_file)
        
        if results.get("background") is None:
            results["background"] = pcr_params.get("background")
        if results.get("scale") is None:
            results["scale"] = pcr_params.get("scale")
        if results.get("U") is None:
            results["U"] = pcr_params.get("U")
        if results.get("V") is None:
            results["V"] = pcr_params.get("V")
        if results.get("W") is None:
            results["W"] = pcr_params.get("W")
            
        for atom, biso in pcr_params.get("atom_biso", {}).items():
            normalized_atom = normalize_atom_name(atom)
            if normalized_atom not in results["atom_biso"]:
                results["atom_biso"][normalized_atom] = biso
    
    refined_params_file = os.path.join(identified_dir, "parameter_refinement", f"{material_name}_refined_parameters.dat")
    if os.path.exists(refined_params_file):
        print(f"Found refined parameters file: {os.path.basename(refined_params_file)}")
        refined_params = get_refined_parameters(refined_params_file)
        
        if "Zero" in refined_params:
            results["zero"] = refined_params["Zero"]
        if "Background" in refined_params:
            results["background"] = refined_params["Background"]
        if "Scale" in refined_params:
            results["scale"] = refined_params["Scale"]
        if "U" in refined_params:
            results["U"] = refined_params["U"]
        if "V" in refined_params:
            results["V"] = refined_params["V"]
        if "W" in refined_params:
            results["W"] = refined_params["W"]
        
        for key in refined_params:
            if key.startswith("Biso "):
                atom_name = key.split("Biso ", 1)[1]
                if atom_name:
                    normalized_atom = normalize_atom_name(atom_name)
                    results["atom_biso"][normalized_atom] = refined_params[key]
        
        if "Lattice a" in refined_params:
            results["a"] = refined_params["Lattice a"]
        if "Lattice b" in refined_params:
            results["b"] = refined_params["Lattice b"]
        if "Lattice c" in refined_params:
            results["c"] = refined_params["Lattice c"]
    
    prf_data = read_prf_data(prf_file)
    
    plot_path = os.path.join(output_dir, f"{material_name}_refinement.png")
    create_refinement_plot(prf_data, plot_path, material_name, crystal_structure)
    
    report_path = os.path.join(output_dir, f"{material_name}_refinement_report.dat")
    create_analysis_report(report_path, results, material_name)
    
    print("\nRietveld refinement analysis completed successfully.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()