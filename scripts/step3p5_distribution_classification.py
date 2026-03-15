# -*- coding: utf-8 -*-
"""
step3p5_distribution_classification.py

Purpose (updated):
  1) Classify each sample folder's Rwp => CLOSE, BOUNDARY, or INVALID.
  2) Log results into classification_report.dat, with columns for
     all user-changed parameters (ZeroShift, Biso, Scale, U, V, W, etc.).
  3) If ratio of CLOSE:BOUNDARY is not ~80:20, do partial re-run logic:
     - Possibly remove some sets or re-adjust param ranges (for all changed parameters).
     - Overwrite the existing sets, or rename them, etc.
  4) Repeat until ratio is near 80/20 or until max iteration is reached.

Updated to better support multiple crystal structures including orthorhombic systems.
"""

import os
import sys
import re
import shutil
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set Arial as default font for all plot elements
import matplotlib
matplotlib.rc('font', family='Arial')

# For zoomed inset
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes,
                                                   mark_inset)

###############################################################################
# Partial re-run logic
###############################################################################

def remove_all_samples(folder_path):
    """
    Remove *all* sample folders so we can start from scratch
    with updated param ranges for the next iteration.
    """
    all_dirs = [
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
        and d.lower().startswith("sample")
    ]
    print("[PartialReRun] remove_all_samples => total={}".format(len(all_dirs)))
    for d in all_dirs:
        try:
            fullp = os.path.join(folder_path, d)
            shutil.rmtree(fullp)
            print("Removed entire folder => {}".format(fullp))
        except:
            pass

def get_lattice_type(input_file):
    """
    Extract the lattice type from inputs.txt
    Returns the lattice type (cubic, orthorhombic, etc.)
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if "11>Enter the lattice type" in line and i+1 < len(lines):
            lattice_type = lines[i+1].strip().lower()
            return lattice_type
    
    # Default to cubic if not found
    return "cubic"

def adjust_tweak_ranges_for_enabled_params(folder_path, input_file, ratio_c=None, i_count=0):
    """
    Simplified function that only adjusts the zero-shift parameter
    while maintaining compatibility with all crystal structures.
    
    Uses the original simple adjustment rules:
    - If too many CLOSE samples (>90%): Multiply zero range by 1.2
    - If too few CLOSE samples (<70%): Divide zero range by 2.0
    - If invalid samples exist: Always halve the zero range
    
    Args:
        folder_path: Path to the working directory
        input_file: Path to the input file
        ratio_c: Percentage of CLOSE samples
        i_count: Count of INVALID samples
    
    Returns:
        Dictionary with only the updated zero_shift_range
    """
    print("[PartialReRun] adjust_tweak_ranges_for_enabled_params => Adjusting zero parameter only.")
    
    # Set default ratio if not provided
    if ratio_c is None:
        ratio_c = 50.0
    
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    zero_shift_key = "7>Input the maximum absolute shift range"
    updated_zero_range = None

    for i, line in enumerate(lines):
        if zero_shift_key in line:
            if (i + 1) < len(lines):
                old_val_str = lines[i+1].strip()
                try:
                    old_val = float(old_val_str)

                    # Apply simple adjustment rules from earlier version
                    if i_count > 0:  # If we have invalid sets
                        new_val = old_val / 2.0
                    elif ratio_c > 90.0:  # Too many CLOSE samples
                        new_val = old_val * 1.2
                    elif ratio_c < 70.0:  # Too few CLOSE samples
                        new_val = old_val / 2.0
                    else:
                        new_val = old_val  # Keep as is if within desired range

                    updated_zero_range = new_val
                    new_lines.append(line)
                    new_lines.append(str(new_val) + "\n")
                    continue
                except:
                    pass
        
        # Keep all other lines unchanged
        new_lines.append(line)

    # Write the updated input file
    with open(input_file, 'w') as f:
        f.writelines(new_lines)

    # Report the updated zero-shift range
    if updated_zero_range is not None:
        print("[PartialReRun] Zero-shift range updated to: {}".format(updated_zero_range))

    # Return dictionary with only the zero_shift_range
    return {"zero_shift_range": updated_zero_range}


def call_step123_with_updated_input(input_file):
    """
    Re-generate or re-run step1->3 for the *new* sets after removing the old.
    """
    cmd1 = 'python scripts/step1_cif2pcr.py "{}"'.format(input_file)
    cmd2 = 'python scripts/step2_pcrfolders_modifiedparameters.py "{}"'.format(input_file)
    cmd3 = 'python scripts/step3_runautofp_fixall.py "{}"'.format(input_file)
    print(" => {}".format(cmd1)); os.system(cmd1)
    print(" => {}".format(cmd2)); os.system(cmd2)
    print(" => {}".format(cmd3)); os.system(cmd3)

###############################################################################
# Parsing logic for .out to gather Rwp, etc.
###############################################################################

def parse_fullprof_out_file(out_path):
    """
    UPDATED to extract Rwp and multiple atom-specific Biso values:
    - We read the line ' => Conventional Rietveld Rp,Rwp,Re and Chi2: ...'
      AFTER we see the line '==> RELIABILITY FACTORS FOR POINTS WITH BRAGG CONTRIBUTIONS FOR PATTERN:'
    - We also track atom-specific Biso values when reported 
    """
    result = {
      "rwp": None, "zero": None, "biso": None,
      "scale": None,
      "U": None, "V": None, "W": None,
      "status": "OK"
    }
    
    # Add dictionary to store atom-specific Biso values
    result["atom_biso"] = {}
    result["atom_names"] = []
    
    if not os.path.isfile(out_path):
        result["status"] = "NO_OUT"
        return result

    found_bragg_section = False
    with open(out_path, "r") as ff:
        lines = ff.readlines()

    # First pass - extract main parameters
    for ln in lines:
        # once we see the line with "REL. FACTORS FOR POINTS WITH BRAGG CONTRIBUTIONS"
        # we set found_bragg_section=True
        if re.search(r"^\s*==>\s*RELIABILITY FACTORS FOR POINTS WITH BRAGG CONTRIBUTIONS FOR PATTERN", ln, re.IGNORECASE):
            found_bragg_section = True
            continue

        if found_bragg_section and "=> Conventional Rietveld Rp,Rwp,Re and Chi2:" in ln:
            # parse
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", ln)
            # example line => "   20.8      50.5      1.80      790.0"
            # so nums could be ["20.8", "50.5", "1.80", "790.0"]
            if len(nums) >= 2:
                try:
                    # second => Rwp
                    result["rwp"] = float(nums[1])
                except:
                    pass

        # also parse zero shift
        if "Zero-point:" in ln:
            z_nums = re.findall(r"Zero-point:\s*([-\+0-9\.]+)", ln)
            if z_nums:
                try:
                    result["zero"] = float(z_nums[0])
                except:
                    pass
                    
        # Parse atom-specific Biso values
        # Format in .out file is typically: "Atom: NAME Type: TYPE ... Biso = X.XXX ..."
        if re.search(r"\bAtom\s*:\s*([A-Za-z0-9]+)", ln, re.IGNORECASE):
            atom_match = re.search(r"\bAtom\s*:\s*([A-Za-z0-9]+)", ln, re.IGNORECASE)
            biso_match = re.search(r"\bBiso\s*=\s*([-\+0-9\.]+)", ln, re.IGNORECASE)
            
            if atom_match and biso_match:
                try:
                    atom_name = atom_match.group(1)
                    biso_val = float(biso_match.group(1))
                    
                    # Store atom's Biso
                    result["atom_biso"][atom_name] = biso_val
                    
                    # Keep track of atom order
                    if atom_name not in result["atom_names"]:
                        result["atom_names"].append(atom_name)
                    
                    # For backward compatibility, use the first atom's Biso
                    if result["biso"] is None:
                        result["biso"] = biso_val
                except:
                    pass
                    
        # Also check for general Biso lines if no atom-specific ones are found
        elif re.search(r"\bBiso\b", ln, re.IGNORECASE) and not result["atom_biso"]:
            mt = re.findall(r"Biso\s*=\s*([-\+0-9\.]+)", ln)
            if mt:
                try:
                    result["biso"] = float(mt[0])
                except:
                    pass
        
        # parse scale factor if present
        if "Scale factor =" in ln:
            scnums = re.findall(r"Scale factor\s*=\s*([-\+0-9\.]+)", ln)
            if scnums:
                try:
                    result["scale"] = float(scnums[0])
                except:
                    pass
        # parse U, V, W if present
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

    # Check for negative Biso (any atom)
    has_negative_biso = False
    for atom_name, biso_val in result["atom_biso"].iteritems():
        if biso_val is not None and biso_val < 0:
            has_negative_biso = True
            break
    
    if has_negative_biso or (result["biso"] is not None and result["biso"] < 0):
        result["status"] = "NEG_BISO"
    if result["rwp"] is None:
        result["status"] = "NO_RWP"
        
    return result

def parse_pcr_file(pcr_path, lattice_type="cubic"):
    """
    Extract parameters from PCR file based on crystal structure
    """
    results = {"zero": None, "biso": None, "scale": None,
               "U": None, "V": None, "W": None}
    
    # Add dictionary for atom-specific Biso values
    results["atom_biso"] = {}
    results["atom_names"] = []  # To preserve atom order
    
    # Add lattice parameters based on structure
    if lattice_type in ["tetragonal", "hexagonal"]:
        results.update({"a": None, "c": None})
    elif lattice_type == "orthorhombic":
        results.update({"a": None, "b": None, "c": None})
    elif lattice_type == "monoclinic":
        results.update({"a": None, "b": None, "c": None, "beta": None})
    elif lattice_type == "triclinic":
        results.update({"a": None, "b": None, "c": None, 
                       "alpha": None, "beta": None, "gamma": None})
    elif lattice_type == "trigonal":
        results.update({"a": None, "alpha": None})
    else:  # cubic
        results.update({"a": None})
    
    if not os.path.isfile(pcr_path):
        return results

    with open(pcr_path, 'r') as f:
        lines = f.readlines()

    for i, ln in enumerate(lines):
        if ln.strip().startswith("!  Zero"):
            if i + 1 < len(lines):
                data = lines[i+1].split()
                try:
                    results["zero"] = float(data[0])
                except:
                    pass
        
        # Find atom positions with Biso values (store all atom Biso values)
        if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', ln):
            parts = ln.split()
            if len(parts) >= 6:
                try:
                    atom_name = parts[0]
                    atom_type = parts[1]
                    biso_val = float(parts[5])
                    
                    # Store atom's Biso in the atom_biso dictionary
                    results["atom_biso"][atom_name] = biso_val
                    
                    # Keep track of atom order in the PCR file
                    if atom_name not in results["atom_names"]:
                        results["atom_names"].append(atom_name)
                    
                    # For backward compatibility, also store the first atom's Biso as "biso"
                    if results["biso"] is None:
                        results["biso"] = biso_val
                except:
                    pass
        
        # Parse scale factor
        if ln.strip().startswith("!  Scale"):
            if i + 1 < len(lines):
                data = lines[i+1].split()
                if data:
                    try:
                        results["scale"] = float(data[0])
                    except:
                        pass
        
        # Parse U, V, W parameters
        if ln.strip().startswith("!       U"):
            if i + 1 < len(lines):
                data = lines[i+1].split()
                if len(data) >= 3:
                    try:
                        results["U"] = float(data[0])
                        results["V"] = float(data[1])
                        results["W"] = float(data[2])
                    except:
                        pass
        
        # Parse lattice parameters
        if ln.strip().startswith("!     a"):
            if i + 1 < len(lines):
                data = lines[i+1].split()
                if len(data) >= 6:
                    try:
                        results["a"] = float(data[0])
                        if lattice_type != "cubic":
                            if lattice_type in ["tetragonal", "hexagonal", "trigonal"]:
                                # For tetragonal/hexagonal/trigonal, a=b≠c
                                results["c"] = float(data[2])
                            elif lattice_type in ["orthorhombic", "monoclinic", "triclinic"]:
                                # For orthorhombic/monoclinic/triclinic, a≠b≠c
                                results["b"] = float(data[1])
                                results["c"] = float(data[2])
                            
                            # For monoclinic, alpha=gamma=90°, beta≠90°
                            if lattice_type == "monoclinic":
                                results["beta"] = float(data[4])
                            
                            # For triclinic, alpha≠beta≠gamma≠90°
                            elif lattice_type == "triclinic":
                                results["alpha"] = float(data[3])
                                results["beta"] = float(data[4])
                                results["gamma"] = float(data[5])
                            
                            # For trigonal, a=b=c, alpha=beta=gamma≠90°
                            elif lattice_type == "trigonal":
                                results["alpha"] = float(data[3])
                    except:
                        pass
    
    return results

###############################################################################
# Additional parsing to retrieve Bragg reflection positions from the .prf file
###############################################################################
def parse_prf_bragg_positions(prf_file):
    """
    Extract the Bragg peak 2θ positions from lines near the end of the .prf,
    where a line contains something like:
       28.4402                  0 (  1  1  1 )  0  1 :  1
    We look for lines with '(' in them (h k l), and parse the first token.

    Returns:
      (ref_positions, first_pos)
      ref_positions: list of all bragg 2theta positions (floats)
      first_pos: the first bragg position or None
    """
    if not os.path.isfile(prf_file):
        return [], None

    with open(prf_file, "r") as ff:
        lines = ff.readlines()

    ref_positions = []
    for line_str in lines:
        ln = line_str.strip()
        if "(" in ln:  # reflection line => parse 2theta from the first token
            parts = ln.split()
            try:
                val = float(parts[0])
                ref_positions.append(val)
            except:
                pass

    if not ref_positions:
        return [], None
    return (ref_positions, ref_positions[0])


def read_prf_data(prf_file):
    """
    Reads the main XY data from .prf (2Theta, Yobs, Ycal) to plot the profile.
    """
    if not os.path.isfile(prf_file):
        return None
    data=[]
    start_idx=-1
    with open(prf_file,'r') as f:
        lines=f.readlines()

    for i,raw in enumerate(lines):
        if re.search(r'^\s*2Theta\s+Yobs\s+Ycal', raw):
            start_idx = i+1
            break
    if start_idx<0:
        return None
    for ln in lines[start_idx:]:
        ln=ln.strip()
        if not ln:
            continue
        parts=ln.split()
        if len(parts)>=3:
            try:
                x=float(parts[0])
                yobs=float(parts[1])
                ycal=float(parts[2])
                data.append([x,yobs,ycal])
            except:
                pass
    if not data:
        return None
    return np.array(data)

def get_run_index(report_path):
    if not os.path.isfile(report_path):
        return 1
    last_run=0
    with open(report_path,'r') as f:
        for ln in f:
            if ln.strip().startswith("Run:"):
                sp=ln.strip().split()
                if len(sp)==2:
                    try:
                        rr=int(sp[1])
                        last_run=max(last_run, rr)
                    except:
                        pass
    return last_run+1

def read_input_parameters(input_file):
    params={}
    if not os.path.isfile(input_file):
        return params
    with open(input_file,'r') as ff:
        lines=ff.readlines()

    # Hard-coded for brevity:
    params["step35_enable"]="Y"
    params["rwp_close"]="30"
    params["rwp_boundary"]="60"
    params["partial_rerun"]="Y"
    params["fig_generate"]="Y"
    params["fig_display"]="N"
    params["lattice_type"] = "cubic"  # Default
    params["zoom_peak"] = "1"  # Default to first Bragg peak
    params["zoom_width"] = "auto"  # Default to auto width
    params["two_theta_range"] = "default"  # Default to 0-100 range

    # Read actual values from input.txt
    for i, line in enumerate(lines):
        if "29>Perform Step 3.5 distribution & classification logic (Y/N)?:" in line and i+1 < len(lines):
            params["step35_enable"] = lines[i+1].strip()
        
        elif "30>Enter Rwp threshold (in %) for a \"close\" fit classification" in line and i+1 < len(lines):
            val = lines[i+1].strip()
            if val.lower() != "default":
                params["rwp_close"] = val
        
        elif "31>Enter max Rwp (in %) for a \"boundary\" fit classification" in line and i+1 < len(lines):
            val = lines[i+1].strip()
            if val.lower() != "default":
                params["rwp_boundary"] = val
        
        elif "32>Allow partial re-run if 80/20 ratio not met? (Y/N)?:" in line and i+1 < len(lines):
            params["partial_rerun"] = lines[i+1].strip()
        
        elif "16>Generate all output figures (Y/N)?" in line and i+1 < len(lines):
            params["fig_generate"] = lines[i+1].strip()
        
        elif "17>Display all output figures (Y/N)?" in line and i+1 < len(lines):
            params["fig_display"] = lines[i+1].strip()
            
        elif "33>Generate classification plots after each re-run? (Y/N):" in line and i+1 < len(lines):
            params["plot_all_runs"] = lines[i+1].strip()
            
        elif "11>Enter the lattice type" in line and i+1 < len(lines):
            params["lattice_type"] = lines[i+1].strip().lower()
            
        elif "35>Select which Bragg reflection peak to zoom into" in line and i+1 < len(lines):
            zoom_peak_val = lines[i+1].strip().lower()
            # Accept either a number or 'auto'
            if zoom_peak_val == 'auto':
                params["zoom_peak"] = zoom_peak_val
            else:
                try:
                    # Convert to integer and validate
                    peak_idx = int(zoom_peak_val)
                    if peak_idx > 0:  # Must be positive
                        params["zoom_peak"] = str(peak_idx)
                except:
                    # Keep default if invalid
                    pass
            
        elif "36>Customize zoom width around the selected peak" in line and i+1 < len(lines):
            zoom_width_val = lines[i+1].strip().lower()
            # Accept either a number or 'auto'
            if zoom_width_val == 'auto':
                params["zoom_width"] = zoom_width_val
            else:
                try:
                    # Convert to float and validate
                    width = float(zoom_width_val)
                    if width > 0:  # Must be positive
                        params["zoom_width"] = str(width)
                except:
                    # Keep default if invalid
                    pass
                    
        elif "37>Enter two-theta range for classification plots" in line and i+1 < len(lines):
            two_theta_range_val = lines[i+1].strip().lower()
            if two_theta_range_val != 'default':
                try:
                    # Try to parse as min,max values
                    range_parts = two_theta_range_val.split(',')
                    if len(range_parts) == 2:
                        min_val = float(range_parts[0].strip())
                        max_val = float(range_parts[1].strip())
                        if min_val < max_val:
                            params["two_theta_range"] = "{},{}".format(min_val, max_val)
                except:
                    # Keep default if invalid
                    pass
                    
        elif "40>Generate classification profile plots (Y/N)?:" in line and i+1 < len(lines):
            params["generate_classification_plots"] = lines[i+1].strip()

        elif "48>Enter DPI for classification profile plots" in line and i+1 < len(lines):
            dpi_val = lines[i+1].strip().lower()
            if dpi_val != 'default':
                try:
                    dpi = int(dpi_val)
                    if dpi > 0:
                        params["plot_dpi"] = str(dpi)
                except:
                    pass

    # If "plot_all_runs" wasn't explicitly defined, use "Y"
    if "plot_all_runs" not in params:
        params["plot_all_runs"] = "Y"

    # If "generate_classification_plots" wasn't defined, default to Y
    if "generate_classification_plots" not in params:
        params["generate_classification_plots"] = "Y"

    # Default DPI to 150 if not specified
    if "plot_dpi" not in params:
        params["plot_dpi"] = "150"

    return params

###############################################################################
def main():
    if len(sys.argv)!=2:
        print("Usage: python step3p5_distribution_classification.py <inputs.txt>")
        sys.exit(1)
    input_file=sys.argv[1]
    if not os.path.isfile(input_file):
        print("Error: not found => {}".format(input_file))
        sys.exit(1)
        
    config=read_input_parameters(input_file)
    if not config.get("step35_enable","N").upper().startswith("Y"):
        print("step3.5 disabled => exit.")
        sys.exit(0)

    # Get lattice type for structure-specific parameter handling
    lattice_type = config.get("lattice_type","cubic").lower()
    print("Crystal structure: {}".format(lattice_type))
    
    # close_thr (30) and bound_thr (60) are default thresholds for classifying 
    # samples by Rwp if the script can't find a "reference Rwp" in sample1
    try:
        close_thr = float(config.get("rwp_close","30"))
    except:
        close_thr=30
    try:
        bound_thr = float(config.get("rwp_boundary","60"))
    except:
        bound_thr=60
    partial_rerun= config.get("partial_rerun","N").upper().startswith("Y")
    fig_generate=  config.get("fig_generate","N").upper().startswith("Y")
    fig_display=   config.get("fig_display","N").upper().startswith("Y")

    # If "plot_all_runs" is "N", we skip figure generation except on final run.
    plot_all_runs = config.get("plot_all_runs","Y").upper().startswith("Y")

    # Get zoom peak selection and width from config
    zoom_peak = config.get("zoom_peak", "1")
    zoom_width = config.get("zoom_width", "auto")

    # Get DPI for plots (default 150)
    try:
        plot_dpi = int(config.get("plot_dpi", "150"))
    except:
        plot_dpi = 150
    print("[step3.5] Using DPI={} for classification plots".format(plot_dpi))

    subfolder_path = os.path.dirname(os.path.abspath(input_file))
    classification_file= os.path.join(subfolder_path,"classification_report.dat")

    max_iter=10
    iteration_count=0
    ratio_c=50.0  # default fallback

    # We'll store reference Rwp from sample1 after we parse in each iteration
    # (In principle, sample1 might appear or vanish, but presumably it always exists.)
    reference_rwp = None

    while True:
        iteration_count+=1

        if iteration_count==1:
            # Iteration 1: Just classify existing samples from batch file's step1-3
            # Do NOT re-run step1-3 since they were already run with the same parameters
            print("[step3.5] Iteration 1: Classifying existing samples (step1-3 already completed by batch)")
        else:
            # Check if partial rerun is disabled - if so, break out after first iteration
            if not partial_rerun:
                print("[step3.5] Partial re-run is disabled (input line 32>N). Stopping after first iteration.")
                break
                
            remove_all_samples(subfolder_path)
            # Pass i_count when calling the parameter tweak function
            updated_ranges = adjust_tweak_ranges_for_enabled_params(
                subfolder_path,
                input_file,
                ratio_c=ratio_c,
                i_count=i_count
            )
            with open(classification_file, "a") as fout:
                fout.write("#\n# Re-tweaking param ranges => iteration {}:\n".format(iteration_count))
                for param, value in updated_ranges.iteritems():
                    fout.write("#   {} updated to: {}\n".format(param, value))
                fout.write("# End re-tweak info.\n")
            call_step123_with_updated_input(input_file)

        valid_folds=[d for d in sorted(os.listdir(subfolder_path))
                     if os.path.isdir(os.path.join(subfolder_path,d))
                     and d.lower().startswith("sample")
                     and not d.lower().endswith("_removed")]
        sample_results=[]

        for samp in valid_folds:
            outdir=os.path.join(subfolder_path, samp)
            out_files=[oo for oo in os.listdir(outdir) if oo.endswith(".out")]
            pcr_files=[pp for pp in os.listdir(outdir) if pp.endswith(".pcr")]
            prf_files=[pr for pr in os.listdir(outdir) if pr.endswith(".prf")]

            if not out_files:
                sample_results.append(
                   {"sample_id":samp,
                    "rwp":None,"zero":None,"biso":None,"scale":None,"U":None,"V":None,"W":None,
                    "status":"NO_OUT","class":"INVALID",
                    "first_bragg_2theta": None,
                    "bragg_positions": []})
                continue

            info = parse_fullprof_out_file(os.path.join(outdir,out_files[0]))
            # parse PCR for final parameters
            if pcr_files:
                pcrpath = os.path.join(outdir, pcr_files[0])
                pcr_vals = parse_pcr_file(pcrpath, lattice_type)
                
                # Copy PCR values to info (standard parameters for all structures)
                info["zero"] = pcr_vals["zero"]
                info["biso"] = pcr_vals["biso"]  # Keep for backward compatibility
                info["scale"] = pcr_vals["scale"]
                info["U"] = pcr_vals["U"]
                info["V"] = pcr_vals["V"]
                info["W"] = pcr_vals["W"]
                
                # Copy atom-specific Biso values
                info["atom_biso"] = pcr_vals["atom_biso"]
                info["atom_names"] = pcr_vals["atom_names"]
                
                # Add lattice parameters based on structure type
                if lattice_type == "cubic":
                    info["a"] = pcr_vals["a"]
                elif lattice_type in ["tetragonal", "hexagonal"]:
                    info["a"] = pcr_vals["a"]
                    info["c"] = pcr_vals["c"]
                elif lattice_type == "orthorhombic":
                    info["a"] = pcr_vals["a"]
                    info["b"] = pcr_vals["b"]
                    info["c"] = pcr_vals["c"]
                elif lattice_type == "monoclinic":
                    info["a"] = pcr_vals["a"]
                    info["b"] = pcr_vals["b"]
                    info["c"] = pcr_vals["c"]
                    info["beta"] = pcr_vals["beta"]
                elif lattice_type == "triclinic":
                    info["a"] = pcr_vals["a"]
                    info["b"] = pcr_vals["b"]
                    info["c"] = pcr_vals["c"]
                    info["alpha"] = pcr_vals["alpha"]
                    info["beta"] = pcr_vals["beta"]
                    info["gamma"] = pcr_vals["gamma"]
                elif lattice_type == "trigonal":
                    info["a"] = pcr_vals["a"]
                    info["alpha"] = pcr_vals["alpha"]

            # parse PRF for reflection positions (and keep first reflection)
            info["bragg_positions"] = []
            info["first_bragg_2theta"] = None
            if prf_files:
                prfpath = os.path.join(outdir, prf_files[0])
                rp, _fp = parse_prf_bragg_positions(prfpath)
                info["bragg_positions"] = rp
                info["first_bragg_2theta"] = _fp

            info["sample_id"] = samp
            sample_results.append(info)

        # (A) Identify reference_rwp from sample1 if present & status=OK
        for sres in sample_results:
            if sres["sample_id"].lower()=="sample1" and sres["status"]=="OK" and sres["rwp"] is not None:
                reference_rwp = sres["rwp"]
                break

        # (B) Classification using updated thresholds:
        #     If we have reference_rwp, we do:
        #        close_up = reference_rwp + 10
        #        bound_up = reference_rwp + 20
        #     else fallback to the old approach with close_thr, bound_thr.
        close_up = None
        bound_up = None
        if reference_rwp is not None:
            close_up = reference_rwp + 10.0
            bound_up = reference_rwp + 20.0

        for sres in sample_results:
            if sres["status"]!="OK":
                sres["class"]="INVALID"
                continue
            if sres["rwp"] is None:
                sres["class"]="INVALID"
                continue
            val = sres["rwp"]

            if close_up is not None and bound_up is not None:
                # new classification with reference-based thresholds
                if val < close_up:
                    sres["class"]="CLOSE"
                elif val < bound_up:
                    sres["class"]="BOUNDARY"
                else:
                    sres["class"]="INVALID"
            else:
                # fallback to old approach:
                if val < close_thr:
                    sres["class"]="CLOSE"
                elif val < bound_thr:
                    sres["class"]="BOUNDARY"
                else:
                    sres["class"]="INVALID"

        tot=len(sample_results)
        c_count=sum(1 for rr in sample_results if rr["class"]=="CLOSE")
        b_count=sum(1 for rr in sample_results if rr["class"]=="BOUNDARY")
        i_count=sum(1 for rr in sample_results if rr["class"]=="INVALID")

        run_idx=get_run_index(classification_file)

        # Append to classification report
        with open(classification_file,"a") as f:
            # Explanation lines
            f.write("\n# Explanation:\n")
            f.write("#   SampleID: unique folder name for each sample.\n")
            f.write("#   Rwp(%)   : final Rwp in percent (Bragg-Only, Conventional line).\n")
            f.write("#   ZeroShift: final zero shift (degrees).\n")
            
            # Collect all unique atom names from all samples
            all_atom_names = set()
            for rr in sample_results:
                if "atom_biso" in rr:
                    all_atom_names.update(rr["atom_biso"].keys())
            
            # Sort atom names for consistent column order
            sorted_atom_names = sorted(list(all_atom_names))
            
            # List each atom's Biso in explanation
            for atom_name in sorted_atom_names:
                f.write("#   Biso_{}  : isotropic displacement parameter for atom {}.\n".format(atom_name, atom_name))
            
            f.write("#   Scale    : final scale factor.\n")
            f.write("#   U, V, W  : final peak shape parameters.\n")
            
            # Add structure-specific lattice parameter explanations
            if lattice_type == "cubic":
                f.write("#   a        : cubic lattice parameter (Å).\n")
            elif lattice_type in ["tetragonal", "hexagonal"]:
                f.write("#   a, c     : tetragonal/hexagonal lattice parameters (Å).\n")
            elif lattice_type == "orthorhombic":
                f.write("#   a, b, c  : orthorhombic lattice parameters (Å).\n")
            elif lattice_type == "monoclinic":
                f.write("#   a, b, c  : monoclinic lattice parameters (Å).\n")
                f.write("#   beta     : monoclinic angle (degrees).\n")
            elif lattice_type == "triclinic":
                f.write("#   a, b, c  : triclinic lattice parameters (Å).\n")
                f.write("#   alpha, beta, gamma: triclinic angles (degrees).\n")
            elif lattice_type == "trigonal":
                f.write("#   a        : trigonal lattice parameter (Å).\n")
                f.write("#   alpha    : trigonal angle (degrees).\n")
                
            f.write("#   Status   : parse result from .out (OK, NO_OUT, NEG_BISO, etc.).\n")
            f.write("#   Class    : classification label [CLOSE, BOUNDARY, INVALID].\n")
            f.write("#   FirstBragg: first bragg peak 2theta from .prf\n")

            f.write("\nRun: {}\n".format(run_idx))
            f.write("DateTime: {}\n".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            f.write("Crystal Structure: {}\n".format(lattice_type))

            f.write("# If 'sample1' is found, we define reference_rwp = that sample's Rwp => Rref.\n")
            if reference_rwp is not None:
                f.write("# Reference Rwp (Rref) from sample1 => {:.3f}\n".format(reference_rwp))
                f.write("# Thresholds => Close < (Rref+10), Boundary < (Rref+20), else Invalid.\n")
            else:
                f.write("# No sample1 found or no valid Rwp => fallback old threshold => close_thr={}, bound_thr={}\n"
                        .format(close_thr, bound_thr))

            # Collect all unique atom names from all samples
            all_atom_names = set()
            for rr in sample_results:
                if "atom_biso" in rr:
                    all_atom_names.update(rr["atom_biso"].keys())
            
            # Sort atom names for consistent column order
            sorted_atom_names = sorted(list(all_atom_names))
            
            # Create header based on lattice type for proper column alignment
            header = "# SampleID        Rwp(%)    ZeroShift    "
            
            # Add Biso columns for each atom
            for atom_name in sorted_atom_names:
                header += "Biso_{}  ".format(atom_name)
            
            header += "Scale      U        V        W    "
            
            # Add lattice parameter headers based on structure
            if lattice_type == "cubic":
                header += "   a     "
            elif lattice_type in ["tetragonal", "hexagonal"]:
                header += "   a        c     "
            elif lattice_type == "orthorhombic":
                header += "   a        b        c     "
            elif lattice_type == "monoclinic":
                header += "   a        b        c      beta   "
            elif lattice_type == "triclinic":
                header += "   a        b        c      alpha    beta    gamma "
            elif lattice_type == "trigonal":
                header += "   a      alpha   "
                
            header += " FirstBragg   Status     Class"
            f.write(header + "\n")

            for rr in sample_results:
                # Rwp => 3 decimals
                if rr["rwp"] is not None:
                    rw_str = "{:8.3f}".format(rr["rwp"])
                else:
                    rw_str = "   NA"

                # zero => 5 decimals
                if rr["zero"] is not None:
                    z_str = "{:11.5f}".format(rr["zero"])
                else:
                    z_str = "     NA"
                
                # Collect Biso values for each atom
                atom_biso_str = ""
                for atom_name in sorted_atom_names:
                    if "atom_biso" in rr and atom_name in rr["atom_biso"] and rr["atom_biso"][atom_name] is not None:
                        atom_biso_str += "{:8.5f}  ".format(rr["atom_biso"][atom_name])
                    else:
                        atom_biso_str += "   NA     "

                # scale => 6 decimals
                if rr["scale"] is not None:
                    s_str = "{:10.6f}".format(rr["scale"])
                else:
                    s_str = "    NA"

                # U, V, W => 5 decimals
                if rr["U"] is not None:
                    U_str = "{:8.5f}".format(rr["U"])
                else:
                    U_str = "   NA"
                if rr["V"] is not None:
                    V_str = "{:8.5f}".format(rr["V"])
                else:
                    V_str = "   NA"
                if rr["W"] is not None:
                    W_str = "{:8.5f}".format(rr["W"])
                else:
                    W_str = "   NA"
                
                # Lattice parameters based on structure
                lattice_str = ""
                if lattice_type == "cubic":
                    if "a" in rr and rr["a"] is not None:
                        lattice_str = "{:8.5f}".format(rr["a"])
                    else:
                        lattice_str = "   NA"
                elif lattice_type in ["tetragonal", "hexagonal"]:
                    if "a" in rr and rr["a"] is not None and "c" in rr and rr["c"] is not None:
                        lattice_str = "{:8.5f} {:8.5f}".format(rr["a"], rr["c"])
                    else:
                        lattice_str = "   NA      NA"
                elif lattice_type == "orthorhombic":
                    if "a" in rr and rr["a"] is not None and "b" in rr and rr["b"] is not None and "c" in rr and rr["c"] is not None:
                        lattice_str = "{:8.5f} {:8.5f} {:8.5f}".format(rr["a"], rr["b"], rr["c"])
                    else:
                        lattice_str = "   NA      NA      NA"
                elif lattice_type == "monoclinic":
                    if "a" in rr and rr["a"] is not None and "b" in rr and rr["b"] is not None and "c" in rr and rr["c"] is not None and "beta" in rr and rr["beta"] is not None:
                        lattice_str = "{:8.5f} {:8.5f} {:8.5f} {:8.5f}".format(rr["a"], rr["b"], rr["c"], rr["beta"])
                    else:
                        lattice_str = "   NA      NA      NA     NA"
                elif lattice_type == "triclinic":
                    if "a" in rr and rr["a"] is not None and "b" in rr and rr["b"] is not None and "c" in rr and rr["c"] is not None and "alpha" in rr and rr["alpha"] is not None and "beta" in rr and rr["beta"] is not None and "gamma" in rr and rr["gamma"] is not None:
                        lattice_str = "{:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}".format(
                            rr["a"], rr["b"], rr["c"], rr["alpha"], rr["beta"], rr["gamma"])
                    else:
                        lattice_str = "   NA      NA      NA     NA     NA     NA"
                elif lattice_type == "trigonal":
                    if "a" in rr and rr["a"] is not None and "alpha" in rr and rr["alpha"] is not None:
                        lattice_str = "{:8.5f} {:8.5f}".format(rr["a"], rr["alpha"])
                    else:
                        lattice_str = "   NA     NA"

                st  = rr["status"]
                cl  = rr.get("class","?")

                # first bragg => 5 decimals, or NA
                if rr.get("first_bragg_2theta") is not None:
                    fb_str = "{:10.5f}".format(rr["first_bragg_2theta"])
                else:
                    fb_str = "    NA"

                line = ("{:<12s}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {:<8s}  {}\n".format(
                         rr["sample_id"], rw_str, z_str, atom_biso_str, s_str,
                         U_str, V_str, W_str, lattice_str, fb_str, st, cl))
                f.write(line)

            f.write("\n# Summary:\n")
            f.write("Number_Total={}\n".format(tot))
            f.write("Number_Close={}\n".format(c_count))
            f.write("Number_Boundary={}\n".format(b_count))
            f.write("Number_Invalid={}\n".format(i_count))
            if tot > 0:
                close_ratio = 100.0 * c_count / tot
                bound_ratio = 100.0 * b_count / tot
                f.write("Final_Close_Ratio= {:.2f}%\n".format(close_ratio))
                f.write("Final_Boundary_Ratio= {:.2f}%\n".format(bound_ratio))
            else:
                close_ratio = 0.0
                bound_ratio = 0.0
                f.write("Final_Close_Ratio= 0%\n")
                f.write("Final_Boundary_Ratio= 0%\n")
            f.write("# End of Run {}\n".format(run_idx))

        # Replace "cb = c_count + b_count" with a direct reference to tot:
        ratio_c = (100.0 * c_count / tot) if tot > 0 else 0.0

        print("[step3.5] iteration={}, run={}, total={}, close={}, boundary={}, invalid={}, close_ratio={:.2f}%"
              .format(
                  iteration_count,
                  run_idx,
                  tot,
                  c_count,
                  b_count,
                  i_count,
                  ratio_c  # (already matches c_count/tot)
              ))

        ######################################################################
        # Overplot with Bragg positions on the main axis (matching class color),
        # but now drawn as short vertical bars further below y=0, so there's
        # a clear gap. Also compute an average first-peak position to center
        # our inset range. We'll shrink the inset's vertical coverage to about
        # 60% of the peak height, and embed the inset as a wide/short box
        # in the upper-right region using inset_axes().
        ######################################################################

        final_iteration = False
        # --------------------------------------------------------------------
        # CHANGE #1: We only stop if
        # and ratio_c is between 75 and 85, OR we hit max_iter
        # --------------------------------------------------------------------
        if (70.0 <= ratio_c <= 90.0) or (iteration_count >= max_iter):
            final_iteration = True

        # First check if we should generate plots for this iteration
        should_generate_plot = False
        if config.get("generate_classification_plots", "Y").upper().startswith("Y") and fig_generate:
            # Only generate plots if:
            # 1. Plot all runs is enabled (33>Y), OR
            # 2. This is the final iteration, OR
            # 3. This is the first/only iteration when partial reruns are disabled
            if plot_all_runs or final_iteration or (iteration_count == 1 and not partial_rerun):
                should_generate_plot = True
                print("[step3.5] Generating multi-sample overplot with distinct markers + inset box.")

        # Now only do the plotting if the condition is met
        if should_generate_plot:
            # Create output directory
            out_fig_dir = os.path.join(subfolder_path, "output_figures")
            if not os.path.exists(out_fig_dir):
                os.mkdir(out_fig_dir)
            
            class_plot_dir = os.path.join(out_fig_dir, "classification_profile_plots")
            if not os.path.exists(class_plot_dir):
                os.mkdir(class_plot_dir)

            # Continue with plotting code
            close_samps = [r for r in sample_results if r.get("class") == "CLOSE"]
            bound_samps = [r for r in sample_results if r.get("class") == "BOUNDARY"]
            inv_samps   = [r for r in sample_results if r.get("class") == "INVALID"]
            
            # NEW CODE: Limit to 1000 samples for plotting if total exceeds 1000
            total_samples = len(close_samps) + len(bound_samps) + len(inv_samps)
            max_plot_samples = 1000
            
            if total_samples > max_plot_samples:
                import random
                # Use a fixed seed for reproducibility
                random.seed(42)
                
                print("[step3.5] Total samples ({}) exceeds max plotting limit ({}). Selecting representative subset.".format(
                    total_samples, max_plot_samples))
                
                # Calculate proportions
                close_prop = float(len(close_samps)) / total_samples
                bound_prop = float(len(bound_samps)) / total_samples
                inv_prop = float(len(inv_samps)) / total_samples
                
                # Calculate number of samples to include from each category
                # Use int() to ensure we get whole numbers
                close_to_plot = int(max_plot_samples * close_prop)
                bound_to_plot = int(max_plot_samples * bound_prop)
                inv_to_plot = max_plot_samples - close_to_plot - bound_to_plot  # Ensure sum is exactly 1000
                
                print("[step3.5] Plotting subset with CLOSE={} ({}%), BOUNDARY={} ({}%), INVALID={} ({}%)".format(
                    close_to_plot, close_prop*100, bound_to_plot, bound_prop*100, inv_to_plot, inv_prop*100))
                
                # Randomly select samples from each category
                if len(close_samps) > close_to_plot:
                    plot_close_samps = random.sample(close_samps, close_to_plot)
                else:
                    plot_close_samps = close_samps
                    
                if len(bound_samps) > bound_to_plot:
                    plot_bound_samps = random.sample(bound_samps, bound_to_plot)
                else:
                    plot_bound_samps = bound_samps
                    
                if len(inv_samps) > inv_to_plot:
                    plot_inv_samps = random.sample(inv_samps, inv_to_plot)
                else:
                    plot_inv_samps = inv_samps
            else:
                # Use all samples if under the limit
                plot_close_samps = close_samps
                plot_bound_samps = bound_samps
                plot_inv_samps = inv_samps

            # Get all Bragg positions from samples
            bragg_positions_by_sample = {}
            for entry in sample_results:
                sample_id = entry["sample_id"]
                positions = entry.get("bragg_positions", [])
                if positions:
                    bragg_positions_by_sample[sample_id] = positions

            # Process zoom peak selection
            zoom_peak_idx = 0
            try:
                if zoom_peak.lower() == "auto":
                    # Find most intense peak position (to be implemented)
                    # For now, use first peak
                    zoom_peak_idx = 0
                else:
                    # Use user-selected peak (convert from 1-based to 0-based index)
                    zoom_peak_idx = int(zoom_peak) - 1
                    if zoom_peak_idx < 0:
                        zoom_peak_idx = 0
            except:
                zoom_peak_idx = 0
            
            # Find the average position of the selected Bragg peak
            selected_positions = []
            for positions in bragg_positions_by_sample.values():
                if zoom_peak_idx < len(positions):
                    selected_positions.append(positions[zoom_peak_idx])
                    
            if selected_positions:
                avg_selected_peak = sum(selected_positions) / len(selected_positions)
            else:
                # Fallback to first peak
                all_fp = [s["first_bragg_2theta"] for s in sample_results
                          if s["first_bragg_2theta"] is not None]
                if all_fp:
                    avg_selected_peak = sum(all_fp) / len(all_fp)
                else:
                    avg_selected_peak = 28.0  # Default value if no peaks found

            two_theta_range = config.get("two_theta_range", "default")
            if two_theta_range.lower() == "default":
                x_min, x_max = 0, 100  # Default range
            else:
                try:
                    range_parts = two_theta_range.split(',')
                    x_min = float(range_parts[0])
                    x_max = float(range_parts[1])
                except:
                    x_min, x_max = 0, 100  # Fallback to default

            fig, ax = plt.figure(figsize=(10,6)), plt.gca()
            ax.set_xlim(x_min, x_max)  # Use the configured range instead of hardcoded values
            # Set larger tick label sizes (28pt for main plot)
            ax.tick_params(axis='both', which='major', labelsize=25)

            # PRF data cache to avoid reading files multiple times
            prf_cache = {}
            bragg_cache = {}

            def read_first_prf_cached(sample_folder):
                if sample_folder in prf_cache:
                    return prf_cache[sample_folder]
                pdir = os.path.join(subfolder_path, sample_folder)
                prf_l = [xx for xx in os.listdir(pdir) if xx.endswith(".prf")]
                if not prf_l:
                    prf_cache[sample_folder] = None
                    return None
                data = read_prf_data(os.path.join(pdir, prf_l[0]))
                prf_cache[sample_folder] = data
                return data

            def get_bragg_positions_cached(sample_folder):
                if sample_folder in bragg_cache:
                    return bragg_cache[sample_folder]
                pdir = os.path.join(subfolder_path, sample_folder)
                prf_l = [xx for xx in os.listdir(pdir) if xx.endswith(".prf")]
                if not prf_l:
                    bragg_cache[sample_folder] = []
                    return []
                prfpath = os.path.join(pdir, prf_l[0])
                rp, _fp = parse_prf_bragg_positions(prfpath)
                bragg_cache[sample_folder] = rp
                return rp

            # Pre-load all PRF data into cache
            all_plot_samps = plot_close_samps + plot_bound_samps + plot_inv_samps
            print "[DEBUG] Pre-loading {} PRF files into cache...".format(len(all_plot_samps))
            for idx, entry in enumerate(all_plot_samps):
                if (idx + 1) % 100 == 0:
                    print "[DEBUG] Cached {}/{} PRF files".format(idx + 1, len(all_plot_samps))
                read_first_prf_cached(entry["sample_id"])
                get_bragg_positions_cached(entry["sample_id"])
            print "[DEBUG] PRF cache complete. {} files loaded.".format(len(prf_cache))
            sys.stdout.flush()

            def plot_one_class(ax_local, sample_list, label_for_legend, line_style, marker_style, color, yobs_plot=False):
                print "[DEBUG] Plotting {} category: {} samples".format(label_for_legend, len(sample_list))
                sys.stdout.flush()
                for idx, entry in enumerate(sample_list):
                    if (idx + 1) % 10 == 0 or idx == 0:
                        print "[DEBUG] Plotting sample {}/{}".format(idx + 1, len(sample_list))
                        sys.stdout.flush()
                    arr = read_first_prf_cached(entry["sample_id"])
                    if arr is None or len(arr) < 1:
                        continue
                    if yobs_plot:
                        ax_local.plot(
                            arr[:,0],
                            arr[:,1],
                            marker='o',
                            color='red',
                            fillstyle='none',
                            linestyle='none',
                            markersize=7,
                            label='Yobs'
                        )
                        yobs_plot = False
                    lbl = label_for_legend if idx == 0 else None
                    # Use markevery to show markers only every Nth point (faster than all points)
                    ax_local.plot(
                        arr[:,0],
                        arr[:,2],
                        linestyle=line_style,
                        marker=marker_style,
                        markevery=50,  # Show marker every 50th point
                        fillstyle='none',
                        color=color,
                        linewidth=0.5,
                        markersize=4,
                        alpha=0.7,
                        label=lbl
                    )
                    # Only draw Bragg positions for first sample (they're the same for all)
                    if idx == 0:
                        bpos = get_bragg_positions_cached(entry["sample_id"])
                        for bp in bpos:
                            ax_local.vlines(
                                x=bp,
                                ymin=-100,
                                ymax=0,
                                color=color,
                                linewidth=0.5
                            )

            yobs_done = False
            if plot_close_samps:
                print "[DEBUG] Plotting CLOSE classification ({} samples)".format(len(plot_close_samps))
                plot_one_class(
                    ax,
                    plot_close_samps,
                    label_for_legend="close",
                    line_style='--',
                    marker_style='o',
                    color='black',
                    yobs_plot=(not yobs_done)
                )
                yobs_done = True
            if plot_bound_samps:
                print "[DEBUG] Plotting BOUNDARY classification ({} samples)".format(len(plot_bound_samps))
                plot_one_class(
                    ax,
                    plot_bound_samps,
                    label_for_legend="boundary",
                    line_style='--',
                    marker_style='o',
                    color='green',
                    yobs_plot=(not yobs_done)
                )
                yobs_done = True
            if plot_inv_samps:
                print "[DEBUG] Plotting INVALID classification ({} samples)".format(len(plot_inv_samps))
                plot_one_class(
                    ax,
                    plot_inv_samps,
                    label_for_legend="invalid",
                    line_style='--',
                    marker_style='o',
                    color='blue',
                    yobs_plot=(not yobs_done)
                )
                yobs_done = True

            curr_ylim = ax.get_ylim()
            if curr_ylim[0] > -100:
                ax.set_ylim(-100, curr_ylim[1])

            ax.set_xlabel(u"2\u03B8 (deg.)", fontsize=25, labelpad=10)  # Add 10 points of padding
            ax.set_ylabel("Intensity (a.u.)", fontsize=25, labelpad=10)  # Add 10 points of padding
            title = "Classification Overplot (Run={})".format(run_idx)
            if lattice_type != "cubic":
                title += " - {} Structure".format(lattice_type.capitalize())
            ax.set_title(title, fontsize=28, pad=15)  # Default ~12 + 10 = 22
            # Move legend to bottom right, outside the plot
            ax.legend(fontsize=22, loc='center left', bbox_to_anchor=(1.02, 0.2))

            print "[DEBUG] Starting to generate plots for each classification category..."

            # Determine zoom width based on structure type and user input
            if zoom_width.lower() == "auto":
                # Use structure-dependent defaults
                if lattice_type == "cubic":
                    x_half_width = 0.1  # Standard width for cubic
                elif lattice_type in ["tetragonal", "hexagonal", "trigonal"]:
                    x_half_width = 0.15  # Slightly wider for tetragonal/hexagonal/trigonal
                elif lattice_type == "orthorhombic":
                    x_half_width = 0.2   # Wider for orthorhombic (often has peak splitting)
                elif lattice_type in ["monoclinic", "triclinic"]:
                    x_half_width = 0.25  # Widest for low-symmetry (monoclinic/triclinic)
                else:
                    x_half_width = 0.15  # Default fallback
            else:
                # Use user-specified width
                try:
                    x_half_width = float(zoom_width) / 2.0
                except:
                    # Default if invalid
                    x_half_width = 0.15
                    
            x_low = avg_selected_peak - x_half_width
            x_high = avg_selected_peak + x_half_width

            # Use the limited sample lists for inset (reuse cache)
            y_inset_max = 0.0
            y_inset_min = float('inf')  # Initialize with infinity to find minimum

            for entry in all_plot_samps:
                arr = read_first_prf_cached(entry["sample_id"])
                if arr is None or len(arr) < 1:
                    continue
                inrange = arr[(arr[:,0]>=x_low) & (arr[:,0]<=x_high)]
                if len(inrange)>0:
                    # Find maximum for y-axis upper limit
                    local_max = max(np.max(inrange[:,1]), np.max(inrange[:,2]))
                    if local_max > y_inset_max:
                        y_inset_max = local_max
                        
                    # Find minimum for y-axis lower limit
                    local_min = min(np.min(inrange[:,1]), np.min(inrange[:,2]))
                    if local_min < y_inset_min:
                        y_inset_min = local_min

            # Add 5% padding to max and min
            y_inset_max *= 1.05
            y_inset_min *= 0.95  # Reduce minimum by 5% to add a small margin

            if y_inset_max < 10:
                y_inset_max = 10.0
            if y_inset_min > 200:  # Don't start too high if minimum is large
                y_inset_min = y_inset_min * 0.8

            print "[DEBUG] Creating zoomed inset at 2-theta = {:.4f}".format(avg_selected_peak)

            axins = inset_axes(
                ax,
                width="30%",
                height="30%",
                loc="upper right"
            )
            axins.set_xlim(x_low, x_high)
            axins.set_ylim(y_inset_min, y_inset_max)  # Use calculated minimum instead of 0

            # Set only 3 y-axis labels for the inset
            axins.yaxis.set_major_locator(plt.MaxNLocator(nbins=2))
            # Set larger tick label sizes for inset (26pt)
            axins.tick_params(axis='both', which='major', labelsize=25)

            def plot_one_class_inset(ax_in, sample_list, line_style, marker_style, color):
                for idx, entry in enumerate(sample_list):
                    arr = read_first_prf_cached(entry["sample_id"])
                    if arr is None or len(arr) < 1:
                        continue
                    # Only plot Yobs for first sample (they're all the same)
                    if idx == 0:
                        ax_in.plot(
                            arr[:,0],
                            arr[:,1],
                            marker='o',
                            markevery=5,  # Fewer points for inset (zoomed in)
                            color='red',
                            fillstyle='none',
                            linestyle='none',
                            markersize=6,
                            zorder=10
                        )
                    # Use markevery for speed
                    ax_in.plot(
                        arr[:,0],
                        arr[:,2],
                        linestyle=line_style,
                        marker=marker_style,
                        markevery=5,  # Fewer points for inset
                        fillstyle='none',
                        color=color,
                        linewidth=0.5,
                        markersize=4,
                        alpha=0.7
                    )

            # Use the limited sample lists for inset plotting
            plot_one_class_inset(axins, plot_close_samps, '--', 'o', 'black')
            plot_one_class_inset(axins, plot_bound_samps, '--', 'o', 'green')
            plot_one_class_inset(axins, plot_inv_samps, '--', 'o', 'blue')

            # Hide the inset-connecting lines by setting ec="none"
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="none")

            # # Add a label at the top of the inset
            # if zoom_peak.lower() == "auto":
            #     inset_label = "<first Bragg reflection>"
            # else:
            #     # Format based on selected peak number
            #     if lattice_type == "cubic":
            #         inset_label = "<Bragg reflection #{}>".format(zoom_peak_idx + 1)
            #     else:
            #         inset_label = "<Bragg reflection #{}>".format(zoom_peak_idx + 1)
                
            # # Add a label at the upper right corner of the inset
            # axins.text(
            #     0.95, 0.95,
            #     inset_label,
            #     transform=axins.transAxes,
            #     ha='right',  # Right-aligned instead of centered
            #     va='top',
            #     fontsize=7,  # Smaller font size
            #     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', pad=2.0),
            #     zorder=5  # Ensure text appears above the data
            # )

            out_pdf = os.path.join(
                class_plot_dir,
                "classification_profile_overplot_run{}_zoomed.pdf".format(run_idx)
            )
            print "[DEBUG] Saving plot with DPI={} to {}".format(plot_dpi, out_pdf)
            plt.savefig(out_pdf, dpi=plot_dpi, bbox_inches='tight', format='pdf')

            print("[step3.5] Overplot with zoom => {}".format(out_pdf))

            if fig_display:
                plt.show()
            else:
                plt.close(fig)

        # --------------------------------------------------------------------
        # CHANGE #2: We break if final_iteration is True, explaining reason.
        # --------------------------------------------------------------------
        if final_iteration:
            if iteration_count >= max_iter:
                print("[step3.5] Reached max iteration => stop.")
            else:
                print("[step3.5] ratio in ±10% around 80% for close classification")
            break
        elif iteration_count == 1 and not partial_rerun:
            # If we're only doing one iteration because partial_rerun is disabled,
            # we should treat this as the final iteration for plotting purposes
            print("[step3.5] Only one iteration requested (partial_rerun=N). Completed.")
            break
        else:
            # proceed to next iteration
            continue

    print("[step3.5] Done after {} iteration(s).".format(iteration_count))


if __name__=="__main__":
    main()