#!/usr/bin/env python


import os
import sys
import shutil
import re
import json
import glob
from pathlib import Path

def setup_refinement_folders(identified_dir):
    refine_root = os.path.join(identified_dir, "Rietveld_Refinement")
    cnn_folder = os.path.join(refine_root, "CNN_ML_refinement")
    
    os.makedirs(refine_root, exist_ok=True)
    os.makedirs(cnn_folder, exist_ok=True)
    
    return refine_root, cnn_folder

def find_catalog_file():
    catalog_path = os.path.join('single_phase_identification', 'database', 'material_catalog.json')
    if not os.path.exists(catalog_path):
        print(f"Error: Material catalog not found at {catalog_path}")
        return None
    
    try:
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
        return catalog
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return None

def find_material_info(identified_dir, catalog):

    folder_name = os.path.basename(identified_dir)
    parts = folder_name.split('_')
    if len(parts) < 2:
        print(f"Error: Cannot extract material name from folder {folder_name}")
        return None, None, None
    
    material_name = parts[1]  
    print(f"Identified material: {material_name}")
    
    param_dir = os.path.join(identified_dir, "parameter_refinement")
    if not os.path.exists(param_dir):
        print(f"Error: Parameter directory not found at {param_dir}")
        return None, None, None
    
    param_files = glob.glob(os.path.join(param_dir, "*_refined_parameters.dat"))
    if not param_files:
        print(f"Error: No refined parameters file found in {param_dir}")
        return None, None, None
    
    param_file = param_files[0]
    
    source_dir = os.path.join(identified_dir, "source_files")
    dat_files = glob.glob(os.path.join(source_dir, "*.dat"))
    if not dat_files:
        print(f"Error: No .dat file found in {source_dir}")
        return None, None, None
    
    dat_file = dat_files[0]
    
    return material_name, param_file, dat_file

def read_refined_parameters(param_file):
    params = {}
    crystal_structure = "cubic"  
    
    try:
        with open(param_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if "Crystal Structure:" in line:
                        crystal_structure = line.split(':', 1)[1].strip()
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = float(value.strip())
                    params[key] = value
        
        params["_crystal_structure"] = crystal_structure
        
        atom_biso = {}
        for key in list(params.keys()):
            if key.startswith("Biso "):
                atom_name = key.split(" ", 1)[1].strip()
                atom_biso[atom_name] = params[key]
        
        params["atom_biso"] = atom_biso
        
        return params
    
    except Exception as e:
        print(f"Error reading parameters: {e}")
        return None

def find_pcr_template(material_name, catalog):

    train_data_dir = os.path.join('data', 'train_data')
    
    material_folders = []
    for folder in os.listdir(train_data_dir):
        # Check if folder name starts with material name (e.g., pbso4_*)
        if folder.lower().startswith(material_name.lower() + "_"):
            material_folders.append(folder)
    
    material_folders.sort(reverse=True)
    
    for folder_name in material_folders:
        source_dir = os.path.join(train_data_dir, folder_name, 'source_files')
        if os.path.isdir(source_dir):
            for f in os.listdir(source_dir):
                if f.lower().endswith('.pcr'):
                    pcr_template = os.path.join(source_dir, f)
                    print(f"Found original PCR template in training data: {pcr_template}")
                    return pcr_template
    
    print(f"No original PCR found in training data for {material_name}, checking catalog...")
    
    if material_name not in catalog:
        print(f"Error: Material '{material_name}' not found in catalog")
        return None
    
    material_info = catalog[material_name]
    pcr_template = material_info.get("pcr_template")
    
    if not pcr_template or not os.path.exists(pcr_template):
        print(f"Error: PCR template for {material_name} not found or does not exist")
        model_folder = material_info.get("model_folder")
        if model_folder:
            model_path = os.path.join("saved_models", model_folder)
            if os.path.exists(model_path):
                pcr_files = []
                
                refinement_path = os.path.join(model_path, "refinement_result")
                if os.path.exists(refinement_path):
                    for root, dirs, files in os.walk(refinement_path):
                        for file in files:
                            if file.endswith(".pcr"):
                                pcr_files.append(os.path.join(root, file))
                
                if not pcr_files:
                    for root, dirs, files in os.walk(model_path):
                        for file in files:
                            if file.endswith(".pcr"):
                                pcr_files.append(os.path.join(root, file))
                
                if pcr_files:
                    pcr_template = pcr_files[0]
                    print(f"Found PCR template in model folder: {pcr_template}")
                    return pcr_template
        
        print("Could not find a PCR template for this material")
        return None
    
    print(f"Using catalog PCR template: {pcr_template}")
    return pcr_template

def create_pcr_with_predicted_params(original_pcr, new_pcr, param_dict):

    if not os.path.exists(original_pcr):
        print(f"Original PCR not found: {original_pcr}")
        return False

    with open(original_pcr, 'r') as f:
        lines = f.readlines()

    crystal_structure = param_dict.get("_crystal_structure", "cubic").lower()
    biso_atoms = param_dict.get("atom_biso", {})
    
    print(f"Creating PCR file for {crystal_structure} structure with {len(biso_atoms)} atom types")
    if biso_atoms:
        print(f"Atom Biso values to apply: {', '.join([atom for atom in biso_atoms])}")

    found_zero = found_bg = found_lat = found_biso = found_scale = found_uvw = False
    new_lines = []

    in_atom_section = False
    processed_atoms = set()

    for i, line in enumerate(lines):
        if "!  Zero    Code    SyCos" in line:
            new_lines.append(line)
            if i + 1 < len(lines):
                zero_val = param_dict.get('Zero', 0.0)
                formatted_val = f"{zero_val:.6f}"
                new_line = f"  {formatted_val}" + lines[i + 1][10:]
                new_lines.append(new_line)
                found_zero = True
            continue

        elif "!   Background coefficients/codes" in line:
            new_lines.append(line)
            if i + 1 < len(lines):
                if 'Background' in param_dict:
                    original_line = lines[i + 1]
                    bg_val = param_dict.get('Background', 0.0)
                    bg_parts = original_line.split()
                    if len(bg_parts) >= 6:
                        new_line = "      " + f"{bg_val:8.3f}" + "     " + "     ".join(bg_parts[2:]) + "\n"
                    else:
                        new_line = "      " + f"{bg_val:8.3f}" + "     -19.888      10.065      -2.284       0.213       0.000\n"
                    new_lines.append(new_line)
                else:
                    new_lines.append(lines[i + 1])
                    print(f"Preserving original background coefficients (not found in parameters)")
                found_bg = True
            continue

        elif "!     a          b         c        alpha      beta       gamma      #Cell Info" in line:
            new_lines.append(line)
            if i + 1 < len(lines):
                if crystal_structure == "cubic":
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = a_val  
                    c_val = a_val  
                    alpha_val = 90.0
                    beta_val = 90.0
                    gamma_val = 90.0
                elif crystal_structure == "tetragonal":
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = a_val  
                    c_val = param_dict.get('Lattice c', 5.430)
                    alpha_val = 90.0
                    beta_val = 90.0
                    gamma_val = 90.0
                elif crystal_structure == "orthorhombic":
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = param_dict.get('Lattice b', 5.430)
                    c_val = param_dict.get('Lattice c', 5.430)
                    alpha_val = 90.0
                    beta_val = 90.0
                    gamma_val = 90.0
                elif crystal_structure == "hexagonal":
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = a_val  
                    c_val = param_dict.get('Lattice c', 5.430)
                    alpha_val = 90.0
                    beta_val = 90.0
                    gamma_val = 120.0
                elif crystal_structure == "monoclinic":
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = param_dict.get('Lattice b', 5.430)
                    c_val = param_dict.get('Lattice c', 5.430)
                    alpha_val = 90.0
                    beta_val = param_dict.get('Beta', 90.0)
                    gamma_val = 90.0
                elif crystal_structure == "triclinic":
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = param_dict.get('Lattice b', 5.430)
                    c_val = param_dict.get('Lattice c', 5.430)
                    alpha_val = param_dict.get('Alpha', 90.0)
                    beta_val = param_dict.get('Beta', 90.0)
                    gamma_val = param_dict.get('Gamma', 90.0)
                elif crystal_structure == "trigonal":
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = a_val  
                    c_val = a_val  
                    alpha_val = param_dict.get('Alpha', 90.0)
                    beta_val = alpha_val  
                    gamma_val = alpha_val  
                else:
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = a_val
                    c_val = a_val
                    alpha_val = 90.0
                    beta_val = 90.0
                    gamma_val = 90.0
                
                new_line = (
                    f"   {a_val:10.6f}"   # a (3 spaces before)
                    f"{b_val:10.6f}"      # b (no extra spaces)
                    f"{c_val:10.6f}"      # c (no extra spaces)
                    f"  {alpha_val:8.6f}"        # alpha (2 spaces before)
                    f"  {beta_val:8.6f}"        # beta (2 spaces before)
                    f"  {gamma_val:8.6f}   "     # gamma (2 spaces before, 3 after)
                    "\n"
                )
                new_lines.append(new_line)
                found_lat = True
            continue

        elif "!  Scale        Shape1      Bov" in line:
            new_lines.append(line)
            if i + 1 < len(lines):
                scale_val = param_dict.get('Scale', 0.0001)
                existing_parts = lines[i + 1].split()
                scale_mantissa = scale_val * 1000
                new_line = f"  {scale_mantissa:7.7f}E-03   {existing_parts[1]:>7}   {existing_parts[2]:>7}   {existing_parts[3]:>7}   {existing_parts[4]:>7}   {existing_parts[5]:>7}       {existing_parts[6]}\n"
                new_lines.append(new_line)
                found_scale = True
            continue

        elif "!       U         V          W           X          Y" in line:
            new_lines.append(line)
            if i + 1 < len(lines):
                u_val = param_dict.get('U', 0.0)
                v_val = param_dict.get('V', 0.0)
                w_val = param_dict.get('W', 0.0)
                new_line = (
                    "     "                   # 5 spaces at start
                    f"{u_val:9.6f}"          # U value (9 chars total)
                    "    "                    # 4 spaces after U
                    f"{v_val:9.6f}"          # V value (9 chars total)
                    "     "                   # 5 spaces after V
                    f"{w_val:9.6f}"          # W value (9 chars total)
                    "     "                   # 5 spaces after W
                )
                
                orig_parts = lines[i + 1].split()
                if len(orig_parts) >= 5:
                    try:
                        x_val = orig_parts[3]
                        y_val = orig_parts[4]
                        new_line += f"{x_val}     {y_val}     0.000000     0.000000       0\n"
                    except:
                        new_line += "0.076699     0.000000     0.000000     0.000000       0\n"
                else:
                    new_line += "0.076699     0.000000     0.000000     0.000000       0\n"
                
                new_lines.append(new_line)
                found_uvw = True
            continue

        elif "!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes" in line:
            new_lines.append(line)
            in_atom_section = True
            continue

        elif in_atom_section and "!    beta11   beta22   beta33   beta12   beta13   beta23  /Codes" in line:
            new_lines.append(line)
            continue

        elif in_atom_section and re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+[-+]?\d*\.\d+', line):
            parts = line.split()
            if len(parts) >= 6:
                atom_name = parts[0].lower()  # Get atom name and convert to lowercase
                processed_atoms.add(atom_name)
                
                biso_val = None
                
                for atom_key, atom_val in biso_atoms.items():
                    if atom_key.lower() == atom_name.lower() or atom_key.lower() in atom_name.lower() or atom_name.lower() in atom_key.lower():
                        biso_val = atom_val
                        print(f"Found Biso match for {atom_name} using key {atom_key} -> {biso_val}")
                        break
                
                if biso_val is None and 'Biso' in param_dict:
                    biso_val = param_dict['Biso']
                    print(f"Using generic Biso value ({biso_val}) for atom {atom_name}")
                
                if biso_val is not None:
                    updated_line_parts = parts[:5]
                    updated_line_parts.append(f"{biso_val:.5f}")
                    updated_line_parts.extend(parts[6:])
                    
                    formatted_line = ""
                    
                    formatted_line += f"{updated_line_parts[0]:<6}"
                    
                    formatted_line += f"{updated_line_parts[1]:<8}"
                    
                    for j in range(2, 5):
                        formatted_line += f"{float(updated_line_parts[j]):8.5f}  "
                    
                    formatted_line += f"{float(updated_line_parts[5]):8.5f}  "
                    
                    for j in range(6, len(updated_line_parts)):
                        formatted_line += f"{updated_line_parts[j]} "
                    
                    if not formatted_line.endswith('\n'):
                        formatted_line += '\n'
                    
                    new_lines.append(formatted_line)
                    found_biso = True
                    print(f"Updated Biso for atom {atom_name} to {biso_val:.5f}")
                else:
                    new_lines.append(line)
                    print(f"No Biso match found for atom {atom_name}, keeping original value")
            else:
                new_lines.append(line)
            continue
        
        elif in_atom_section and "!-------> Profile Parameters for Pattern #" in line:
            in_atom_section = False
            new_lines.append(line)
            continue
            
        else:
            new_lines.append(line)

    with open(new_pcr, 'w') as ff:
        ff.writelines(new_lines)

    missing_params = []
    if not found_zero:
        missing_params.append("Zero shift")
    if not found_bg:
        missing_params.append("Background")
    if not found_lat:
        missing_params.append("Lattice")
    if not found_biso:
        missing_params.append("Biso")
    if not found_scale:
        missing_params.append("Scale")
    if not found_uvw:
        missing_params.append("U,V,W")
        
    if missing_params:
        print(f"Warning: did not set the following parameters: {', '.join(missing_params)}")
    else:
        print(f"Successfully updated all parameters for {crystal_structure} structure")

    print(f"Created new .PCR => {new_pcr}")
    return True

def remove_duplicate_parameter_lines(pcr_path):

    if not os.path.isfile(pcr_path):
        return
        
    with open(pcr_path, 'r') as f:
        lines = f.readlines()
        
    cleaned_lines = []
    i = 0
    in_atom_section = False
    
    param_markers = [
        "!  Zero    Code    SyCos",
        "!   Background coefficients/codes",
        "!     a          b         c        alpha      beta       gamma      #Cell Info",
        "!  Scale        Shape1      Bov",
        "!       U         V          W           X          Y"
    ]
    
    atom_section_marker = "!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes"
    atom_section_end_markers = [
        "!-------> Profile Parameters for Pattern #",
        "! Pr1    Pr2    Pr3   Brind"
    ]
    
    while i < len(lines):
        line = lines[i]
        
        if atom_section_marker in line:
            in_atom_section = True
            cleaned_lines.append(line)
            i += 1
            continue
            
        if in_atom_section and any(marker in line for marker in atom_section_end_markers):
            in_atom_section = False
            
        if in_atom_section:
            cleaned_lines.append(line)
            i += 1
            continue
            
        if any(marker in line for marker in param_markers):
            cleaned_lines.append(line)
            if i + 1 < len(lines):
                cleaned_lines.append(lines[i + 1])
            i += 3
        else:
            cleaned_lines.append(line)
            i += 1
            
    with open(pcr_path, 'w') as f:
        f.writelines(cleaned_lines)
        
    print(f"[Post-Fix] Removed old parameter lines in {pcr_path} while preserving atom section")

def prepare_refinement_files(params, pcr_template, identified_dir, dat_file, refine_root, cnn_folder):

    cnn_pcr_path = os.path.join(cnn_folder, "cnn_refined.pcr")
    if not create_pcr_with_predicted_params(pcr_template, cnn_pcr_path, params):
        print("Error creating PCR file with predicted parameters")
        return False
    
    remove_duplicate_parameter_lines(cnn_pcr_path)
    
    expected_dat_name = None
    
    with open(pcr_template, 'r') as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            if "!File names of data(patterns) files" in line and i+1 < len(lines):
                expected_dat_name = lines[i+1].strip()
                print(f"Found expected DAT filename in PCR: '{expected_dat_name}'")
                break
    
    if expected_dat_name:
        cnn_dat_path = os.path.join(cnn_folder, expected_dat_name)
        print(f"Copying DAT file with the name expected in PCR: {expected_dat_name}")
    else:
        material_name = os.path.basename(identified_dir).split('_')[1]  # Extract material name
        expected_dat_name = f"{material_name}.dat"  # Use material name for DAT
        cnn_dat_path = os.path.join(cnn_folder, expected_dat_name)
        print(f"Using material name for DAT file: {expected_dat_name}")
        
        with open(cnn_pcr_path, 'r') as f:
            pcr_content = f.read()
        
        with open(cnn_pcr_path, 'w') as f:
            for line in pcr_content.splitlines(True):
                if "!File names of data(patterns) files" in line:
                    f.write(line)
                    f.write(f"{expected_dat_name}\n")
                else:
                    f.write(line)
    
    shutil.copy2(dat_file, cnn_dat_path)
    
    print(f"Copied DAT file => {cnn_dat_path}")
    print("\nRefinement files prepared successfully.")
    
    with open(cnn_pcr_path, 'r') as f:
        pcr_content = f.readlines()
        for i, line in enumerate(pcr_content):
            if "!File names of data(patterns) files" in line and i+1 < len(pcr_content):
                dat_in_pcr = pcr_content[i+1].strip()
                if dat_in_pcr != os.path.basename(cnn_dat_path):
                    print(f"Warning: PCR file references '{dat_in_pcr}' but we copied to '{os.path.basename(cnn_dat_path)}'")
                else:
                    print(f"Verified: PCR file correctly references '{dat_in_pcr}'")
                break
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python Rietveld_Refinement_step1_MI.py <identified_material_folder>")
        sys.exit(1)

    identified_dir = sys.argv[1]
    if not os.path.exists(identified_dir):
        print(f"Error: Identified material folder not found: {identified_dir}")
        sys.exit(1)

    catalog = find_catalog_file()
    if not catalog:
        sys.exit(1)

    material_name, param_file, dat_file = find_material_info(identified_dir, catalog)
    if not material_name or not param_file or not dat_file:
        sys.exit(1)

    params = read_refined_parameters(param_file)
    if not params:
        sys.exit(1)

    pcr_template = find_pcr_template(material_name, catalog)
    if not pcr_template:
        sys.exit(1)

    refine_root, cnn_folder = setup_refinement_folders(identified_dir)

    if not prepare_refinement_files(params, pcr_template, identified_dir, dat_file, refine_root, cnn_folder):
        sys.exit(1)

    print("\nStep 1 completed. Ready for Rietveld refinement step 2.")

if __name__ == "__main__":
    main()