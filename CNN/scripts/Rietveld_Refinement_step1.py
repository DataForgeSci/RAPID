import os
import sys
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def read_ml_inputs(ml_file):

    if not os.path.exists(ml_file):
        print(f"Error: {ml_file} not found. Exiting.")
        sys.exit(1)

    lines = []
    with open(ml_file, 'r') as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                # skip blank & comment lines from the count
                continue
            lines.append(t)
            if len(lines) == 7:
                break

    if len(lines) < 5:
        print("Error: The file must have at least 5 non-comment, non-blank lines. Exiting.")
        sys.exit(1)

    line1 = lines[0].strip()
    model_name = "default_model"
    if ";" in line1:
        parts = line1.split(";", 1)
        model_name = parts[1].strip() if len(parts) > 1 else "default_model"

    dataset_line = lines[2].strip()
    experimental_line = lines[3].strip()
    refine_line = lines[4].lower().strip()
    
    omit_background = False
    if len(lines) >= 7:
        omit_background = lines[6].lower().strip() == 'y'

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
        print("No dataset folder name found on line3. Exiting.")
        sys.exit(1)

    for dataset_name in dataset_folders:
        print(f"\nProcessing refinement for dataset: {dataset_name}")
    return model_name, dataset_folders, experimental_line, refine_line, omit_background

def read_refined_parameters(refined_param_file, omit_background=False):

    params = {}
    if not os.path.isfile(refined_param_file):
        print(f"Refined param file not found: {refined_param_file}")
        return params

    print(f"Reading refined parameters from: {refined_param_file}")
    print(f"Background parameter omitted: {omit_background}")

    crystal_structure = "cubic"  
    biso_atoms = []  
    
    with open(refined_param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
                
            key_part, val_part = line.split('=', 1)
            key = key_part.strip().lower()  # Convert to lowercase for case-insensitive matching
            
            if key == 'lattice a' and crystal_structure == "cubic":
                crystal_structure = "cubic"  # Single a parameter
            elif key == 'lattice b':
                crystal_structure = "orthorhombic"  # Structure has b parameter
            elif key == 'lattice c' and crystal_structure == "cubic":
                crystal_structure = "tetragonal"  # c different from a means tetragonal
            elif key == 'alpha' and float(val_part.strip()) != 90.0:
                crystal_structure = "triclinic"  # Non-90 alpha
            elif key == 'beta' and float(val_part.strip()) != 90.0:
                crystal_structure = "monoclinic"  # Non-90 beta
            elif key == 'gamma' and float(val_part.strip()) != 90.0:
                if crystal_structure == "cubic":
                    crystal_structure = "hexagonal"  # Only gamma ≠ 90°
                elif crystal_structure == "triclinic":
                    pass  # Already identified as triclinic
                else:
                    crystal_structure = "trigonal"  # Different classification
                
            if key.startswith("biso"):
                atom_parts = key.split()
                if len(atom_parts) > 1:
                    atom_name = atom_parts[1].strip().lower()  # Lowercase for consistency
                    if atom_name and atom_name not in biso_atoms:
                        biso_atoms.append(atom_name)

    print(f"Detected crystal structure: {crystal_structure}")
    if biso_atoms:
        print(f"Detected atoms with Biso parameters: {', '.join(biso_atoms)}")
    
    with open(refined_param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if (not line) or ('=' not in line):
                continue
            key_part, val_part = line.split('=', 1)
            key = key_part.strip()
            key_lower = key.lower()  # Lowercase for matching
            val_str = val_part.strip()
            try:
                val = float(val_str)
            except:
                val = 0.0

            if omit_background and key_lower == 'background':
                print(f"Ignoring Background parameter (value {val}) as it was omitted from CNN training")
                continue
                
            if key_lower.startswith("biso"):
                atom_parts = key_lower.split()
                if len(atom_parts) > 1:
                    atom_name = atom_parts[1].strip()
                    params[key] = val
                    
                    if "atom_biso" not in params:
                        params["atom_biso"] = {}
                    params["atom_biso"][atom_name] = val
                    print(f"Found Biso for atom '{atom_name}' with value: {val}")
            else:
                print(f"Found parameter '{key}' with value: {val}")
                params[key] = val
    
    if omit_background and 'Background' not in params:
        print(f"Background parameter is NOT included in refined parameters (will use original PCR value)")

    params["_crystal_structure"] = crystal_structure
    params["_biso_atoms"] = biso_atoms

    print(f"Final refined parameters dictionary: {params}")
    return params

def create_pcr_with_cnn_params(original_pcr, new_pcr, param_dict, omit_background=False):

    if not os.path.exists(original_pcr):
        print(f"Original PCR not found: {original_pcr}")
        return False

    with open(original_pcr, 'r') as f:
        lines = f.readlines()

    crystal_structure = param_dict.get("_crystal_structure", "cubic")
    biso_atoms = param_dict.get("_biso_atoms", [])
    
    print(f"Creating PCR file for {crystal_structure} structure with {len(biso_atoms)} atom types")
    print(f"Atom Biso values to apply: {[atom for atom in biso_atoms]}")

    atom_biso_values = {}
    for key, val in param_dict.items():
        if key.lower().startswith("biso"):
            atom_parts = key.lower().split()
            if len(atom_parts) > 1:
                atom_name = atom_parts[1].strip()  # Extract atom name and convert to lowercase
                atom_biso_values[atom_name] = val
                print(f"Atom Biso mapping: {atom_name} -> {val}")
    
    if "atom_biso" in param_dict:
        for atom_name, val in param_dict["atom_biso"].items():
            atom_name_lower = atom_name.lower()
            if atom_name_lower not in atom_biso_values:
                atom_biso_values[atom_name_lower] = val
                print(f"Additional atom Biso from atom_biso: {atom_name_lower} -> {val}")

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
                if omit_background:
                    new_lines.append(lines[i + 1])
                    print(f"Preserving original background coefficients (background was omitted from CNN training)")
                    found_bg = True
                else:
                    original_line = lines[i + 1]
                    bg_val = param_dict.get('Background', 0.0)
                    bg_parts = original_line.split()
                    if len(bg_parts) >= 6:
                        new_line = "      " + f"{bg_val:8.3f}" + "     " + "     ".join(bg_parts[2:]) + "\n"
                    else:
                        new_line = "      " + f"{bg_val:8.3f}" + "     -19.888      10.065      -2.284       0.213       0.000\n"
                    new_lines.append(new_line)
                    found_bg = True
            continue

        elif "!     a          b         c        alpha      beta       gamma      #Cell Info" in line:
            new_lines.append(line)
            if i + 1 < len(lines):
                if crystal_structure == "cubic":
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = a_val  # same as a
                    c_val = a_val  # same as a
                    alpha_val = 90.0
                    beta_val = 90.0
                    gamma_val = 90.0
                elif crystal_structure == "tetragonal":
                    a_val = param_dict.get('Lattice a', 5.430)
                    b_val = a_val  # same as a
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
                    b_val = a_val  # same as a
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
                    b_val = a_val  # same as a
                    c_val = a_val  # same as a
                    alpha_val = param_dict.get('Alpha', 90.0)
                    beta_val = alpha_val  # same as alpha
                    gamma_val = alpha_val  # same as alpha
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
                
                if atom_name in atom_biso_values:
                    biso_val = atom_biso_values[atom_name]
                else:
                    for atom_key, atom_val in atom_biso_values.items():
                        if atom_key.lower() == atom_name.lower() or atom_key.lower() in atom_name.lower() or atom_name.lower() in atom_key.lower():
                            biso_val = atom_val
                            print(f"Found Biso match for {atom_name} using key {atom_key} -> {biso_val}")
                            break
                
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

    for atom in biso_atoms:
        atom_lower = atom.lower()
        if atom_lower not in processed_atoms:
            print(f"Warning: Atom '{atom}' from refined parameters not found in PCR file")

    with open(new_pcr, 'w') as ff:
        ff.writelines(new_lines)

    missing_params = []
    if not found_zero:
        missing_params.append("Zero shift")
    if not found_bg and not omit_background:
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


def organize_refinement_folders(refine_root, new_pcr_path, original_pcr, original_dat):

    cnn_folder = os.path.join(refine_root, "CNN_ML_refinement")
    soln_folder = os.path.join(refine_root, "solution_refinement")
    os.makedirs(cnn_folder, exist_ok=True)
    os.makedirs(soln_folder, exist_ok=True)
    
    try:
        cnn_pcr_dest = os.path.join(cnn_folder, os.path.basename(new_pcr_path))
        shutil.move(new_pcr_path, cnn_pcr_dest)
        
        soln_pcr_dest = os.path.join(soln_folder, os.path.basename(original_pcr))
        shutil.copy2(original_pcr, soln_pcr_dest)
        
        dat_name = os.path.basename(original_dat)
        shutil.copy2(original_dat, os.path.join(cnn_folder, dat_name))
        shutil.copy2(original_dat, os.path.join(soln_folder, dat_name))
        
        print(f"\nCreated folder structure:")
        print(f"  {cnn_folder}/")
        print(f"    - {os.path.basename(new_pcr_path)}")
        print(f"    - {dat_name}")
        print(f"  {soln_folder}/")
        print(f"    - {os.path.basename(original_pcr)}")
        print(f"    - {dat_name}")
        
    except Exception as e:
        print(f"Error organizing refinement folders: {e}")


def main():

    if len(sys.argv) < 2:
        print("Usage: python Rietveld_Refinement.py <path_to_inputs>")
        sys.exit(1)

    inputs_file = sys.argv[1]
    model_name, dataset_folders, experimental_dat, refine_line, omit_background = read_ml_inputs(inputs_file)

    if refine_line != 'y':
        print("User said 'n' for Rietveld refinement (line5). Exiting without refinement.")
        return

    for dataset_name in dataset_folders:
        print(f"\nProcessing refinement for dataset: {dataset_name}")
        print(f"Background parameter will {'be preserved (omitted from CNN)' if omit_background else 'use CNN prediction'}")
        
        dataset_path = os.path.join('data', 'train_data', dataset_name)
        if not os.path.isdir(dataset_path):
            print(f"Error: dataset path '{dataset_path}' not found.")
            continue  

        source_dir = os.path.join(dataset_path, 'source_files')
        if not os.path.isdir(source_dir):
            print(f"Error: source_files not found at '{source_dir}'")
            continue  

        original_pcr = None
        original_dat = None
        for f in os.listdir(source_dir):
            fp = os.path.join(source_dir, f)
            if f.lower().endswith('.pcr'):
                original_pcr = fp
            elif f.lower().endswith('.dat'):
                original_dat = fp

        if not original_pcr or not original_dat:
            print(f"Missing .pcr or .dat in source_files for {dataset_name} => cannot proceed.")
            continue  

        experiment_base = os.path.splitext(experimental_dat)[0]
        model_dir = os.path.join("saved_models", model_name,
                                "refinement_result", dataset_name, experiment_base)

        if not os.path.exists(model_dir):
            print(f"Error: {model_dir} does not exist.")
            continue 

        refined_param = None
        for f in os.listdir(model_dir):
            if f.endswith('_refined_parameters.dat'):
                refined_param = os.path.join(model_dir, f)
                break
        if not refined_param:
            print(f"No *_refined_parameters.dat found in {model_dir} => cannot proceed.")
            continue  

        refine_root = os.path.join(model_dir, "Rietveld_Refinement")
        os.makedirs(refine_root, exist_ok=True)

        param_dict = read_refined_parameters(refined_param, omit_background)
        new_pcr_path = os.path.join(refine_root, 'cnn_refined.pcr')
        
        success = create_pcr_with_cnn_params(original_pcr, new_pcr_path, param_dict, omit_background)
        
        if not success:
            continue  

        remove_duplicate_parameter_lines(new_pcr_path)

        organize_refinement_folders(refine_root, new_pcr_path, original_pcr, original_dat)

        print("\nRietveld refinement steps completed successfully. All files in:")
        print(refine_root)

if __name__=="__main__":
    main()