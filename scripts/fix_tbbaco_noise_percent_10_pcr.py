# -*- coding: utf-8 -*-

import os
import re
import sys

def find_reference_pcr_file(pcr_file_path):

    current_dir = os.path.dirname(pcr_file_path)
    
    for i in range(5):  # Try up to 5 levels up
        ref_path = os.path.join(current_dir, 'dat_vestacif_files', 'reference_pcr_format', 'tbbaco_noise_percent_10.pcr')
        if os.path.exists(ref_path):
            print("Found reference tbbaco_noise_percent_10.pcr at: {}".format(ref_path))
            return ref_path
            
        ref_path = os.path.join(current_dir, 'dat_vestacif_files', 'tbbaco_noise_percent_10.pcr')
        if os.path.exists(ref_path):
            print("Found reference tbbaco_noise_percent_10.pcr at: {}".format(ref_path))
            return ref_path
        
        current_dir = os.path.dirname(current_dir)
    
    print("Warning: Reference tbbaco_noise_percent_10.pcr file not found after searching up directories")
    print("Searched from: {}".format(os.path.dirname(pcr_file_path)))
    return None

def extract_biso_values_from_reference(ref_pcr_path):

    biso_values = {}
    
    try:
        with open(ref_pcr_path, 'r') as file:
            lines = file.readlines()
            
        in_atom_section = False
        for i, line in enumerate(lines):
            if '!Atom   Typ       X        Y        Z     Biso' in line:
                in_atom_section = True
                continue
            elif in_atom_section and '!-------> Profile Parameters' in line:
                break
            elif in_atom_section and re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', line):
                parts = line.split()
                if len(parts) >= 6:
                    atom_name = parts[0]
                    biso_value = parts[5]
                    biso_values[atom_name] = biso_value
                    print("Found reference Biso for {}: {}".format(atom_name, biso_value))
        
        return biso_values
    
    except Exception as e:
        print("Error reading reference PCR file: {}".format(e))
        return {}

def fix_tbbaco_noise_percent_10_pcr(pcr_file_path):

    try:
        ref_pcr_path = find_reference_pcr_file(pcr_file_path)
        if not ref_pcr_path:
            print("Warning: Could not find reference tbbaco_noise_percent_10.pcr file, skipping fixes")
            return False
        
        ref_biso_values = extract_biso_values_from_reference(ref_pcr_path)
        if not ref_biso_values:
            print("Warning: No Biso values found in reference file")
            return False
        
        with open(pcr_file_path, 'r') as file:
            lines = file.readlines()
        
        in_atom_section = False
        atom_line_count = 0
        
        for i, line in enumerate(lines):
            if '!Atom   Typ       X        Y        Z     Biso' in line:
                in_atom_section = True
                continue
            elif in_atom_section and '!-------> Profile Parameters' in line:
                in_atom_section = False
                break
            
            if in_atom_section and re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', line):
                parts = line.split()
                if len(parts) >= 7:
                    atom_name = parts[0]
                    
                    if atom_name in ref_biso_values:
                        ref_biso = ref_biso_values[atom_name]
                        parts[5] = ref_biso
                        
                        new_line = "{:<2} {:<2}      {}  {}  {}  {}   {}   0   0   2    0  \n".format(
                            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6])
                        lines[i] = new_line
                        
                        print("Updated {} Biso from {} to {}".format(atom_name, line.split()[5], ref_biso))
                    
                    if i+1 < len(lines):
                        lines[i+1] = "                  0.00     0.00     0.00     0.00      0.00\n"
                        print("Set Biso codeword to 0.00 for {}".format(atom_name))
        
        with open(pcr_file_path, 'w') as file:
            file.writelines(lines)
        
        print("Successfully fixed tbbaco_noise_percent_10 PCR file Biso values and codewords")
        return True
    
    except Exception as e:
        print("Error fixing tbbaco_noise_percent_10 PCR file: {}".format(e))
        return False

def fix_tbbaco_noise_percent_10_pcr_with_root(pcr_file_path, root_dir):

    try:
        ref_pcr_path = os.path.join(root_dir, 'dat_vestacif_files', 'reference_pcr_format', 'tbbaco_noise_percent_10.pcr')
        if not os.path.exists(ref_pcr_path):
            ref_pcr_path = os.path.join(root_dir, 'dat_vestacif_files', 'tbbaco_noise_percent_10.pcr')
        
        if not os.path.exists(ref_pcr_path):
            print("Warning: Reference tbbaco_noise_percent_10.pcr not found in root directory: {}".format(root_dir))
            return False
        
        print("Using reference tbbaco_noise_percent_10.pcr from: {}".format(ref_pcr_path))
        
        ref_biso_values = extract_biso_values_from_reference(ref_pcr_path)
        if not ref_biso_values:
            print("Warning: No Biso values found in reference file")
            return False
        
        with open(pcr_file_path, 'r') as file:
            lines = file.readlines()
        
        in_atom_section = False
        
        for i, line in enumerate(lines):
            if '!Atom   Typ       X        Y        Z     Biso' in line:
                in_atom_section = True
                continue
            elif in_atom_section and '!-------> Profile Parameters' in line:
                in_atom_section = False
                break
            
            if in_atom_section and re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', line):
                parts = line.split()
                if len(parts) >= 7:
                    atom_name = parts[0]
                    
                    if atom_name in ref_biso_values:
                        ref_biso = ref_biso_values[atom_name]
                        parts[5] = ref_biso
                        
                        new_line = "{:<2} {:<2}      {}  {}  {}  {}   {}   0   0   2    0  \n".format(
                            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6])
                        lines[i] = new_line
                        
                        print("Updated {} Biso from {} to {}".format(atom_name, line.split()[5], ref_biso))
                    
                    if i+1 < len(lines):
                        lines[i+1] = "                  0.00     0.00     0.00     0.00      0.00\n"
                        print("Set Biso codeword to 0.00 for {}".format(atom_name))
        
        with open(pcr_file_path, 'w') as file:
            file.writelines(lines)
        
        print("Successfully fixed tbbaco_noise_percent_10 PCR file Biso values and codewords")
        return True
    
    except Exception as e:
        print("Error fixing tbbaco_noise_percent_10 PCR file: {}".format(e))
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_tbbaco_noise_percent_10_pcr.py <path_to_pcr_file> [root_dir]")
        sys.exit(1)
    
    pcr_file_path = sys.argv[1]
    if not os.path.exists(pcr_file_path):
        print("Error: PCR file '{}' does not exist".format(pcr_file_path))
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        root_dir = sys.argv[2]
        fix_tbbaco_noise_percent_10_pcr_with_root(pcr_file_path, root_dir)
    else:
        fix_tbbaco_noise_percent_10_pcr(pcr_file_path)