import re
import os
import pandas as pd
import shutil
import sys
import time
import matplotlib.pyplot as plt 
import numpy as np 
import random
import itertools
from mpl_toolkits.mplot3d import Axes3D
import sobol_seq

def align_zero_parameter(lines):
    zero_line_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('!  Zero    Code    SyCos    Code   SySin    Code  Lambda     Code MORE ->Patt# 1'):
            zero_line_index = i + 1
            break

    if zero_line_index == -1:
        print("Zero parameter line not found.")
        return lines

    zero_value_str = re.findall(r"[-+]?\d*\.\d+|\d+", lines[zero_line_index])[0]
    zero_length = len(str(int(float(zero_value_str))))  # Number of digits before the decimal point

    shift = max(0, zero_length - 1)

    adjusted_line = lines[zero_line_index][shift:] if shift > 0 else lines[zero_line_index]

    parts = re.split(r"(\s+)", adjusted_line)
    for i in range(len(parts)):
        if '-' in parts[i]:
            parts[i - 1] = parts[i - 1][:-1]  # Remove one space from the part before the negative value
            break
    adjusted_line = ''.join(parts)

    lines[zero_line_index] = adjusted_line

    return lines

# def align_background_parameter(lines):
#     background_line_index = -1
#     for i, line in enumerate(lines):
#         if line.strip().startswith('!   Background coefficients/codes  for Pattern#  1  (Polynomial of 6th degree)'):
#             background_line_index = i + 1
#             break

#     if background_line_index == -1:
#         print("Background parameters line not found.")
#         return lines

#     background_line = lines[background_line_index]

#     bg_value_match = re.search(r"[-+]?\d*\.\d+", background_line)
#     if bg_value_match:
#         bg_value_length = len(bg_value_match.group().split('.')[0])  # Length of the integer part

#         if bg_value_length >= 2:
#             shift = bg_value_length - 1  # For 10th digit, shift left by 1; for 100th digit, shift left by 2
#             background_line = background_line[shift:]
#         elif bg_value_length == 1:
#             background_line = ' ' + background_line

#     parts = re.split(r"(\s+)", background_line)
#     for i in range(len(parts)):
#         if '-' in parts[i]:
#             parts[i - 1] = parts[i - 1][:-1]  # Remove one space from the part before the negative value
#             break
#     adjusted_background_line = ''.join(parts)

#     lines[background_line_index] = adjusted_background_line

#     return lines

def align_background_parameter(lines):
    background_line_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('!   Background coefficients/codes  for Pattern#  1  (Polynomial of 6th degree)'):
            background_line_index = i + 1
            break

    if background_line_index == -1:
        print("Background parameters line not found.")
        return lines

    background_parts = lines[background_line_index].split()
    if background_parts:
        lines[background_line_index] = "   " + "  ".join(background_parts) + "\n"

    return lines

def align_lattice_parameters(lines):
    lattice_line_index = -1
    for i, line in enumerate(lines):
        if '!     a          b         c        alpha      beta       gamma      #Cell Info' in line:
            lattice_line_index = i + 1
            break

    if lattice_line_index == -1 or lattice_line_index + 1 >= len(lines):
        print("Lattice parameters line or the line below it not found.")
        return lines

    lattice_params_line = lines[lattice_line_index].strip()
    reference_line = lines[lattice_line_index + 1].strip()

    lattice_params = [part for part in re.split(r"(\s+)", lattice_params_line) if part.strip() != ""]
    reference_params = [part for part in re.split(r"(\s+)", reference_line) if part.strip() != ""]

    df = pd.DataFrame([reference_params, lattice_params]).T

    for i in range(len(lattice_params)):
        df.iloc[i, 1] = "{:10.6f}".format(float(lattice_params[i]))

    aligned_lattice_params_line = ' ' + ' '.join(df[1].tolist())
    lines[lattice_line_index] = aligned_lattice_params_line + '\n'

    return lines

def append_spaces_to_lattice_line(pcr_file):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    target_line_index = -1
    for i, line in enumerate(lines):
        if '!     a          b         c        alpha      beta       gamma      #Cell Info' in line:
            target_line_index = i + 1
            break

    if target_line_index != -1 and target_line_index < len(lines):
        lines[target_line_index] = lines[target_line_index].rstrip() + '   \n'

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def create_folders_and_move_pcr_files(subfolder_path, pcr_filenames, dataset_status):
    for pcr_filename in pcr_filenames:
        folder_name = os.path.splitext(pcr_filename)[0]
        folder_path = os.path.join(subfolder_path, folder_name)

        dataset_status[pcr_filename] = 'generated'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        original_pcr_path = os.path.join(subfolder_path, pcr_filename)
        new_pcr_path = os.path.join(folder_path, pcr_filename)

        try:
            if os.path.exists(original_pcr_path):
                shutil.move(original_pcr_path, new_pcr_path)
                dataset_status[pcr_filename] = 'moved'  # Update status
                # print "Moved PCR file {} into folder {}".format(pcr_filename, folder_path)

            else:
                dataset_status[pcr_filename] = 'missing'
        except Exception as e:
            dataset_status[pcr_filename] = 'error'
            print "Error moving file {}: {}".format(original_pcr_path, e)

def copy_paste_dat_file(subfolder_path, pcr_filename, dat_file_name, dataset_status):
    folder_name = os.path.splitext(pcr_filename)[0]
    folder_path = os.path.join(subfolder_path, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    original_dat_path = os.path.join(subfolder_path, dat_file_name)
    new_dat_path = os.path.join(folder_path, dat_file_name)

    if os.path.exists(original_dat_path):
        shutil.copy(original_dat_path, new_dat_path)
    else:
        # Mark .dat file as missing
        dataset_status[dat_file_name] = 'missing'


def get_random_value(base_value, variation, scale=False):
    if scale:
        delta = base_value * (variation / 100.0)
    else:
        delta = variation
    return base_value + random.uniform(-delta, delta)


def read_original_values(pcr_file_path, atom_format='4-line'):
    original_values = {}
    with open(pcr_file_path, 'r') as file:
        lines = file.readlines()

    # Find atom section boundaries
    atom_section_start = -1
    atom_section_end = -1
    
    for i, line in enumerate(lines):
        if '!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes' in line:
            atom_section_start = i + 1  # Start from the line after the header
        elif atom_section_start > 0 and '!-------> Profile Parameters for Pattern #' in line:
            atom_section_end = i
            break

    atom_data = []
    atom_biso_values = {}
    
    lines_per_atom = 2 if atom_format.lower() == '2-line' else 4
    
    if atom_section_start > 0 and atom_section_end > 0:
        current_line = atom_section_start
        while current_line < atom_section_end:
            if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+[-+]?\d*\.\d+', lines[current_line]):
                parts = lines[current_line].split()
                if len(parts) >= 6:
                    atom_name = parts[0]
                    atom_type = parts[1]
                    biso_value = parts[5]
                    atom_data.append({
                        'name': atom_name,
                        'type': atom_type,
                        'biso': biso_value
                    })
                    atom_biso_values[atom_name] = biso_value
                current_line += lines_per_atom
            else:
                current_line += 1
    
    original_values['atoms'] = atom_data
    original_values['atom_biso_values'] = atom_biso_values

    for i, line in enumerate(lines):
        if '!  Zero    Code    SyCos    Code   SySin    Code  Lambda     Code MORE ->Patt# 1' in line:
            zero_line_index = i + 1
            parts = re.split(r"(\s+)", lines[zero_line_index])
            original_values['zero'] = parts[2]

        elif '!   Background coefficients/codes  for Pattern#  1  (Polynomial of 6th degree)' in line:
            bg_line_index = i + 1
            parts = re.split(r"(\s+)", lines[bg_line_index])
            original_values['bg'] = parts[2]

        elif '!     a          b         c        alpha      beta       gamma      #Cell Info' in line:
            lattice_line_index = i + 1
            parts = re.split(r"(\s+)", lines[lattice_line_index])
            original_values['a'] = parts[2]
            original_values['b'] = parts[4]
            original_values['c'] = parts[6]
            original_values['alpha'] = parts[8]
            original_values['beta'] = parts[10]
            original_values['gamma'] = parts[12]

        elif '!  Scale        Shape1      Bov      Str1      Str2      Str3   Strain-Model' in line:
            if i + 1 < len(lines):
                scale_line = lines[i + 1].strip()
                parts = scale_line.split()
                if len(parts) >= 1:
                    original_values['scale'] = parts[0]

        elif '!       U         V          W           X          Y        GauSiz   LorSiz Size-Model' in line:
            if i + 1 < len(lines):
                uvw_line = lines[i + 1].strip()
                parts = uvw_line.split()
                if len(parts) >= 3:
                    original_values['U'] = parts[0]
                    original_values['V'] = parts[1]
                    original_values['W'] = parts[2]

    if 'atoms' in original_values and len(original_values['atoms']) > 0:
        original_values['biso'] = original_values['atoms'][0]['biso']
    
    return original_values

def modify_combined_parameters(pcr_file_path, zero_value, background_value, a_value, b_value, 
                             c_value, alpha_value, beta_value, gamma_value, 
                             biso_shifts, scale_value, u_value, v_value, w_value, 
                             combined_shift_str, shift_strs, lattice_type, bg_range=0, atom_format='4-line'):
    with open(pcr_file_path, 'r') as file:
        lines = file.readlines()

    atom_section_start = -1
    atom_section_end = -1
    
    for i, line in enumerate(lines):
        if '!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes' in line:
            atom_section_start = i + 1  # Start from the line after the header
        elif atom_section_start > 0 and '!-------> Profile Parameters for Pattern #' in line:
            atom_section_end = i
            break
    
    lines_per_atom = 2 if atom_format.lower() == '2-line' else 4
    
    if atom_section_start > 0 and atom_section_end > 0:
        current_line = atom_section_start
        atom_index = 0
        
        while current_line < atom_section_end:
            if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+[-+]?\d*\.\d+', lines[current_line]):
                parts = lines[current_line].split()
                if len(parts) >= 6:
                    try:
                        current_biso = float(parts[5])
                        
                        if isinstance(biso_shifts, list) and atom_index < len(biso_shifts):
                            atom_shift = biso_shifts[atom_index]
                            new_biso = current_biso * (1 + atom_shift / 100.0)
                        else:
                            new_biso = current_biso * (1 + biso_shifts / 100.0)
                            
                        parts[5] = "{:.5f}".format(new_biso)
                        
                        if lines_per_atom == 2:
                            if len(parts[0]) <= 2:  # Short atom name (e.g., "O")
                                new_line = "{:6s} {:7s} {:8s} {:8s} {:8s} {:8s} {:8s} {:3s} {:3s} {:3s} {:3s}\n".format(
                                    parts[0], parts[1], parts[2], parts[3], parts[4],
                                    parts[5], parts[6], parts[7], parts[8], parts[9], parts[10] if len(parts) > 10 else ""
                                )
                            else:  # Longer atom name (e.g., "Ce")
                                new_line = "{:3s} {:3s} {:8s} {:8s} {:8s} {:8s} {:8s} {:3s} {:3s} {:3s} {:3s}\n".format(
                                    parts[0], parts[1], parts[2], parts[3], parts[4],
                                    parts[5], parts[6], parts[7], parts[8], parts[9], parts[10] if len(parts) > 10 else ""
                                )
                        else:
                            # For 4-line format (traditional PCR)
                            new_line = "{:4s} {:4s} {:8s} {:8s} {:8s} {:8s} {:8s} {:3s} {:3s} {:3s} {:4s}\n".format(
                                parts[0], parts[1], parts[2], parts[3], parts[4],
                                parts[5], parts[6], parts[7], parts[8], parts[9], parts[10] if len(parts) > 10 else ""
                            )
                        
                        lines[current_line] = new_line
                        atom_index += 1
                    except (ValueError, IndexError) as e:
                        print("Error updating Biso value for atom at line {}: {}".format(current_line+1, e))
                current_line += lines_per_atom
            else:
                current_line += 1

    for i, line in enumerate(lines):
        if '!  Zero    Code    SyCos    Code   SySin    Code  Lambda     Code MORE ->Patt# 1' in line:
            parts = re.split(r"(\s+)", lines[i + 1])
            parts[2] = "{:.6f}".format(float(zero_value))
            lines[i + 1] = ''.join(parts)

        elif '!   Background coefficients/codes  for Pattern#  1  (Polynomial of 6th degree)' in line:
            if bg_range == 0:
                continue  # Skip any modification to preserve original background formatting
            else:
                parts = re.split(r"(\s+)", lines[i + 1])
                parts[2] = "{:.6f}".format(float(background_value))
                lines[i + 1] = ''.join(parts)

        elif '!     a          b         c        alpha      beta       gamma      #Cell Info' in line:
            parts = re.split(r"(\s+)", lines[i + 1])
            # For exact values from reference files, use string representation
            parts[2] = str(a_value)
            parts[4] = str(b_value)
            parts[6] = str(c_value)
            parts[8] = str(alpha_value)
            parts[10] = str(beta_value)
            parts[12] = str(gamma_value)
            lines[i + 1] = ''.join(parts)

        elif '!  Scale        Shape1      Bov      Str1      Str2      Str3   Strain-Model' in line:
            parts = lines[i + 1].split()
            
            original_value = parts[0]
            uses_scientific = 'E' in original_value or 'e' in original_value
            
            if uses_scientific:
                parts[0] = "{:.7E}".format(float(scale_value))
            else:
                parts[0] = "{:.7f}".format(float(scale_value))
            
            new_line = "  {} {:8} {:8} {:8} {:8} {:8}       {}\n".format(
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
            )
            lines[i + 1] = new_line

        elif '!       U         V          W           X          Y        GauSiz   LorSiz Size-Model' in line:
            parts = lines[i + 1].split()
            parts[0] = "{:10.6f}".format(float(u_value))
            parts[1] = "{:11.6f}".format(float(v_value))
            parts[2] = "{:11.6f}".format(float(w_value))
            new_line = "   {:>10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}       {}\n".format(
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]
            )
            lines[i + 1] = new_line

    lines = align_zero_parameter(lines)
    lines = align_background_parameter(lines)
    lines = align_lattice_parameters(lines)

    import hashlib
    param_hash = hashlib.md5(str(combined_shift_str).encode()).hexdigest()[:10]
    base_pcr_name = os.path.basename(pcr_file_path)
    base_name_without_ext = os.path.splitext(base_pcr_name)[0]
    new_filename = "{}_shift_{}.pcr".format(base_name_without_ext, param_hash)
    
    dir_path = os.path.dirname(pcr_file_path)
    new_pcr_file_path = os.path.join(dir_path, new_filename)
    
    with open(new_pcr_file_path, 'w') as file:
        file.writelines(lines)
    
    return new_pcr_file_path

def generate_parameter_sets_from_input(zero_range, bg_range, lattice_range, biso_range, 
                                      scale_range, u_range, v_range, w_range, 
                                      lattice_type, uniform_count, random_count, num_atoms):
    tweaked_indices = []
    ranges = []
    
    if zero_range != 0:
        tweaked_indices.append(0)
        ranges.append(zero_range)
    if bg_range != 0:
        tweaked_indices.append(1)
        ranges.append(bg_range)
    if lattice_range != 0:
        if lattice_type == 'cubic':
            tweaked_indices.append(2)
            ranges.append(lattice_range)
        elif lattice_type in ['tetragonal', 'hexagonal']:
            tweaked_indices.extend([2, 3])
            ranges.extend([lattice_range, lattice_range])
        elif lattice_type == 'orthorhombic':
            tweaked_indices.extend([2, 3, 4])
            ranges.extend([lattice_range, lattice_range, lattice_range])
        elif lattice_type == 'monoclinic':
            tweaked_indices.extend([2, 3, 4, 5])
            ranges.extend([lattice_range] * 4)
        elif lattice_type == 'triclinic':
            tweaked_indices.extend([2, 3, 4, 5, 6, 7])
            ranges.extend([lattice_range] * 6)
        elif lattice_type == 'trigonal':
            tweaked_indices.extend([2, 3])
            ranges.extend([lattice_range, lattice_range])

    base_idx = 8
    
    if biso_range != 0:
        for i in range(num_atoms):
            tweaked_indices.append(base_idx + i)
            ranges.append(biso_range)
    
    next_idx = base_idx + num_atoms
    if scale_range != 0:
        tweaked_indices.append(next_idx)
        ranges.append(scale_range)
        next_idx += 1
    if u_range != 0:
        tweaked_indices.append(next_idx)
        ranges.append(u_range)
        next_idx += 1
    if v_range != 0:
        tweaked_indices.append(next_idx)
        ranges.append(v_range)
        next_idx += 1
    if w_range != 0:
        tweaked_indices.append(next_idx)
        ranges.append(w_range)
        next_idx += 1

    total_params = next_idx
    
    if not tweaked_indices:
        point = [0] * total_params  # Make room for all parameters
        return [point], [point]

    if len(tweaked_indices) == 1:
        uniform_combinations = []
        random_combinations = []
        tweaked_idx = tweaked_indices[0]
        range_val = ranges[0]

        print "Tweaking parameter at index {} with range {}".format(tweaked_idx, range_val)

        uniform_values = np.linspace(-range_val, range_val, uniform_count)
        uniform_values = np.round(uniform_values, 10)
        
        for val in uniform_values:
            point = [0] * total_params  # Initialize with zeros for all parameters
            point[tweaked_idx] = val
            uniform_combinations.append(tuple(point))
        
        print "Number of uniform points generated: {}".format(len(uniform_combinations))

        unique_points = set(uniform_combinations)
        attempt_count = 0
        
        while len(random_combinations) < random_count:
            point = [0] * total_params
            point[tweaked_idx] = round(random.uniform(-range_val, range_val), 8)
            point_tuple = tuple(point)
            
            if point_tuple not in unique_points:
                random_combinations.append(point_tuple)
                unique_points.add(point_tuple)
            
            attempt_count += 1
            if attempt_count > random_count * 10:  # Prevent infinite loop
                break

        print "Number of random points generated: {}".format(len(random_combinations))
        print "Total attempts: {}".format(attempt_count)

        return uniform_combinations, random_combinations

    else:
        if is_perfect_cube(uniform_count):
            points_per_dim = int(round(uniform_count ** (1.0/len(tweaked_indices))))
            
            param_points = []
            for range_val in ranges:
                points = np.linspace(-range_val, range_val, points_per_dim)
                param_points.append(points)

            uniform_combinations = []
            for point in itertools.product(*param_points):
                full_point = [0] * total_params  # Initialize with zeros
                for idx, val in zip(tweaked_indices, point):
                    full_point[idx] = val
                uniform_combinations.append(tuple(full_point))
        else:
            sobol_points = sobol_seq.i4_sobol_generate(len(tweaked_indices), uniform_count)
            
            uniform_combinations = []
            for point in sobol_points:
                full_point = [0] * total_params  # Initialize with zeros
                for i, (idx, range_val) in enumerate(zip(tweaked_indices, ranges)):
                    scaled_val = point[i] * (2 * range_val) - range_val
                    full_point[idx] = scaled_val
                uniform_combinations.append(tuple(full_point))

        unique_points = set()
        filtered_uniform_combinations = []
        for point in uniform_combinations:
            if point not in unique_points:
                unique_points.add(point)
                filtered_uniform_combinations.append(point)

        random_combinations = []
        attempt_count = 0
        
        while len(random_combinations) < random_count:
            point = [0] * total_params  # Initialize with zeros
            for idx, range_val in zip(tweaked_indices, ranges):
                point[idx] = random.uniform(-range_val, range_val)
            point_tuple = tuple(point)
            
            if point_tuple not in unique_points:
                random_combinations.append(point_tuple)
                unique_points.add(point_tuple)
            
            attempt_count += 1
            if attempt_count > random_count * 10:  # Prevent infinite loop
                break

        print "Number of uniform points generated: {}".format(len(filtered_uniform_combinations))
        print "Number of random points generated: {}".format(len(random_combinations))
        print "Total attempts: {}".format(attempt_count)

        return filtered_uniform_combinations, random_combinations

def read_input_parameters(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if '>' in lines[i]:
                key = lines[i].strip()
                if i + 1 < len(lines) and '>' not in lines[i + 1]:
                    value = lines[i + 1].strip()
                    parameters[key] = value
    return parameters


def is_perfect_cube(n):
    cube_root = int(round(n ** (1/3.0)))
    return cube_root ** 3 == n

def generate_grid_points(N, zero_range, bg_range, lattice_a_range, *other_lattice_ranges):
    
    num_points_per_axis = int(round(N ** (1 / 3)))  # We ensure the uniqueness for the first 3 dimensions (zero, bg, a)
    
    zero_points = np.linspace(-zero_range, zero_range, num_points_per_axis)
    bg_points = np.linspace(-bg_range, bg_range, num_points_per_axis)
    lattice_a_points = np.linspace(-lattice_a_range, lattice_a_range, num_points_per_axis)
    
    other_points = [np.linspace(-r, r, num_points_per_axis) for r in other_lattice_ranges]
    
    unique_points = list(itertools.product(zero_points, bg_points, lattice_a_points))
    
    other_combinations = list(itertools.product(*other_points))
    
    full_combinations = [point + other for point, other in zip(unique_points, itertools.cycle(other_combinations))]
    
    full_combinations = full_combinations[:N]
    
    return np.array(full_combinations)  # Convert to NumPy array

def generate_sobol_points_generalized(N, zero_range, bg_range, lattice_a_range, *other_lattice_ranges):
    
    points = sobol_seq.i4_sobol_generate(3 + len(other_lattice_ranges), N)  # Total dimensions = 3 + other lattice params
    
    points[:, 0] = points[:, 0] * (2 * zero_range) - zero_range
    points[:, 1] = points[:, 1] * (2 * bg_range) - bg_range
    points[:, 2] = points[:, 2] * (2 * lattice_a_range) - lattice_a_range
    
    for i, r in enumerate(other_lattice_ranges):
        points[:, 3 + i] = points[:, 3 + i] * (2 * r) - r
    
    return np.array(points)  # Convert to NumPy array


def generate_sobol_points(N, zero_range, bg_range, lattice_a_range):
    points = sobol_seq.i4_sobol_generate(3, N)
    
    points[:, 0] = points[:, 0] * (2 * zero_range) - zero_range
    points[:, 1] = points[:, 1] * (2 * bg_range) - bg_range
    points[:, 2] = points[:, 2] * (2 * lattice_a_range) - lattice_a_range
    
    return np.array(points)  # Convert to NumPy array

def main():
    if len(sys.argv) != 2:
        print("Usage: python step2_pcrfolders_modifiedparameters.py <path_to_inputs.txt>")
        sys.exit(1)

    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file_path = sys.argv[1]
    parameters = read_input_parameters(input_file_path)
    # print("Debug: Working directory:", working_dir)
    # print("Debug: Input file path:", input_file_path)

    dataset_status = {}

    input_filename = os.path.basename(input_file_path)
    timestamp = input_filename.replace('inputs_', '').replace('.txt', '')
    
    base_subfolder_name = parameters.get('5>Enter the name of the subfolder to be created in the data directory (the purpose of the subfolder is to organize and store related data files and refinement results for analysis):')
    subfolder_name = base_subfolder_name + "_" + timestamp
    
    print("Using subfolder name: {}".format(subfolder_name))
    
    dat_file_name = parameters.get('2>Name of .DAT file')

    pcr_file_path = os.path.join(working_dir, 'data', subfolder_name, parameters.get('3>Name of .PCR file to be generated'))
    
    atom_format = parameters.get('46>Atom data format in PCR file (enter "4-line" for standard format with thermal parameters, or "2-line" for simplified format without thermal parameters - check your PCR file to see if each atom has 4 lines or 2 lines of data):', '4-line')

    original_values = read_original_values(pcr_file_path, atom_format)    

    num_atoms = len(original_values['atoms']) if 'atoms' in original_values else 1
    print "Number of atoms detected in structure: {}".format(num_atoms)
    
    if 'atoms' in original_values:
        print "Original Biso values for each atom:"
        for atom in original_values['atoms']:
            print "  {} ({}): Biso = {}".format(atom['name'], atom['type'], atom['biso'])

    lattice_type = parameters.get('11>Enter the lattice type (cubic, tetragonal, orthorhombic, hexagonal, monoclinic, triclinic, trigonal):')

    num_datasets = int(parameters.get('13>Input the number of datasets to be generated: (The first half will be uniformly distributed points on the grid defined by the ranges in 7>, 9>, and 12>, and the second half will be randomly distributed points on this grid)'))

    zero_range = float(parameters.get('7>Input the maximum absolute shift range for generating a random number around the zero parameter value (example: 0.03). You may ignore this part if you did not skip part 6>:'))
    bg_range = float(parameters.get('9>Input the maximum absolute shift range for generating a random number around the background parameter value (example: 0.03). You may ignore this part if you did not skip part 8>:'))
    lattice_range = float(parameters.get('12>Input the percentage range for generating a random number around the lattice parameters with degrees of freedom (example: 3%). You may ignore this part if you did not skip part 10>:').replace('%', ''))
    biso_range = float(parameters.get('23>Input the percentage range for generating a random number around the Biso parameter with degrees of freedom (example: 3%):').replace('%', ''))
    scale_range = float(parameters.get('24>Input the percentage range for generating a random number around the scale factor parameter with degrees of freedom (example: 5%):').replace('%', ''))
    u_range = float(parameters.get('25>Input the percentage range for generating a random number around the U, parameter with degrees of freedom (example: 1%):').replace('%', ''))
    v_range = float(parameters.get('26>Input the percentage range for generating a random number around the V, parameter with degrees of freedom (example: 1%):').replace('%', ''))
    w_range = float(parameters.get('27>Input the percentage range for generating a random number around the W, parameter with degrees of freedom (example: 1%):').replace('%', ''))
    
    generate_figures = parameters.get('16>Generate all output figures (Y/N)?', 'Y').strip().upper()
    display_figures = parameters.get('17>Display all output figures (Y/N)?', 'Y').strip().upper()

    if generate_figures == 'N':
        plt.ioff()  # Do not generate or display any figures
    elif display_figures == 'N':
        plt.ioff()  # Do not display figures, but they can still be generated
    else:
        plt.ion()  # Enable interactive mode for generating and displaying figures

    subfolder_path = os.path.join(working_dir, 'data', subfolder_name)
    # print("Debug: subfolder path:", subfolder_path)
    if not os.path.exists(subfolder_path):
        print("Error: The subfolder '{}' does not exist.".format(subfolder_name))
        return

    pcr_file_name = None
    for file_name in os.listdir(subfolder_path):
        if file_name.endswith('.pcr') and not any(suffix in file_name for suffix in ['_zero', '_bg', '_lp']):
            pcr_file_name = file_name
            break

    if not pcr_file_name:
        print("Error: No .pcr file found in the subfolder '{}'.".format(subfolder_name))
        return

    pcr_file_path = os.path.join(subfolder_path, pcr_file_name)

    uniform_combinations, random_combinations = generate_parameter_sets_from_input(
        zero_range=zero_range,
        bg_range=bg_range,
        lattice_range=lattice_range,
        biso_range=biso_range,
        scale_range=scale_range,
        u_range=u_range,
        v_range=v_range,
        w_range=w_range,
        lattice_type=lattice_type,
        uniform_count=num_datasets//2,
        random_count=num_datasets//2,
        num_atoms=num_atoms
    )

    parameter_sets = np.vstack((uniform_combinations, random_combinations))

    print "\nProcessing parameter sets:"
    print "=" * 150
    for idx, param_set in enumerate(parameter_sets):
        basic_params = param_set[:8]  # Zero, BG, lattice parameters
        
        atom_biso_shifts = param_set[8:8+num_atoms]
        
        other_params = param_set[8+num_atoms:]
        
        basic_str = "  ".join(["{:14.10f}".format(val) for val in basic_params])
        
        biso_str = ""
        for a_idx, atom_shift in enumerate(atom_biso_shifts):
            atom_name = original_values['atoms'][a_idx]['name'] if a_idx < len(original_values['atoms']) else "Atom{}".format(a_idx+1)
            biso_str += "{}:{:8.5f}%  ".format(atom_name, atom_shift)
        
        other_str = "  ".join(["{:14.10f}".format(val) for val in other_params])
        
        print "Set {:2d}: [ {} | Biso: {} | {} ]".format(idx, basic_str, biso_str, other_str)
    print "=" * 150

    unique_parameter_sets = set(tuple(point) for point in parameter_sets)
    if len(unique_parameter_sets) == len(parameter_sets):
        print "The combined set is unique."
    else:
        print "The combined set has duplicates."

    if len(unique_parameter_sets) != len(parameter_sets):
        print "Warning: Some parameter sets may have been skipped or duplicated."
        print "Unique parameter sets: {}".format(len(unique_parameter_sets))
        print "Total parameter sets: {}".format(len(parameter_sets))

    print("Number of uniform sets: {}".format(len(uniform_combinations)))
    print("Number of random sets: {}".format(len(random_combinations)))
    print("Total number of random sampling for shifting and scaling of parameters: {}".format(len(parameter_sets)))
    print("Total number of unique parameter sets: {}".format(len(unique_parameter_sets)))

    random_combinations_array = np.array(random_combinations)

    generate_figures = parameters.get('16>Generate all output figures (Y/N)?', 'Y').strip().upper()
    display_figures = parameters.get('17>Display all output figures (Y/N)?', 'Y').strip().upper()

    active_params = []
    if zero_range != 0:
        active_params.append(('Zero Shift', 0))
    if bg_range != 0:
        active_params.append(('Background Shift', 1))
    if lattice_range != 0:
        active_params.append(('Lattice A Scaling', 2))
    if biso_range != 0:
        active_params.append(('Biso Scaling', 8))
    if scale_range != 0:
        active_params.append(('Scale Factor Scaling', 9))
    if u_range != 0:
        active_params.append(('U Parameter', 10))
    if v_range != 0:
        active_params.append(('V Parameter', 11))
    if w_range != 0:
        active_params.append(('W Parameter', 12))

    default_params = [('Zero Shift', 0), ('Background Shift', 1), ('Lattice A Scaling', 2)]
    for param in default_params:
        if len(active_params) < 3 and param not in active_params:
            active_params.append(param)

    active_params = active_params[:3]
    param_names, param_indices = zip(*active_params)

    if generate_figures == 'Y' and parameters.get('41>Generate parameter shift distribution plots (Y/N)?:', 'Y').strip().upper() == 'Y':
        fig = plt.figure(figsize=(10, 8))
        if lattice_type == 'cubic':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[param_indices[0]] for x in uniform_combinations], 
                      [x[param_indices[1]] for x in uniform_combinations], 
                      [x[param_indices[2]] for x in uniform_combinations], 
                      c='blue', label='Uniform Set')
            ax.set_xlabel(param_names[0])
            ax.set_ylabel(param_names[1])
            ax.set_zlabel(param_names[2])
        elif lattice_type in ['tetragonal', 'hexagonal']:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in uniform_combinations], [x[1] for x in uniform_combinations], [x[2] for x in uniform_combinations], c='blue', label='Uniform Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice C Scaling')
        elif lattice_type == 'orthorhombic':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in uniform_combinations], [x[1] for x in uniform_combinations], [x[2] for x in uniform_combinations], c='blue', label='Uniform Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice B Scaling')
            ax.set_zlabel('Lattice C Scaling')
        elif lattice_type == 'monoclinic':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in uniform_combinations], [x[1] for x in uniform_combinations], [x[2] for x in uniform_combinations], c='blue', label='Uniform Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice B Scaling')
            ax.set_zlabel('Lattice C Scaling')
            ax.set_zlabel('Lattice Beta Scaling')
        elif lattice_type == 'triclinic':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in uniform_combinations], [x[1] for x in uniform_combinations], [x[2] for x in uniform_combinations], c='blue', label='Uniform Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice B Scaling')
            ax.set_zlabel('Lattice C Scaling')
            ax.set_zlabel('Lattice Alpha Scaling')
            ax.set_zlabel('Lattice Beta Scaling')
            ax.set_zlabel('Lattice Gamma Scaling')
        elif lattice_type == 'trigonal':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in uniform_combinations], [x[1] for x in uniform_combinations], [x[2] for x in uniform_combinations], c='blue', label='Uniform Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice Alpha Scaling')

        output_figures_folder = os.path.join(subfolder_path, 'output_figures')
        if not os.path.exists(output_figures_folder):
            os.makedirs(output_figures_folder)
        
        parameter_shifts_folder = os.path.join(output_figures_folder, 'parameter_shift_distribution_plots')
        if not os.path.exists(parameter_shifts_folder):
            os.makedirs(parameter_shifts_folder)
            
        plt.legend()
        uniform_plot_path = os.path.join(parameter_shifts_folder, "selectedparametershifts_uniformdistribution.png")
        plt.savefig(uniform_plot_path)

        if display_figures == 'Y':
            plt.show(block=False)
            plt.pause(1)  # Keep the plot open for 1 second
            plt.close()
        else:
            plt.close()
        
        fig = plt.figure(figsize=(10, 8))
        if lattice_type == 'cubic':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[param_indices[0]] for x in random_combinations], 
                    [x[param_indices[1]] for x in random_combinations], 
                    [x[param_indices[2]] for x in random_combinations], 
                    c='red', label='Random Set')
            ax.set_xlabel(param_names[0])
            ax.set_ylabel(param_names[1])
            ax.set_zlabel(param_names[2])
        elif lattice_type in ['tetragonal', 'hexagonal']:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in random_combinations], [x[1] for x in random_combinations], [x[2] for x in random_combinations], c='red', label='Random Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice C Scaling')
        elif lattice_type == 'orthorhombic':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in random_combinations], [x[1] for x in random_combinations], [x[2] for x in random_combinations], c='red', label='Random Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice B Scaling')
            ax.set_zlabel('Lattice C Scaling')
        elif lattice_type == 'monoclinic':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in random_combinations], [x[1] for x in random_combinations], [x[2] for x in random_combinations], c='red', label='Random Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice B Scaling')
            ax.set_zlabel('Lattice C Scaling')
            ax.set_zlabel('Lattice Beta Scaling')
        elif lattice_type == 'triclinic':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in random_combinations], [x[1] for x in random_combinations], [x[2] for x in random_combinations], c='red', label='Random Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice B Scaling')
            ax.set_zlabel('Lattice C Scaling')
            ax.set_zlabel('Lattice Alpha Scaling')
            ax.set_zlabel('Lattice Beta Scaling')
            ax.set_zlabel('Lattice Gamma Scaling')
        elif lattice_type == 'trigonal':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([x[0] for x in random_combinations], [x[1] for x in random_combinations], [x[2] for x in random_combinations], c='red', label='Random Set')
            ax.set_xlabel('Zero Shift')
            ax.set_ylabel('Background Shift')
            ax.set_zlabel('Lattice A Scaling')
            ax.set_zlabel('Lattice Alpha Scaling')

        plt.legend()
        random_plot_path = os.path.join(parameter_shifts_folder, "selectedparametershifts_randomdistribution.png")
        plt.savefig(random_plot_path)

        if display_figures == 'Y':
            plt.show(block=False)
            plt.pause(1)  # Keep the plot open for 1 second
            plt.close()
        else:
            plt.close()

    for idx, param_set in enumerate(parameter_sets):

        zero_shift = param_set[0]
        bg_shift = param_set[1]
        
        if lattice_type == 'cubic':
            lattice_shift = param_set[2]
            a_value = float(original_values['a']) * (1 + lattice_shift / 100.0)
            b_value = a_value
            c_value = a_value
            alpha_value = 90.0
            beta_value = 90.0
            gamma_value = 90.0
        elif lattice_type in ['tetragonal', 'hexagonal']:
            lattice_a_shift = param_set[2]
            lattice_c_shift = param_set[3]
            a_value = float(original_values['a']) * (1 + lattice_a_shift / 100.0)
            b_value = a_value
            c_value = float(original_values['c']) * (1 + lattice_c_shift / 100.0)
            alpha_value = 90.0
            beta_value = 90.0
            gamma_value = 120.0 if lattice_type == 'hexagonal' else 90.0
        elif lattice_type == 'orthorhombic':
            lattice_a_shift = param_set[2]
            lattice_b_shift = param_set[3]
            lattice_c_shift = param_set[4]
            a_value = float(original_values['a']) * (1 + lattice_a_shift / 100.0)
            b_value = float(original_values['b']) * (1 + lattice_b_shift / 100.0)
            c_value = float(original_values['c']) * (1 + lattice_c_shift / 100.0)
            alpha_value = 90.0
            beta_value = 90.0
            gamma_value = 90.0
        elif lattice_type == 'monoclinic':
            lattice_a_shift = param_set[2]
            lattice_b_shift = param_set[3]
            lattice_c_shift = param_set[4]
            lattice_beta_shift = param_set[5]
            a_value = float(original_values['a']) * (1 + lattice_a_shift / 100.0)
            b_value = float(original_values['b']) * (1 + lattice_b_shift / 100.0)
            c_value = float(original_values['c']) * (1 + lattice_c_shift / 100.0)
            alpha_value = 90.0
            beta_value = float(original_values['beta']) * (1 + lattice_beta_shift / 100.0)
            gamma_value = 90.0
        elif lattice_type == 'triclinic':
            lattice_a_shift = param_set[2]
            lattice_b_shift = param_set[3]
            lattice_c_shift = param_set[4]
            lattice_alpha_shift = param_set[5]
            lattice_beta_shift = param_set[6]
            lattice_gamma_shift = param_set[7]
            a_value = float(original_values['a']) * (1 + lattice_a_shift / 100.0)
            b_value = float(original_values['b']) * (1 + lattice_b_shift / 100.0)
            c_value = float(original_values['c']) * (1 + lattice_c_shift / 100.0)
            alpha_value = float(original_values['alpha']) * (1 + lattice_alpha_shift / 100.0)
            beta_value = float(original_values['beta']) * (1 + lattice_beta_shift / 100.0)
            gamma_value = float(original_values['gamma']) * (1 + lattice_gamma_shift / 100.0)
        elif lattice_type == 'trigonal':
            lattice_a_shift = param_set[2]
            lattice_alpha_shift = param_set[3]
            a_value = float(original_values['a']) * (1 + lattice_a_shift / 100.0)
            b_value = a_value
            c_value = a_value
            alpha_value = float(original_values['alpha']) * (1 + lattice_alpha_shift / 100.0)
            beta_value = alpha_value
            gamma_value = alpha_value

        zero_value = float(original_values['zero']) + zero_shift
        background_value = float(original_values['bg']) + bg_shift

        atom_biso_shifts = param_set[8:8+num_atoms].tolist()
        
        next_idx = 8 + num_atoms
        scale_shift = param_set[next_idx]
        next_idx += 1
        u_shift = param_set[next_idx]
        next_idx += 1
        v_shift = param_set[next_idx]
        next_idx += 1
        w_shift = param_set[next_idx]
        
        scale_value = float(original_values['scale']) * (1 + scale_shift / 100.0)
        u_value = float(original_values['U']) * (1 + u_shift / 100.0)
        v_value = float(original_values['V']) * (1 + v_shift / 100.0)
        w_value = float(original_values['W']) * (1 + w_shift / 100.0)

        zero_shift_str = "{:.10f}".format(zero_shift).replace('.', 'p').replace('-', 'n')
        bg_shift_str = "{:.10f}".format(bg_shift).replace('.', 'p').replace('-', 'n')

        shift_strs = [zero_shift_str, bg_shift_str]
        if lattice_type == 'cubic':
            shift_strs.append("{:.10f}".format(lattice_shift).replace('.', 'p').replace('-', 'n'))
        elif lattice_type == 'tetragonal':
            shift_strs.append("{:.10f}".format(lattice_a_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_c_shift).replace('.', 'p').replace('-', 'n'))
        elif lattice_type == 'orthorhombic':
            shift_strs.append("{:.10f}".format(lattice_a_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_b_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_c_shift).replace('.', 'p').replace('-', 'n'))
        elif lattice_type == 'hexagonal':
            shift_strs.append("{:.10f}".format(lattice_a_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_c_shift).replace('.', 'p').replace('-', 'n'))
        elif lattice_type == 'monoclinic':
            shift_strs.append("{:.10f}".format(lattice_a_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_b_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_c_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_beta_shift).replace('.', 'p').replace('-', 'n'))
        elif lattice_type == 'triclinic':
            shift_strs.append("{:.10f}".format(lattice_a_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_b_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_c_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_alpha_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_beta_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_gamma_shift).replace('.', 'p').replace('-', 'n'))
        elif lattice_type == 'trigonal':
            shift_strs.append("{:.10f}".format(lattice_a_shift).replace('.', 'p').replace('-', 'n'))
            shift_strs.append("{:.10f}".format(lattice_alpha_shift).replace('.', 'p').replace('-', 'n'))

        biso_shift_strs = []
        for atom_idx, atom_shift in enumerate(atom_biso_shifts):
            atom_shift_str = "{:.10f}".format(atom_shift).replace('.', 'p').replace('-', 'n')
            biso_shift_strs.append(atom_shift_str)
        
        shift_strs.extend(biso_shift_strs)  # Add atom-specific Biso shifts
        
        scale_shift_str = "{:.10f}".format(scale_shift).replace('.', 'p').replace('-', 'n')
        u_shift_str = "{:.10f}".format(u_shift).replace('.', 'p').replace('-', 'n')
        v_shift_str = "{:.10f}".format(v_shift).replace('.', 'p').replace('-', 'n')
        w_shift_str = "{:.10f}".format(w_shift).replace('.', 'p').replace('-', 'n')
        shift_strs.extend([scale_shift_str, u_shift_str, v_shift_str, w_shift_str])

        combined_shift_str = '_'.join(shift_strs)

        modified_pcr_path = modify_combined_parameters(
            pcr_file_path, 
            zero_value, background_value, 
            a_value, b_value, c_value, 
            alpha_value, beta_value, gamma_value,
            atom_biso_shifts,  # Now passing list of atom-specific shifts 
            scale_value, u_value, v_value, w_value,
            combined_shift_str, shift_strs, lattice_type,
            bg_range,  # Pass the bg_range parameter
            atom_format  # Pass the atom_format parameter
        )

        sample_pcr_filename = "sample{}.pcr".format(idx + 1)

        new_pcr_file_path = os.path.join(subfolder_path, sample_pcr_filename)
        
        try:
            if os.path.exists(modified_pcr_path):
                shutil.copy2(modified_pcr_path, new_pcr_file_path)
                os.remove(modified_pcr_path)  # Remove the original to clean up
                print "Created and renamed PCR file to {}".format(sample_pcr_filename)
            else:
                print "Warning: Modified PCR file not found: {}".format(modified_pcr_path)
                continue
        except Exception as e:
            print "Error handling PCR file {}: {}".format(modified_pcr_path, e)
            continue

        append_spaces_to_lattice_line(new_pcr_file_path)

        create_folders_and_move_pcr_files(subfolder_path, [sample_pcr_filename], dataset_status)
        copy_paste_dat_file(subfolder_path, sample_pcr_filename, dat_file_name, dataset_status)

    for attempt in range(3):
        missing_sets = [dataset for dataset, status in dataset_status.items() if status == 'missing']
        if not missing_sets:
            break

        print "Retrying missing datasets (Attempt {}):".format(attempt + 1)
        for param_set in missing_sets:
            create_folders_and_move_pcr_files(subfolder_path, [param_set], dataset_status)

    print "Dataset Status Summary:"
    total_generated = sum(1 for status in dataset_status.values() if status == 'generated')
    total_moved = sum(1 for status in dataset_status.values() if status == 'moved')
    total_missing = sum(1 for status in dataset_status.values() if status == 'missing')

    print "Total datasets generated: {}".format(total_generated)
    print "Total datasets moved: {}".format(total_moved)
    print "Total missing datasets: {}".format(total_missing)

    pcr_folders = []
    for folder in os.listdir(subfolder_path):
        folder_path = os.path.join(subfolder_path, folder)
        if os.path.isdir(folder_path) and folder not in ['output_figures', 'source_files']:
            if any(f.endswith('.pcr') for f in os.listdir(folder_path)):
                pcr_folders.append(folder)
    
    def natural_sort_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    pcr_folders.sort(key=natural_sort_key)
    
    sample_number = 1  # Initialize counter
    for folder in pcr_folders:
        old_folder_path = os.path.join(subfolder_path, folder)
        new_folder_name = "sample{}".format(sample_number)
        new_folder_path = os.path.join(subfolder_path, new_folder_name)
        
        try:
            if os.path.normcase(old_folder_path) != os.path.normcase(new_folder_path):
                os.rename(old_folder_path, new_folder_path)
                # print "Renamed folder '{}' to '{}'".format(folder, new_folder_name)
            else:
                # print "Folder already has the correct name: {}".format(new_folder_name)
                pass
        except Exception as e:
            print "Error renaming folder {}: {}".format(folder, e)
            continue  # Skip file renaming if folder rename failed
        
        pcr_files = [f for f in os.listdir(new_folder_path) if f.endswith('.pcr')]
        for file_name in pcr_files:
            old_pcr_path = os.path.join(new_folder_path, file_name)
            new_pcr_name = "sample{}.pcr".format(sample_number)
            new_pcr_path = os.path.join(new_folder_path, new_pcr_name)
            try:
                if os.path.normcase(old_pcr_path) != os.path.normcase(new_pcr_path):
                    os.rename(old_pcr_path, new_pcr_path)
                    # print "Renamed PCR file '{}' to '{}'".format(file_name, new_pcr_name)
                else:
                    # print "PCR file already has the correct name: {}".format(new_pcr_name)
                    pass
            except Exception as e:
                print "Error renaming PCR file {}: {}".format(file_name, e)
                pass
        
        sample_number += 1 

if __name__ == "__main__":
    main()
