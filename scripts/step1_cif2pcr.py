# -*- coding: utf-8 -*-
import re
import os
import sys
import numpy as np
import pandas as pd
import shutil
import time
import matplotlib.pyplot as plt
from datetime import datetime
import importlib

def generate_pcr_file(file_path):
    content = """COMM   O3 Y2
! Current global Chi2 (Bragg contrib.) =      847.4    
NPATT      1       1 <- Flags for patterns (1:refined, 0: excluded)
W_PAT   1.000
!Nph Dum Ias Nre Cry Opt Aut
   1   0   0   0   0   0   1
!Job Npr Nba Nex Nsc Nor Iwg Ilo Res Ste Uni Cor Anm Int
   0   7   0   2   0   1   0   0   0   0   0   0   0   0  !-> Patt#: 1
!
!File names of data(patterns) files
Y2O3.dat
!
!Mat Pcr NLI Rpa Sym Sho
   0   1   0   0   0   0
!Ipr Ppl Ioc Ls1 Ls2 Ls3 Prf Ins Hkl Fou Ana
   0   0   1   0   4   0   3  0   0   0   0  !-> Patt#: 1
!
! Lambda1  Lambda2    Ratio    Bkpos    Wdt    Cthm     muR   AsyLim   Rpolarz  2nd-muR -> Patt# 1
 1.540560 1.540560  1.00000   25.000  15.000  0.9100  0.0000   60.00    0.0000  0.0000
!
!NCY  Eps  R_at  R_an  R_pr  R_gl
 1  0.10  1.00  1.00  1.00  1.00
!     Thmin       Step       Thmax    PSD    Sent0  -> Patt#: 1
     3.1200   0.020004   150.0000   0.000   0.000
!
! Excluded regions (LowT  HighT) for Pattern#  1
        1.00       20.00
      100.00      160.00
! 
!
       1    !Number of refined parameters
!
!  Zero    Code    SyCos    Code   SySin    Code  Lambda     Code MORE ->Patt# 1
  -0.00032    0.0  0.00000    0.0  0.00000    0.0 0.000000    0.00   0
!   Background coefficients/codes  for Pattern#  1  (Polynomial of 6th degree)
      19.181     -19.888      10.065      -2.284       0.213       0.000
        0.00        0.00        0.00        0.00        0.00        0.00
!-------------------------------------------------------------------------------
!  Data for PHASE number:   1  ==> Current R_Bragg for Pattern#  1:    36.86
!-------------------------------------------------------------------------------
O3 Y2
!
!Nat Dis Ang Jbt Isy Str Furth        ATZ     Nvk More
   3   0   0   0   0   0   0       3613.0464   0   0
!Contributions (0/1) of this phase to the  1 patterns
 1
!Irf Npr Jtyp  Nsp_Ref Ph_Shift for Pattern#  1
   0   7    0      0      0
! Pr1    Pr2    Pr3   Brind.   Rmua   Rmub   Rmuc     for Pattern#  1
  0.000  0.000  1.000  1.000  1.000  1.000  1.000
!
I a -3                   <--Space group symbol
!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes
!    beta11   beta22   beta33   beta12   beta13   beta23  /Codes
Y1     Y       0.25000  0.25000  0.25000  0.00000   0.16667   0   0   2    0  
                  0.00     0.00     0.00     0.00      0.00
      0.00000  0.00000  0.00000  0.00000  0.00000   0.00000
         0.00     0.00     0.00     0.00     0.00      0.00
Y2     Y       0.96730  0.00000  0.25000  0.00000   0.50000   0   0   2    0  
                  0.00     0.00     0.00     0.00      0.00
      0.00000  0.00000  0.00000  0.00000  0.00000   0.00000
         0.00     0.00     0.00     0.00     0.00      0.00
O1     O       0.39150  0.15380  0.38000  0.00000   1.00000   0   0   2    0  
                  0.00     0.00     0.00     0.00      0.00
      0.00000  0.00000  0.00000  0.00000  0.00000   0.00000
         0.00     0.00     0.00     0.00     0.00      0.00
!-------> Profile Parameters for Pattern #  1
!  Scale        Shape1      Bov      Str1      Str2      Str3   Strain-Model
 0.6684669E-03   0.000   0.00000   0.00000   0.00000   0.00000       0
    0.00000     0.000     0.000     0.000     0.000     0.000
!       U         V          W           X          Y        GauSiz   LorSiz Size-Model
     0.004995    -0.006075     0.004575     0.076699     0.000000     0.000000     0.000000       0
      0.000      0.000      0.000      0.000      0.000      0.000      0.000
!     a          b         c        alpha      beta       gamma      #Cell Info
  10.607961  10.607961  10.607961  90.000000  90.000000  90.000000   
    0.00000    0.00000    0.00000    0.00000    0.00000    0.00000
!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4  
  1.00000  0.00000  0.00000  0.00000  0.00000  0.00000
     0.00     0.00     0.00     0.00     0.00     0.00
!  2Th1/TOF1    2Th2/TOF2  Pattern to plot
      20.000     100.000       1
"""
    with open(file_path, 'w') as file:
        file.write(content)

def get_lattice_parameters(cif_file):
    lattice_params = []
    space_group = ""
    with open(cif_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('_cell_length_a') or \
               line.startswith('_cell_length_b') or \
               line.startswith('_cell_length_c') or \
               line.startswith('_cell_angle_alpha') or \
               line.startswith('_cell_angle_beta') or \
               line.startswith('_cell_angle_gamma'):
                number = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
                number = re.sub(r'\(.*?\)', '', number)  # Remove brackets and content inside
                lattice_params.append(number)
            elif line.startswith('_space_group_name_H-M_alt'):
                space_group = re.findall(r"'(.*?)'", line)[0]
    return lattice_params, space_group

def format_lattice_line(original_line, lattice_params):
    parts = re.split(r"(\s+)", original_line)
    formatted_line = ''
    param_index = 0
    for part in parts:
        if re.match(r"[-+]?\d*\.\d+|\d+", part):
            # Preserve original string representation instead of formatting
            formatted_line += lattice_params[param_index]
            param_index += 1
        else:
            formatted_line += part
    return formatted_line
    
def replace_space_group_line(original_line, space_group):
    new_line = original_line.replace(original_line.strip(), space_group + ' ' * 19 + '<--Space group symbol')
    return new_line

def replace_parameters(pcr_file, lattice_params, space_group):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    new_lattice_line = format_lattice_line(lines[74], lattice_params)
    new_space_group_line = replace_space_group_line(lines[51], space_group)

    lines[51] = new_space_group_line + '\n'
    lines[74] = new_lattice_line

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def get_atom_positions(cif_file):
    atom_positions = []
    with open(cif_file, 'r') as file:
        lines = file.readlines()
        atom_section = False
        for line in lines:
            if '_atom_site_type_symbol' in line:
                atom_section = True
                continue
            if atom_section:
                if line.strip() == '':
                    break
                parts = line.split()
                if len(parts) >= 5:
                    x = re.sub(r'\(.*?\)', '', parts[2])  # Remove brackets and content inside
                    y = re.sub(r'\(.*?\)', '', parts[3])  # Remove brackets and content inside
                    z = re.sub(r'\(.*?\)', '', parts[4])  # Remove brackets and content inside
                    atom_positions.append((parts[0], x, y, z))
    return atom_positions

def get_biso_values(cif_file):
    biso_values = {}
    with open(cif_file, 'r') as file:
        lines = file.readlines()
        atom_section = False
        column_indices = {}  # To store indices of different columns
        
        for i, line in enumerate(lines):
            if '_atom_site_label' in line:
                header_start = i
                atom_section = True
                
                for j, header_line in enumerate(lines[i:i+10]):  # Check next few lines for headers
                    if '_atom_site_label' in header_line:
                        column_indices['label'] = j
                    elif '_atom_site_U_iso_or_equiv' in header_line:
                        column_indices['uiso'] = j
                    elif '_atom_site_type_symbol' in header_line:
                        column_indices['type'] = j
                        break
                
                continue
            
            if atom_section and line.strip() and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= max(column_indices.values()) + 1:
                    atom_label = parts[0]
                    for j, part in enumerate(parts):
                        if (part == 'Uiso' or part == 'Uani') and j+1 < len(parts):
                            try:
                                u_value = float(parts[j+1])
                                biso = u_value * 8 * (np.pi**2)
                                biso_values[atom_label] = round(biso, 5)
                                print("Calculated Biso for %s: %.5f from %s value: %.5f" % 
                                      (atom_label, biso, part, u_value))
                            except ValueError:
                                print("Could not convert U value to float for atom %s" % atom_label)
                            break
                elif line.strip() == '':
                    atom_section = False
                    break
    
    if not biso_values:
        print("Warning: No thermal parameters (Uiso/Uani) found in CIF file. Using default value of 0.7")
        for atom, _, _, _ in get_atom_positions(cif_file):
            biso_values[atom] = 0.7
    
    return biso_values

def replace_atom_positions(pcr_file, atom_positions, biso_values):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    atom_line_index = -1
    for i, line in enumerate(lines):
        if '!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes' in line:
            atom_line_index = i + 2
            break

    if atom_line_index == -1:
        raise ValueError("Atom position line not found in PCR file.")

    phase_line_index = -1
    for i, line in enumerate(lines):
        if '!  Data for PHASE number:' in line:
            phase_line_index = i + 2
            break
    
    if phase_line_index != -1 and phase_line_index < len(lines):
        chemical_name = os.path.basename(pcr_file).split('.')[0]
        lines[phase_line_index] = chemical_name + '\n'

    for i, atom in enumerate(atom_positions):
        current_line_index = atom_line_index + i * 4
        if current_line_index >= len(lines) or '!-------> Profile Parameters for Pattern #  1' in lines[current_line_index]:
            biso_value = biso_values.get(atom[0], 0.00000)  # Get Biso value or default to 0
            atom_line = "{}     {}       {}  {}  {}  {}   1.00000   0   0   2    0  \n".format(
                atom[0], ''.join(filter(str.isalpha, atom[0])), 
                atom[1], atom[2], atom[3], str(biso_value))
            lines.insert(current_line_index, atom_line)
            lines.insert(current_line_index + 1, "                  0.00     0.00     0.00     0.00      0.00\n")
            lines.insert(current_line_index + 2, "      0.00000  0.00000  0.00000  0.00000  0.00000   0.00000\n")
            lines.insert(current_line_index + 3, "         0.00     0.00     0.00     0.00     0.00      0.00\n")
        else:
            biso_value = biso_values.get(atom[0], 0.00000) 
            parts = re.split(r"(\s+)", lines[current_line_index])
            parts[0] = atom[0]
            parts[2] = ''.join(filter(str.isalpha, atom[0]))
            parts[4] = "{:.5f}".format(float(atom[1]))
            parts[6] = "{:.5f}".format(float(atom[2]))
            parts[8] = "{:.5f}".format(float(atom[3]))
            parts[10] = "{:.5f}".format(biso_value)
            lines[current_line_index] = ''.join(parts)

    u_v_w_x_y_index = -1
    for i, line in enumerate(lines):
        if '!       U         V          W           X          Y        GauSiz   LorSiz Size-Model' in line:
            u_v_w_x_y_index = i
            break

    if u_v_w_x_y_index != -1 and u_v_w_x_y_index + 2 < len(lines):
        target_line_index = u_v_w_x_y_index + 2
        target_line = lines[target_line_index].rstrip()  # Remove any trailing whitespace
        lines[target_line_index] = target_line + '      0.000\n'

    with open(pcr_file, 'w') as file:
        file.writelines(lines)


def remove_blank_line_before_atom_line(pcr_file):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()
    
    atom_line_index = -1
    for i, line in enumerate(lines):
        if '!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes' in line:
            atom_line_index = i
            break

    if (atom_line_index > 0) and (lines[atom_line_index - 1].strip() == ''):
        del lines[atom_line_index - 1]

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def get_space_group_operations(cif_file):
    operations = []
    with open(cif_file, 'r') as file:
        lines = file.readlines()
        operation_section = False
        for line in lines:
            if '_space_group_symop_operation_xyz' in line:
                operation_section = True
                continue
            if operation_section:
                if line.strip() == '':
                    break
                operations.append(line.strip().strip("'"))
    return operations

def apply_space_group_operations(atom_positions, operations):
    transformed_positions = {}
    
    for atom in atom_positions:
        atom_name, x, y, z = atom
        x, y, z = float(x), float(y), float(z)
        positions = []

        for operation in operations:
            operation = operation.replace('1/2', '0.5').replace('1/4', '0.25').replace('3/4', '0.75')
            operation = operation.replace('x', '({})'.format(x)).replace('y', '({})'.format(y)).replace('z', '({})'.format(z))
            
            terms = operation.split(',')
            x_new = eval(terms[0])
            y_new = eval(terms[1])
            z_new = eval(terms[2])
            
            positions.append((x_new, y_new, z_new))

        transformed_positions[atom_name] = positions
    return transformed_positions


from collections import OrderedDict

def calculate_independent_positions(transformed_positions):
    independent_positions = OrderedDict()
    for atom in transformed_positions:
        positions = transformed_positions[atom]
        positions_mod = [(np.mod(x, 1), np.mod(y, 1), np.mod(z, 1)) for x, y, z in positions]
        unique_positions = list(set(positions_mod))
        independent_positions[atom] = unique_positions
    return independent_positions

def calculate_occupancies(independent_positions, space_group_operations):
    total_operations = len(space_group_operations)
    occupancies = OrderedDict()
    for atom in independent_positions:
        positions = independent_positions[atom]
        print("Atom: %s, Independent positions: %d, Total operations: %d" % (atom, len(positions), total_operations))
        occupancies[atom] = len(positions) / float(total_operations)
    return occupancies

def replace_occupancies(pcr_file, atom_positions, occupancies):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    atom_line_index = -1
    for i, line in enumerate(lines):
        if '!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes' in line:
            atom_line_index = i + 2
            break

    if atom_line_index == -1:
        raise ValueError("Atom position line not found in PCR file.")

    for i, atom in enumerate(atom_positions):
        atom_name = atom[0]
        current_line_index = atom_line_index + i * 4
        if current_line_index >= len(lines) or '!-------> Profile Parameters for Pattern #  1' in lines[current_line_index]:
            break
        parts = re.split(r"(\s+)", lines[current_line_index])
        parts[12] = "{:.5f}".format(occupancies[atom_name])
        lines[current_line_index] = ''.join(parts)

    with open(pcr_file, 'w') as file:
        file.writelines(lines)
        
def get_chemical_name(cif_file):
    chemical_name = ""
    with open(cif_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('_chemical_name_common'):
                chemical_name = line.split()[1].strip().strip("'")
                if not chemical_name:
                    chemical_name = os.path.basename(cif_file).split('_vesta.cif')[0]
                return chemical_name
    return os.path.basename(cif_file).split('_vesta.cif')[0]

def replace_first_line_pcr(pcr_file, common_name):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    lines[0] = "COMM   {}.pcr\n".format(common_name)

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def update_atom_count(pcr_file, atom_positions):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    target_line_index = -1
    for i, line in enumerate(lines):
        if '!Nat Dis Ang Jbt Isy Str Furth        ATZ     Nvk More' in line:
            target_line_index = i + 1
            break

    if target_line_index != -1:
        parts = re.split(r"(\s+)", lines[target_line_index])
        for j, part in enumerate(parts):
            if part.strip().isdigit():
                parts[j] = "{}".format(len(atom_positions))
                break
        lines[target_line_index] = ''.join(parts)

        if target_line_index + 1 < len(lines) and lines[target_line_index + 1].strip() == '':
            del lines[target_line_index + 1]

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def update_data_file_name(pcr_file, pcr_file_name):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    target_line_index = -1
    for i, line in enumerate(lines):
        if '!File names of data(patterns) files' in line:
            target_line_index = i + 1
            break

    if target_line_index != -1:
        dat_file_name = os.path.splitext(pcr_file_name)[0] + '.dat'
        lines[target_line_index] = dat_file_name + '\n'

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def align_atomic_positions_second_column(pcr_file):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    atomic_lines_indices = []
    for i, line in enumerate(lines):
        if re.match(r'^[A-Za-z]+\s+[A-Za-z]+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+', line):
            atomic_lines_indices.append(i)

    for line_index in atomic_lines_indices:
        reference_line_index = line_index + 3
        if reference_line_index >= len(lines):
            continue

        atomic_line = lines[line_index].rstrip()
        reference_line = lines[reference_line_index].rstrip()

        atomic_parts = re.split(r"(\s+)", atomic_line)
        reference_parts = re.split(r"(\s+)", reference_line)

        atom_type = atomic_parts[2].strip()
        try:
            ref_pos_index = reference_parts[0].index('0') - 2
        except ValueError:
            ref_pos_index = 0

        if len(atom_type) > 1:
            new_pos = ref_pos_index
            atomic_parts[2] = ' ' * new_pos + atom_type
        else:
            new_pos = ref_pos_index + 1
            atomic_parts[2] = ' ' * new_pos + atom_type

        lines[line_index] = ''.join(atomic_parts) + '\n'

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def align_lattice_parameters(pcr_file):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    lattice_line_index = -1
    for i, line in enumerate(lines):
        if '!     a          b         c        alpha      beta       gamma      #Cell Info' in line:
            lattice_line_index = i + 1
            break

    if lattice_line_index == -1 or lattice_line_index + 1 >= len(lines):
        print("Lattice parameters line or the line below it not found.")
        return

    lattice_params_line = lines[lattice_line_index].strip()
    reference_line = lines[lattice_line_index + 1].strip()

    lattice_params = [part for part in re.split(r"(\s+)", lattice_params_line) if part.strip() != ""]
    reference_params = [part for part in re.split(r"(\s+)", reference_line) if part.strip() != ""]

    df = pd.DataFrame([reference_params, lattice_params]).T

    for i in range(len(lattice_params)):
        df.iloc[i, 1] = "{:10.6f}".format(float(lattice_params[i]))

    aligned_lattice_params_line = ' ' + ' '.join(df[1].tolist())
    lines[lattice_line_index] = aligned_lattice_params_line + '\n'

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def align_atomic_positions_columns(pcr_file):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    atomic_lines_indices = []
    for i, line in enumerate(lines):
        if re.match(r'^[A-Za-z]+\s+[A-Za-z]+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+', line):
            atomic_lines_indices.append(i)

    for line_index in atomic_lines_indices:
        reference_line_index = line_index + 1
        if reference_line_index >= len(lines):
            continue

        atomic_parts = re.split(r"(\s+)", lines[line_index].rstrip())
        reference_parts = re.split(r"(\s+)", lines[reference_line_index].rstrip())

        for col_index in range(6, 15, 4):
            if len(reference_parts) > col_index and re.match(r"[-+]?\d*\.\d+|\d+", reference_parts[col_index]):
                ref_value = reference_parts[col_index].strip()
                ref_last_digit_pos = ref_value.rfind('0')
                atomic_value = atomic_parts[col_index].strip()

                if len(atomic_value) > 0:
                    atomic_value = "{:.5f}".format(float(atomic_value))
                    atomic_last_digit_pos = atomic_value.rfind('0')
                    shift = ref_last_digit_pos - atomic_last_digit_pos
                    atomic_parts[col_index] = ' ' * shift + atomic_value

        lines[line_index] = ''.join(atomic_parts) + '\n'

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def check_second_column_for_multiple_letters(pcr_file):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    atomic_lines_indices = []
    for i, line in enumerate(lines):
        if re.match(r'^[A-Za-z]+\s+[A-Za-z]+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+', line):
            atomic_lines_indices.append(i)

    lines_with_multiple_letters = []
    for line_index in atomic_lines_indices:
        atomic_parts = re.split(r"(\s+)", lines[line_index].rstrip())
        atom_type = atomic_parts[2].strip()
        if len(atom_type) > 1:
            lines_with_multiple_letters.append(line_index + 1)  # +1 to match line number in the file
            atomic_parts[3] = atomic_parts[3][1:]  # Remove one space from the next part
            lines[line_index] = ''.join(atomic_parts) + '\n'

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

    return lines_with_multiple_letters, lines

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

def append_spaces_to_first_two_atomic_lines(pcr_file):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    atomic_lines_indices = []
    for i, line in enumerate(lines):
        if re.match(r'^[A-Za-z]+\s+[A-Za-z]+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+', line):
            atomic_lines_indices.append(i)
            if len(atomic_lines_indices) == 2:
                break

    for index in atomic_lines_indices:
        lines[index] = lines[index].rstrip() + '  \n'

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def replace_ins_value(pcr_file, new_ins_value):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    header_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('!Ipr Ppl Ioc Ls1 Ls2 Ls3 Prf Ins Hkl Fou Ana'):
            header_index = i
            break

    if header_index != -1 and header_index + 1 < len(lines):
        target_line_index = header_index + 1
        parts = re.split(r"(\s+)", lines[target_line_index])
        numeric_indices = [index for index, part in enumerate(parts) if part.strip().isdigit()]

        if len(numeric_indices) >= 8:
            parts[numeric_indices[7]] = "{}".format(new_ins_value)
        lines[target_line_index] = ''.join(parts)

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def fix_atom_labels_spacing(pcr_file):

    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    atom_line_indices = []
    for i, line in enumerate(lines):
        # Look for lines with atom name, type, and then floating point coordinates
        if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+', line):
            atom_line_indices.append(i)
    
    for i in atom_line_indices:
        try:
            parts = lines[i].split()
            if len(parts) >= 7:  # Must have at least name, type, x, y, z, Biso, occ
                try:
                    name = parts[0]
                    typ = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    biso = float(parts[5])
                    occ = float(parts[6])
                    
                    lines[i] = "{:<2} {:<2}      {:.5f}  {:.5f}  {:.5f}  {:.5f}   {:.5f}   0   0   2    0  \n".format(
                        name, typ, x, y, z, biso, occ)
                except ValueError as e:
                    print("Warning: Unable to parse atom values in line {}".format(i+1))
        except Exception as e:
            print("Error processing atom line {}: {}".format(i+1, str(e)))
    
    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def move_pcr_file_to_data_subfolder(pcr_file_path):
    pcr_file_name = os.path.basename(pcr_file_path)
    subfolder_name = os.path.splitext(pcr_file_name)[0]
    target_directory = os.path.join('data', subfolder_name)

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    target_path = os.path.join(target_directory, pcr_file_name)
    os.rename(pcr_file_path, target_path)

    print("PCR file created: {}".format(pcr_file_name))
    print("Moved to directory: {}".format(target_directory))

def delete_atom_rows(pcr_file, nat):
    if nat < 3:
        rows_to_delete = (3 - nat) * 4  # Calculate the number of rows to delete
        with open(pcr_file, 'r') as file:
            lines = file.readlines()

        profile_params_index = -1
        for i, line in enumerate(lines):
            if '!-------> Profile Parameters for Pattern #  1' in line:
                profile_params_index = i
                break

        if profile_params_index != -1 and profile_params_index - rows_to_delete - 1 >= 0:
            del lines[profile_params_index - rows_to_delete - 1:profile_params_index - 1]

        with open(pcr_file, 'w') as file:
            file.writelines(lines)

def align_columns_atom_lines_additional(pcr_file):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if re.match(r'^[A-Za-z]+\d*\s+[A-Za-z]+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+', line):
            parts = re.split(r"(\s+)", line)
            first_column_value = parts[0].strip()
            if len(first_column_value) == 3:
                parts[1] = parts[1][1:] if len(parts[1]) > 1 else parts[1]
                parts[3] = parts[3][1:] if len(parts[3]) > 1 else parts[3]

                lines[i] = ''.join(parts)

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

def align_columns_negative_sign(pcr_file):
    with open(pcr_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if re.match(r'^[A-Za-z]+\d*\s+[A-Za-z]+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+', line):
            parts = re.split(r"(\s+)", line)

            for col_index in [4, 6, 8, 12]:
                if len(parts) > col_index and '-' in parts[col_index]:
                    parts[col_index - 1] = parts[col_index - 1][:-1]

            lines[i] = ''.join(parts)

    with open(pcr_file, 'w') as file:
        file.writelines(lines)

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

def apply_reference_pcr_format(pcr_file_path, ref_pcr_name, root_dir, copy_asymmetry=False, copy_scan_range=False):

    try:
        if not ref_pcr_name or ref_pcr_name.lower() == 'default':
            print("Using default Si format - no reference PCR applied.")
            return True
            
        ref_pcr_base = os.path.splitext(ref_pcr_name)[0]
        
        module_name = "reference_pcr_{}".format(ref_pcr_base)
        
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        
        try:
            ref_module = importlib.import_module(module_name)
            print("Successfully imported reference module: {}".format(module_name))
        except ImportError:
            print("Warning: Reference module '{}' not found. Using default format.".format(module_name))
            return True
        
        if hasattr(ref_module, 'apply_reference_params_extended'):
            result = ref_module.apply_reference_params_extended(pcr_file_path, root_dir, copy_asymmetry, copy_scan_range)
        else:
            result = ref_module.apply_reference_params(pcr_file_path, root_dir)
        
        return result
        
    except Exception as e:
        print("Error applying reference PCR format: {}".format(str(e)))
        print("Using default Si format instead.")
        return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python step1_cif2pcr.py <path_to_inputs.txt>")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    if not os.path.exists(input_file_path):
        print("Error: The file '{}' does not exist.".format(input_file_path))
        sys.exit(1)

    parameters = read_input_parameters(input_file_path)

    cif_file = parameters.get('1>Name of .CIF file')
    dat_file = parameters.get('2>Name of .DAT file')
    pcr_file_name = parameters.get('3>Name of .PCR file to be generated')
    new_ins_value = int(parameters.get('4>Value of INS'))
    subfolder_name = parameters.get('5>Enter the name of the subfolder to be created in the data directory (the purpose of the subfolder is to organize and store related data files and refinement results for analysis):')

    reference_pcr = parameters.get('34>Reference PCR file to extract parameters from (leave blank or enter "default" to use default):', '').strip()

    copy_asymmetry = parameters.get('38>Copy asymmetry-shape parameters and preferred orientation parameters from reference PCR (Y/N)?:', 'N').strip().upper() == 'Y'
    copy_scan_range = parameters.get('39>Copy scan-range and pattern-selection parameters from reference PCR (Y/N)?:', 'N').strip().upper() == 'Y'

    generate_figures = parameters.get('16>Generate all output figures (Y/N)?', 'Y').strip().upper()
    display_figures = parameters.get('17>Display all output figures (Y/N)?', 'Y').strip().upper()

    if generate_figures == 'N':
        plt.ioff()  # Disable interactive mode to prevent figure display
    elif display_figures == 'N':
        plt.ioff()  # Do not display figures, but they can still be generated
    else:
        plt.ion()  # Ensure interactive mode is on if both are 'Y'

    working_dir = os.path.dirname(os.path.abspath(input_file_path))

    root_dir = os.path.dirname(os.path.dirname(working_dir))

    subfolder_path = working_dir

    pcr_file_path = os.path.join(working_dir, pcr_file_name)

    generate_pcr_file(pcr_file_path)

    if reference_pcr and reference_pcr.lower() != 'default':
        print("Applying reference PCR format from: {}".format(reference_pcr))
        if apply_reference_pcr_format(pcr_file_path, reference_pcr, root_dir, copy_asymmetry, copy_scan_range):
            print("Successfully applied reference PCR format.")
        else:
            print("Warning: Could not fully apply reference PCR format. Using default format.")
    else:
        print("Using default Si format - no reference PCR applied.")

    if generate_figures == 'N':
        plt.ioff()  # Do not generate or display any figures
    elif display_figures == 'N':
        plt.ioff()  # Do not display figures, but they can still be generated
    else:
        plt.ion()  # Ensure interactive mode is on if both are 'Y'

    current_directory = os.getcwd()
    print("Current directory: {}".format(current_directory))
    print("Available files:", os.listdir(current_directory))

    cif_file_path = os.path.join(root_dir, 'dat_vestacif_files', cif_file)
    if not os.path.isfile(cif_file_path):
        print("Error: The file '{}' does not exist.".format(cif_file_path))
        sys.exit(1)


    lattice_params, space_group = get_lattice_parameters(cif_file_path)
    replace_parameters(pcr_file_path, lattice_params, space_group)

    atom_positions = get_atom_positions(cif_file_path)
    biso_values = get_biso_values(cif_file_path)
    replace_atom_positions(pcr_file_path, atom_positions, biso_values)

    remove_blank_line_before_atom_line(pcr_file_path)

    space_group_operations = get_space_group_operations(cif_file_path)
    transformed_positions = apply_space_group_operations(atom_positions, space_group_operations)
    independent_positions = calculate_independent_positions(transformed_positions)
    occupancies = calculate_occupancies(independent_positions, space_group_operations)

    replace_occupancies(pcr_file_path, atom_positions, occupancies)

    chemical_name = get_chemical_name(cif_file_path)
    replace_first_line_pcr(pcr_file_path, chemical_name)

    update_atom_count(pcr_file_path, atom_positions)
    update_data_file_name(pcr_file_path, pcr_file_name)

    align_lattice_parameters(pcr_file_path)
    align_atomic_positions_second_column(pcr_file_path)
    align_atomic_positions_columns(pcr_file_path)
    append_spaces_to_lattice_line(pcr_file_path)
    check_second_column_for_multiple_letters(pcr_file_path)
    append_spaces_to_first_two_atomic_lines(pcr_file_path)

    fix_atom_labels_spacing(pcr_file_path)

    delete_atom_rows(pcr_file_path, len(atom_positions))

    align_columns_atom_lines_additional(pcr_file_path)

    align_columns_negative_sign(pcr_file_path)

    # Replace new Ins value
    replace_ins_value(pcr_file_path, new_ins_value)

    # shutil.move(pcr_file_path, os.path.join(subfolder_path, pcr_file_name))

    # shutil.copy(cif_file_path, os.path.join(subfolder_path, cif_file))

    # dat_file_path = os.path.join(root_dir, 'dat_vestacif_files', dat_file)
    # shutil.copy(dat_file_path, os.path.join(subfolder_path, dat_file))

    shutil.move(pcr_file_path, os.path.join(subfolder_path, pcr_file_name))

    shutil.copy(cif_file_path, os.path.join(subfolder_path, cif_file))

    dat_file_path = os.path.join(root_dir, 'dat_vestacif_files', dat_file)
    shutil.copy(dat_file_path, os.path.join(subfolder_path, dat_file))

    if reference_pcr and reference_pcr.lower() == "si_scale1000times.pcr":
        pcr_in_subfolder = os.path.join(subfolder_path, pcr_file_name)
        try:
            from fix_si_scale1000times_pcr import fix_si_scale1000times_pcr
            print("Applying additional fixes for Si_scale1000times format...")
            fix_si_scale1000times_pcr(pcr_in_subfolder)
        except ImportError:
            print("Warning: Could not import fix_si_scale1000times_pcr, skipping additional fixes")
    elif reference_pcr and reference_pcr.lower() == "ceo2.pcr":
        pcr_in_subfolder = os.path.join(subfolder_path, pcr_file_name)
        try:
            from fix_ceo2_pcr import fix_ceo2_pcr
            print("Applying additional fixes for CeO2 format...")
            fix_ceo2_pcr(pcr_in_subfolder)
        except ImportError:
            print("Warning: Could not import fix_ceo2_pcr, skipping additional fixes")
    elif reference_pcr and reference_pcr.lower() == "tbbaco.pcr":
        pcr_in_subfolder = os.path.join(subfolder_path, pcr_file_name)
        try:
            from fix_tbbaco_pcr import fix_tbbaco_pcr_with_root
            print("Applying additional fixes for tbbaco format...")
            fix_tbbaco_pcr_with_root(pcr_in_subfolder, root_dir)
        except ImportError:
            print("Warning: Could not import fix_tbbaco_pcr, skipping additional fixes")

    print("###")

    for atom, positions in transformed_positions.iteritems():
        print("Transformed positions for {}:".format(atom))
        for position in positions:
            print("  {}".format(position))
        print("Independent positions:")
        for position in independent_positions[atom]:
            print("  {}".format(position))
        print("Number of independent positions: {}".format(len(independent_positions[atom])))
        print("Occupancy for {}: {:.5f}".format(atom, occupancies[atom]))

    print("###")

    print("###")
    print("Imported parameters:")
    print("1. Lattice parameters:")
    for param in lattice_params:
        print("   {}".format(param))
    print("\n2. Space group symbol:")
    print("   {}".format(space_group))
    print("\n3. Atomic positions (x, y, z):")
    for atom in atom_positions:
        print("   {}: ({}, {}, {})".format(atom[0], atom[1], atom[2], atom[3]))
    print("\n4. Occupancies:")
    for atom in atom_positions:
        print("   {}: {:.5f}".format(atom[0], occupancies[atom[0]]))

    print("\n5. Common name:")
    print("   {}".format(chemical_name))
    print("###")
if __name__ == "__main__":
    main()