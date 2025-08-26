# -*- coding: utf-8 -*-
import re
import os
import pandas as pd
import shutil
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pprint
import struct

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def extract_and_plot_prf_data(prf_file_path, interpolated_df, parameter_name, generate_figures, display_figures, generate_ycal_plots):
    with open(prf_file_path, 'r') as file:
        lines = file.readlines()

    start_index = -1
    end_index = -1

    for i, line in enumerate(lines):
        if re.match(r'^\s*2Theta\s+Yobs\s+Ycal\s+Yobs-Ycal\s+Backg\s+Posr\s+\(hkl\)\s+K', line):
            start_index = i + 1
            break

    if start_index == -1:
        print("Error: Could not find the 2Theta header line in PRF file: {}".format(prf_file_path))
        return

    for i in range(start_index, len(lines)):
        if '(' in lines[i] and ')' in lines[i] and ':' in lines[i]:
            end_index = i
            break

    if end_index == -1:
        end_index = len(lines)

    data = []
    for line in lines[start_index:end_index]:
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 3:
            try:
                data.append([float(parts[0]), float(parts[2])])  # 2Theta and Ycal
            except (ValueError, IndexError):
                continue  # Skip lines that can't be parsed

    df = pd.DataFrame(data, columns=['2Theta', 'Ycal'])

    plt.figure(figsize=(10, 6))
    plt.plot(df['2Theta'], df['Ycal'], ':', color='green', alpha=0.3, label='_nolegend_')  # Hide from legend
    plt.scatter(df['2Theta'], df['Ycal'], edgecolors='green', marker='x', label='Original Ycal')

    
    if parameter_name in interpolated_df.columns:
        col_index = interpolated_df.columns.get_loc(parameter_name)
    else:
        print("Error: {} not found in interpolated_df columns".format(parameter_name))
        return

    plt.plot(interpolated_df.iloc[:, 0], interpolated_df.iloc[:, col_index], ':', color='red', alpha=0.3, label='_nolegend_')  # Add dotted line in red
    plt.scatter(interpolated_df.iloc[:, 0], interpolated_df.iloc[:, col_index], edgecolors='red', facecolors='none', label='Interpolated Ycal')
    
    plt.xlabel('2Theta')
    plt.ylabel('Ycal')
    plt.title('Comparison of Original and Interpolated Data, {}'.format(parameter_name))
    plt.legend()
    plt.grid(True)

    if generate_figures == 'Y' and generate_ycal_plots == 'Y':
        subfolder_path = os.path.dirname(os.path.dirname(prf_file_path))
        output_figures_folder = os.path.join(subfolder_path, 'output_figures')
        if not os.path.exists(output_figures_folder):
            os.makedirs(output_figures_folder)
        
        ycal_plots_folder = os.path.join(output_figures_folder, 'ycal_original_versus_interpolated_plots')
        if not os.path.exists(ycal_plots_folder):
            os.makedirs(ycal_plots_folder)
            
        plot_filename = "comparison_{}_2thetaYcal_plot.png".format(parameter_name)
        plot_filepath = os.path.join(ycal_plots_folder, plot_filename)
        plt.savefig(plot_filepath)

    if generate_figures == 'Y' and display_figures == 'Y':
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    else:
        plt.close()

def extract_prf_data(prf_file_path):
    with open(prf_file_path, 'r') as file:
        lines = file.readlines()

    start_index = -1
    for i, line in enumerate(lines):
        if re.search(r'^\s*2Theta\s+Yobs\s+Ycal', line):
            start_index = i + 1
            break

    if start_index == -1:
        print("Error: Could not find the expected '2Theta Yobs Ycal' header in PRF file: {}".format(prf_file_path))
        return None

    data = []
    for line in lines[start_index:]:
        line = line.strip()
        # Stop if we hit a line with Bragg reflection format (contains parentheses and colon)
        if '(' in line and ')' in line and ':' in line:
            continue
        # Skip empty lines or non-data lines
        if not line or not re.match(r'^\s*[-+]?\d', line):
            continue
        
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 5:  # At least 5 columns for 2Theta, Yobs, Ycal, diff, backg
            try:
                two_theta = float(parts[0])
                yobs = float(parts[1])
                ycal = float(parts[2])
                diff = float(parts[3])
                backg = float(parts[4])
                data.append([two_theta, yobs, ycal, diff, backg])
            except (ValueError, IndexError):
                continue

    # Convert to DataFrame
    if data:
        df = pd.DataFrame(data, columns=['2Theta', 'Yobs', 'Ycal', 'Diff', 'Backg'])
        return df
    else:
        print("Warning: No valid data extracted from PRF file: {}".format(prf_file_path))
        return None
    
def create_combined_dat_files(subfolder_path, data_folder, lattice_type, parameters):
    base_name_without_timestamp = data_folder.split('_')[0]
    
    sampling_range_input = None
    for key in parameters:
        if key.startswith("14>"):
            sampling_range_input = parameters[key]
            break
    
    if sampling_range_input is None:
        sampling_range_input = "N;10,0.4,154"  # Default from inputs.txt

    data_frames = {}

    for folder in os.listdir(subfolder_path):
        folder_path = os.path.join(subfolder_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".prf"):
                    prf_file_path = os.path.join(folder_path, file)
                    df = extract_prf_data(prf_file_path)
                    if df is not None:
                        data_frames[folder] = df
                    break

    if len(data_frames) == 0:
        print("Error: No PRF files found in the subfolders.")
        return

    if sampling_range_input.startswith('Y'):
        min_theta = min(df['2Theta'].min() for df in data_frames.values() if df is not None)
        max_theta = max(df['2Theta'].max() for df in data_frames.values() if df is not None)
        common_theta_start = np.floor(min_theta)
        common_theta_end = np.ceil(max_theta / 10) * 10 + 10
        step = 0.01  # Default step size
    else:
        _, custom_range = sampling_range_input.split(";")
        common_theta_start, step, common_theta_end = map(float, custom_range.split(','))

    common_theta = np.round(np.arange(common_theta_start, common_theta_end + step, step), 2)

    interpolated_data = {'2_Theta_(deg)': common_theta}
    for folder, df in data_frames.items():
        if df is not None:
            col_name = folder
            interpolated_data[col_name] = np.interp(common_theta, df['2Theta'], df['Ycal'])

    combined_df = pd.DataFrame(interpolated_data).round(2)

    columns_order = ['2_Theta_(deg)'] + sorted([col for col in combined_df.columns if col != '2_Theta_(deg)'])
    combined_df = combined_df[columns_order]

    combined_df['2_Theta_(deg)'] = combined_df['2_Theta_(deg)'].apply(lambda x: "{:.2f}".format(x))

    for col in [c for c in combined_df.columns if c != '2_Theta_(deg)']:
        combined_df[col] = combined_df[col].apply(lambda x: "{:.2f}".format(x))

    use_binary = parameters.get('21>Generate .BIN files instead of .DAT files? (Y/N):').strip().upper() == 'Y'
    
    if use_binary:
        combined_dat_filename = "%s_simulated_data_column_for_plot.bin" % base_name_without_timestamp
        combined_dat_filepath = os.path.join(subfolder_path, combined_dat_filename)
        with open(combined_dat_filepath, 'wb') as f:
            num_rows, num_cols = combined_df.shape
            f.write(struct.pack('2i', num_rows, num_cols))
            numpy_array = combined_df.values.astype(np.float32)
            f.write(numpy_array.tobytes())
        print("Combined data saved to %s" % combined_dat_filepath)
    else:
        combined_dat_filename = "%s_simulated_data_column_for_plot.dat" % base_name_without_timestamp
        combined_dat_filepath = os.path.join(subfolder_path, combined_dat_filename)
        combined_df.to_csv(combined_dat_filepath, index=False, sep=' ', header=False)
        print("Combined data saved to %s" % combined_dat_filepath)

    if use_binary:
        two_theta_range_filename = "%s_simulated_data_2theta_param_info.bin" % base_name_without_timestamp
        two_theta_range_filepath = os.path.join(subfolder_path, two_theta_range_filename)
        
        param_list = ['zero', 'background']
        
        if lattice_type == 'cubic':
            param_list.append('lattice parameter a')
        elif lattice_type == 'tetragonal':
            param_list.extend(['lattice parameter a', 'lattice parameter c'])
        elif lattice_type == 'orthorhombic':
            param_list.extend(['lattice parameter a', 'lattice parameter b', 'lattice parameter c'])
        elif lattice_type == 'hexagonal':
            param_list.extend(['lattice parameter a', 'lattice parameter c'])
        elif lattice_type == 'monoclinic':
            param_list.extend(['lattice parameter a', 'lattice parameter b', 'lattice parameter c', 'lattice parameter beta'])
        elif lattice_type == 'triclinic':
            param_list.extend(['lattice parameter a', 'lattice parameter b', 'lattice parameter c',
                            'lattice parameter alpha', 'lattice parameter beta', 'lattice parameter gamma'])
        elif lattice_type == 'trigonal':
            param_list.extend(['lattice parameter a', 'lattice parameter alpha'])

        atom_names = set()
        for folder in os.listdir(subfolder_path):
            folder_path = os.path.join(subfolder_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".pcr"):
                        pcr_file_path = os.path.join(folder_path, file)
                        with open(pcr_file_path, 'r') as pcr_file:
                            in_atom_section = False
                            for line in pcr_file:
                                if '!Atom   Typ' in line:
                                    in_atom_section = True
                                    continue
                                elif in_atom_section and '!------->' in line:
                                    in_atom_section = False
                                    break
                                elif in_atom_section and re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', line):
                                    parts = line.split()
                                    if len(parts) > 0:
                                        atom_names.add(parts[0])

        for atom in sorted(atom_names):
            param_list.append('Biso_{}'.format(atom))

        param_list.append('Scale factor')
        param_list.extend(['U parameter', 'V parameter', 'W parameter'])

        with open(two_theta_range_filepath, 'wb') as f:
            f.write(struct.pack('3d', common_theta_start, common_theta_end, step))
            
            f.write(struct.pack('i', len(param_list)))
            
            for param in param_list:
                f.write(struct.pack('32s', param.encode('utf-8')))
    else:
        two_theta_range_filename = "%s_simulated_data_2theta_param_info.dat" % base_name_without_timestamp
        two_theta_range_filepath = os.path.join(subfolder_path, two_theta_range_filename)
        with open(two_theta_range_filepath, 'w') as f:
            f.write("Initial=%.2f, Final=%.2f, Step=%.2f\n" % (common_theta_start, common_theta_end, step))
            f.write("\nzero\nbackground\n")
            
            # Write lattice parameters based on lattice type
            if lattice_type == 'cubic':
                f.write("lattice parameter a\n")
            elif lattice_type == 'tetragonal':
                f.write("lattice parameter a\n")
                f.write("lattice parameter c\n")
            elif lattice_type == 'orthorhombic':
                f.write("lattice parameter a\n")
                f.write("lattice parameter b\n")
                f.write("lattice parameter c\n")
            elif lattice_type == 'hexagonal':
                f.write("lattice parameter a\n")
                f.write("lattice parameter c\n")
            elif lattice_type == 'monoclinic':
                f.write("lattice parameter a\n")
                f.write("lattice parameter b\n")
                f.write("lattice parameter c\n")
                f.write("lattice parameter beta\n")
            elif lattice_type == 'triclinic':
                f.write("lattice parameter a\n")
                f.write("lattice parameter b\n")
                f.write("lattice parameter c\n")
                f.write("lattice parameter alpha\n")
                f.write("lattice parameter beta\n")
                f.write("lattice parameter gamma\n")
            elif lattice_type == 'trigonal':
                f.write("lattice parameter a\n")
                f.write("lattice parameter alpha\n")

            atom_names = set()
            for folder in os.listdir(subfolder_path):
                folder_path = os.path.join(subfolder_path, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith(".pcr"):
                            pcr_file_path = os.path.join(folder_path, file)
                            with open(pcr_file_path, 'r') as pcr_file:
                                in_atom_section = False
                                for line in pcr_file:
                                    if '!Atom   Typ' in line:
                                        in_atom_section = True
                                        continue
                                    elif in_atom_section and '!------->' in line:
                                        in_atom_section = False
                                        break
                                    elif in_atom_section and re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', line):
                                        parts = line.split()
                                        if len(parts) > 0:
                                            atom_names.add(parts[0])

            for atom in sorted(atom_names):
                f.write("Biso_{}\n".format(atom))

            f.write("Scale factor\n")
            f.write("U parameter\n")
            f.write("V parameter\n")
            f.write("W parameter\n")


def create_xyy_file_from_dat(dat_path, output_xyy_path, folder_params, lattice_type):
    df = pd.read_csv(dat_path, sep=' ', header=None)
    sorted_folders = sorted(list(folder_params.keys()), key=natural_sort_key)
    df.columns = ['2_Theta_(deg)'] + sorted_folders
    
    with open(output_xyy_path, 'w') as f:
        for col in df.columns[1:]:  # Skip the first column which is 2Theta
            row_values = " ".join(map(str, df[col].values))
            folder_name = col
            params = folder_params.get(folder_name, None)
            if params:
                param_list = [params['zero'], params['bg']]
                
                if lattice_type == 'cubic':
                    param_list.append(params['a'])
                elif lattice_type == 'tetragonal':
                    param_list.extend([params['a'], params['c']])
                elif lattice_type == 'orthorhombic':
                    param_list.extend([params['a'], params['b'], params['c']])
                elif lattice_type == 'hexagonal':
                    param_list.extend([params['a'], params['c']])
                elif lattice_type == 'monoclinic':
                    param_list.extend([params['a'], params['b'], params['c'], params['beta']])
                elif lattice_type == 'triclinic':
                    param_list.extend([params['a'], params['b'], params['c'], 
                                    params['alpha'], params['beta'], params['gamma']])
                elif lattice_type == 'trigonal':
                    param_list.extend([params['a'], params['alpha']])
                
                for atom_name in sorted(params['atom_names']):
                    if atom_name in params['atom_biso']:
                        param_list.append(params['atom_biso'][atom_name])
                
                param_list.extend([
                    "{:.10f}".format(float(params['scale'])),  # Converts to decimal (e.g., 0.0006782252)
                    params['U'],  # U parameter
                    params['V'],  # V parameter  
                    params['W']   # W parameter
                ])
                
                row_values += " " + " ".join(str(p) for p in param_list)
            else:
                print("No parameters found for folder_name:", folder_name)
            f.write(row_values + "\n")

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

    required_params = [
        '16>Generate all output figures (Y/N)?',
        '17>Display all output figures (Y/N)?',
        '19>Make simulated_dataset folder and save its files (sample PCR, PRF files) (Y/N)?:',
        '20>Make output_figures folder and save its files (analysis plots, distributions) (Y/N)?:',
        '21>Generate .BIN files instead of .DAT files? (Y/N):',
        '23>Input the percentage range for generating a random number around the Biso parameter with degrees of freedom (example: 3%):',
        '24>Input the percentage range for generating a random number around the scale factor parameter with degrees of freedom (example: 5%):',
        '25>Input the percentage range for generating a random number around the U, parameter with degrees of freedom (example: 1%):',
        '26>Input the percentage range for generating a random number around the V, parameter with degrees of freedom (example: 1%):',
        '27>Input the percentage range for generating a random number around the W, parameter with degrees of freedom (example: 1%):',
        '40>Generate classification profile plots (Y/N)?:',
        '41>Generate parameter shift distribution plots (Y/N)?:',
        '42>Generate PRF files folder (Y/N)?:',
        '43>Generate Rietveld refinement plots (Y/N)?:',
        '44>Generate Ycal original versus interpolated plots (Y/N)?:',
        '46>Atom data format in PCR file (enter "4-line" for standard format with thermal parameters, or "2-line" for simplified format without thermal parameters - check your PCR file to see if each atom has 4 lines or 2 lines of data):'
    ]
    
    for param in required_params:
        if param not in parameters:
            print("Warning: Missing '%s' in inputs.txt." % param)
            # Provide default values for missing parameters
            if param == '16>Generate all output figures (Y/N)?':
                parameters[param] = 'Y'
            elif param == '17>Display all output figures (Y/N)?':
                parameters[param] = 'N'
            elif param == '19>Make simulated_dataset folder and save its files (sample PCR, PRF files) (Y/N)?:':
                parameters[param] = 'Y'
            elif param == '20>Make output_figures folder and save its files (analysis plots, distributions) (Y/N)?:':
                parameters[param] = 'Y'
            elif param == '21>Generate .BIN files instead of .DAT files? (Y/N):':
                parameters[param] = 'N'
            elif param.startswith('2'):
                parameters[param] = '1%'  # Default for percentage parameters
            # Default values for the new parameters (generate all by default)
            elif param == '40>Generate classification profile plots (Y/N)?:':
                parameters[param] = 'Y'
            elif param == '41>Generate parameter shift distribution plots (Y/N)?:':
                parameters[param] = 'Y'
            elif param == '42>Generate PRF files folder (Y/N)?:':
                parameters[param] = 'Y'
            elif param == '43>Generate Rietveld refinement plots (Y/N)?:':
                parameters[param] = 'Y'
            elif param == '44>Generate Ycal original versus interpolated plots (Y/N)?:':
                parameters[param] = 'Y'
            elif param == '46>Atom data format in PCR file (enter "4-line" for standard format with thermal parameters, or "2-line" for simplified format without thermal parameters - check your PCR file to see if each atom has 4 lines or 2 lines of data):':
                parameters[param] = '4-line'  # Default to 4-line format if not specified

    return parameters

def read_modified_parameters(pcr_file_path, atom_format='4-line'):
    values = {
        'zero': '0.0',
        'bg': '0.0',
        'a': '0.0',
        'b': '0.0',
        'c': '0.0',
        'alpha': '0.0',
        'beta': '0.0',
        'gamma': '0.0',
        'scale': '0.0',
        'U': '0.0',
        'V': '0.0',
        'W': '0.0',
        'atom_biso': {},   # Dictionary to store atom-specific Biso values
        'atom_names': []   # List to preserve atom order
    }

    if not os.path.exists(pcr_file_path):
        print("Warning: PCR file does not exist: {}".format(pcr_file_path))
        return values

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
        while current_line < atom_section_end:
            if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', lines[current_line]):
                parts = lines[current_line].split()
                if len(parts) >= 6:
                    atom_name = parts[0]
                    biso_value = parts[5]
                    values['atom_biso'][atom_name] = biso_value
                    values['atom_names'].append(atom_name)
                current_line += lines_per_atom
            else:
                current_line += 1
    
    try:
        for i, line in enumerate(lines):
            if '!  Zero    Code    SyCos    Code   SySin    Code  Lambda     Code MORE ->Patt# 1' in line:
                zero_line_index = i + 1
                parts = re.split(r'\s+', lines[zero_line_index].strip())
                if parts and len(parts) > 0:
                    values['zero'] = parts[0]

            elif '!   Background coefficients/codes  for Pattern#  1  (Polynomial of 6th degree)' in line:
                bg_line_index = i + 1
                parts = re.split(r'\s+', lines[bg_line_index].strip())
                if parts and len(parts) > 0:
                    values['bg'] = parts[0]

            elif '!     a          b         c        alpha      beta       gamma      #Cell Info' in line:
                lattice_line_index = i + 1
                parts = re.split(r'\s+', lines[lattice_line_index].strip())
                if parts and len(parts) >= 6:
                    values['a'] = parts[0]
                    values['b'] = parts[1]
                    values['c'] = parts[2]
                    values['alpha'] = parts[3]
                    values['beta'] = parts[4]
                    values['gamma'] = parts[5]

            elif '!  Scale' in line and 'Strain-Model' in line:  # More robust line detection
                if i + 1 < len(lines):
                    scale_line = lines[i + 1]
                    try:
                        scale_str = scale_line.split()[0].strip()  # First field contains the scale
                        values['scale'] = "{:.7E}".format(float(scale_str))
                    except (ValueError, IndexError) as e:
                        print("Warning: Error reading Scale value: {}".format(e))
                        values['scale'] = '0.0'

            elif '!       U' in line and 'Size-Model' in line:  # More robust line detection
                if i + 1 < len(lines):
                    uvw_line = lines[i + 1]
                    try:
                        parts = uvw_line.split()
                        if len(parts) >= 3:
                            values['U'] = parts[0]
                            values['V'] = parts[1]
                            values['W'] = parts[2]
                    except (ValueError, IndexError) as e:
                        print("Warning: Error reading U,V,W values: {}".format(e))
                        values['U'] = '0.0'
                        values['V'] = '0.0'
                        values['W'] = '0.0'

    except Exception as e:
        print "Warning: Error reading parameters from %s: %s" % (pcr_file_path, str(e))
    
    return values

def find_subfolder_with_prefix(subfolder_path, prefix):
    for folder_name in os.listdir(subfolder_path):
        if folder_name.startswith(prefix):
            return folder_name
    raise IOError("No subfolder found with prefix '{}' in '{}'.".format(prefix, subfolder_path))

def create_binary_file_from_dat(dat_path, output_path, folder_params, lattice_type):
    if dat_path.endswith('.bin'):
        with open(dat_path, 'rb') as f:
            num_rows, num_cols = struct.unpack('2i', f.read(8))
            data = np.fromfile(f, dtype=np.float32)
            data = data.reshape(num_rows, num_cols)
            df = pd.DataFrame(data[:, 1:])  # Exclude 2theta column
    else:
        df = pd.read_csv(dat_path, sep=' ', header=None)
        if len(df.columns) > 1:
            df = df.iloc[:, 1:]  # Remove 2theta column
    
    with open(output_path, 'wb') as f:
        num_samples = len(folder_params)
        num_points = len(df.index)
        header = struct.pack('2i', num_samples, num_points)
        f.write(header)

        max_params = 0
        for folder_name in folder_params:
            params = folder_params[folder_name]
            param_count = 2  # zero, bg
            
            if lattice_type == 'cubic':
                param_count += 1  # a
            elif lattice_type in ['tetragonal', 'hexagonal']:
                param_count += 2  # a, c
            elif lattice_type == 'orthorhombic':
                param_count += 3  # a, b, c
            elif lattice_type == 'monoclinic':
                param_count += 4  # a, b, c, beta
            elif lattice_type == 'triclinic':
                param_count += 6  # a, b, c, alpha, beta, gamma
            elif lattice_type == 'trigonal':
                param_count += 2  # a, alpha
            
            if 'atom_biso' in params:
                param_count += len(params['atom_biso'])
            
            param_count += 4
            
            max_params = max(max_params, param_count)
        
        for i, folder_name in enumerate(folder_params.keys()):
            intensity_values = df.iloc[:, i].astype(np.float32).values
            f.write(intensity_values.tobytes())
            
            params = folder_params[folder_name]
            param_values = [float(params[k]) for k in ['zero', 'bg']]
            
            if lattice_type == 'cubic':
                param_values.append(float(params['a']))
            elif lattice_type in ['tetragonal', 'hexagonal']:
                param_values.extend([float(params['a']), float(params['c'])])
            elif lattice_type == 'orthorhombic':
                param_values.extend([float(params['a']), float(params['b']), float(params['c'])])
            elif lattice_type == 'monoclinic':
                param_values.extend([float(params['a']), float(params['b']), float(params['c']), float(params['beta'])])
            elif lattice_type == 'triclinic':
                param_values.extend([float(params['a']), float(params['b']), float(params['c']), 
                                   float(params['alpha']), float(params['beta']), float(params['gamma'])])
            elif lattice_type == 'trigonal':
                param_values.extend([float(params['a']), float(params['alpha'])])
            
            if 'atom_biso' in params and 'atom_names' in params:
                for atom_name in sorted(params['atom_names']):
                    if atom_name in params['atom_biso']:
                        param_values.append(float(params['atom_biso'][atom_name]))
            
            param_values.extend([
                float(params['scale']),
                float(params['U']),
                float(params['V']),
                float(params['W'])
            ])
            
            param_array = np.array(param_values, dtype=np.float32)
            f.write(param_array.tobytes())

def read_binary_files(file_path, shape=None, dtype=np.float32):
    data = np.fromfile(file_path, dtype=dtype)
    if shape is not None:
        data = data.reshape(shape)
    return data

def condense(df, posr_available):
    
    yobs_max = df['Yobs'].max()
    diff_max = df['Diff'].max()
    diff_min = df['Diff'].min()
    
    desired_spacing = yobs_max * 0.03
    
    diff_offset = -(abs(diff_min) + desired_spacing)
    
    diff_adjusted = df['Diff'] + diff_offset
    
    min_posr_height = yobs_max * 0.05 if posr_available else None
    
    return diff_adjusted, min_posr_height

def create_rietveld_plots_from_prf(prf_file_path, generate_figures, display_figures, generate_rietveld_plots, generate_prf_files):
    if not os.path.exists(prf_file_path):
        print("Error: PRF file not found: {}".format(prf_file_path))
        return
        
    with open(prf_file_path, 'r') as file:
        lines = file.readlines()

    start_index = -1
    bragg_start_index = -1

    for i, line in enumerate(lines):
        if re.search(r'^\s*2Theta\s+Yobs\s+Ycal', line):
            start_index = i + 1
            break

    if start_index == -1:
        print("Error: Could not find the expected '2Theta Yobs Ycal' header in PRF file:", prf_file_path)
        return

    data = []
    bragg_positions = []
    
    for i, line in enumerate(lines[start_index:], start=start_index):
        line = line.strip()
        if not line:
            continue
            
        if '(' in line and ')' in line and ':' in line:
            if bragg_start_index == -1:
                bragg_start_index = i
            parts = line.split()
            try:
                theta = float(parts[0])  # First field is 2theta
                bragg_positions.append(theta)
                continue
            except (ValueError, IndexError):
                continue

        if bragg_start_index == -1 or i < bragg_start_index:
            parts = line.split()
            if len(parts) >= 5:  # At least 2Theta, Yobs, Ycal, Diff, Backg
                try:
                    two_theta = float(parts[0])
                    yobs = float(parts[1])
                    ycal = float(parts[2])
                    diff = float(parts[3])
                    backg = float(parts[4])
                    data.append([two_theta, yobs, ycal, diff, backg])
                except (ValueError, IndexError):
                    continue

    if not data:
        print("Error: No valid data extracted from PRF file:", prf_file_path)
        return
        
    df = pd.DataFrame(data, columns=['2Theta', 'Yobs', 'Ycal', 'Diff', 'Backg'])

    diff_adjusted, min_posr_height = condense(df, posr_available=True)

    plt.figure(figsize=(12, 8))
    
    plt.plot(df['2Theta'], df['Ycal'], 'k-', linewidth=0.5, label='Ycalc')
    
    plt.scatter(df['2Theta'], df['Yobs'], 
               facecolors='none', 
               edgecolors='red',
               marker='o',
               s=3,  # Small points
               linewidth=0.5,  # Thin edge
               label='Yobs')
    
    plt.plot(df['2Theta'], diff_adjusted, 'b-', linewidth=0.5, label='Yobs-Ycalc')

    if bragg_positions:
        tick_bottom = diff_adjusted.max() + (df['Yobs'].max() - df['Yobs'].min()) * 0.07
        tick_height = (df['Yobs'].max() - df['Yobs'].min()) * 0.10
        
        for theta in bragg_positions:
            plt.vlines(theta, 
                      ymin=tick_bottom,
                      ymax=tick_bottom + tick_height,
                      colors='green', 
                      linestyles='-',
                      linewidth=0.3)
        
        plt.vlines([], [], [], 
                  colors='green',
                  linestyles='-',
                  linewidth=0.3,
                  label='Bragg Position')

    plt.grid(True, linestyle='-', alpha=0.3)

    x_min = df['2Theta'].min()
    x_max = df['2Theta'].max()
    plt.xlim(x_min, x_max)

    plt.xlabel('2-theta (deg.)')
    plt.ylabel('Intensity (arb. units)')
    sample_name = os.path.basename(prf_file_path).split('.')[0]
    plt.title('Rietveld Refinement Plot - %s' % sample_name)
    
    plt.legend(fontsize=8)

    if generate_figures == 'Y':
        subfolder_path = os.path.dirname(os.path.dirname(prf_file_path))
        output_figures_folder = os.path.join(subfolder_path, 'output_figures')
        if not os.path.exists(output_figures_folder):
            os.makedirs(output_figures_folder)
        
        if generate_rietveld_plots == 'Y':
            rietveld_plots_folder = os.path.join(output_figures_folder, 'rietveld_refinement_plots')
            if not os.path.exists(rietveld_plots_folder):
                os.makedirs(rietveld_plots_folder)
                
            plot_filename = "{}_rietveld_fit.png".format(sample_name)
            plot_filepath = os.path.join(rietveld_plots_folder, plot_filename)
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        if generate_prf_files == 'Y':
            prf_files_folder = os.path.join(output_figures_folder, 'PRF_files')
            if not os.path.exists(prf_files_folder):
                os.makedirs(prf_files_folder)
            
            shutil.copy(prf_file_path, prf_files_folder)

    if generate_figures == 'Y' and display_figures == 'Y':
        plt.show(block=False)
        plt.pause(1)
    plt.close()

def main():
    if len(sys.argv) != 2:
        print "Usage: python step4_overplots_3datfiles.py <path_to_inputs.txt>"
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    if not os.path.exists(input_file_path):
        print "Error: The file '%s' does not exist." % input_file_path
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print "Debug: Script directory: %s" % script_dir
    print "Debug: Working directory: %s" % working_dir
    print "Debug: Input file path: %s" % input_file_path
    
    input_filename = os.path.basename(input_file_path)
    timestamp = input_filename.replace('inputs_', '').replace('.txt', '')
    
    parameters = read_input_parameters(input_file_path)
    base_subfolder_name = parameters.get('5>Enter the name of the subfolder to be created in the data directory (the purpose of the subfolder is to organize and store related data files and refinement results for analysis):')
    subfolder_name = base_subfolder_name + "_" + timestamp
    
    subfolder_path = os.path.join(working_dir, 'data', subfolder_name)
    print "Debug: Subfolder path: %s" % subfolder_path
    
    if not os.path.exists(subfolder_path):
        print "Error: The subfolder '%s' does not exist." % subfolder_path
        sys.exit(1)

    lattice_type = parameters.get('11>Enter the lattice type (cubic, tetragonal, orthorhombic, hexagonal, monoclinic, triclinic, trigonal):')
    if not lattice_type:
        print "Warning: Lattice type not found in input parameters. Defaulting to 'cubic'."
        lattice_type = 'cubic'
    
    lattice_type = lattice_type.lower()  # Normalize to lowercase

    generate_figures = parameters.get('16>Generate all output figures (Y/N)?', 'Y').strip().upper()
    display_figures = parameters.get('17>Display all output figures (Y/N)?', 'Y').strip().upper()
    
    generate_classification_plots = parameters.get('40>Generate classification profile plots (Y/N)?:', 'Y').strip().upper()
    generate_parameter_shift_plots = parameters.get('41>Generate parameter shift distribution plots (Y/N)?:', 'Y').strip().upper()
    generate_prf_files = parameters.get('42>Generate PRF files folder (Y/N)?:', 'Y').strip().upper()
    generate_rietveld_plots = parameters.get('43>Generate Rietveld refinement plots (Y/N)?:', 'Y').strip().upper()
    generate_ycal_plots = parameters.get('44>Generate Ycal original versus interpolated plots (Y/N)?:', 'Y').strip().upper()

    if generate_figures == 'N':
        plt.ioff()  # Disable both generation and display of figures
    elif display_figures == 'N':
        plt.ioff()  # Disable display but allow generation
    else:
        plt.ion()  # Enable interactive mode for generating and displaying figures

    create_combined_dat_files(subfolder_path, subfolder_name, lattice_type, parameters)
   
    base_name_without_timestamp = subfolder_name.split('_')[0]
   
    use_binary = parameters.get('21>Generate .BIN files instead of .DAT files? (Y/N):').strip().upper() == 'Y'
    combined_dat_filename = "%s_simulated_data_column_for_plot.{}".format('bin' if use_binary else 'dat') % base_name_without_timestamp
    combined_dat_filepath = os.path.join(subfolder_path, combined_dat_filename)

    if use_binary:
        try:
            with open(combined_dat_filepath, 'rb') as f:
                num_rows, num_cols = struct.unpack('2i', f.read(8))
                data = np.fromfile(f, dtype=np.float32)
                data = data.reshape(num_rows, num_cols)
                combined_df = pd.DataFrame(data)
        except Exception as e:
            print("Error reading binary file: {}".format(e))
            sys.exit(1)
    else:
        try:
            combined_df = pd.read_csv(combined_dat_filepath, sep=' ', header=None)
        except Exception as e:
            print("Error reading DAT file: {}".format(e))
            sys.exit(1)

    interpolated_df = combined_df

    folder_names = [folder for folder in os.listdir(subfolder_path) 
                   if os.path.isdir(os.path.join(subfolder_path, folder)) 
                   and folder not in ['output_figures', 'source_files', 'simulated_dataset']]
    folder_names.sort(key=natural_sort_key)  # Sort folders naturally
    
    if len(folder_names) + 1 != len(interpolated_df.columns):
        print("Warning: Mismatch between number of folders ({}) and data columns ({})".format(
            len(folder_names), len(interpolated_df.columns)-1))
        print("Folders found:", folder_names)
        print("Number of columns in data:", len(interpolated_df.columns))
    else:
        interpolated_df.columns = ['2_Theta_(deg)'] + folder_names
    
    print "Contents of interpolated_df:"
    print interpolated_df.head()

    use_binary = parameters.get('21>Generate .BIN files instead of .DAT files? (Y/N):').strip().upper() == 'Y'

    atom_format = parameters.get('46>Atom data format in PCR file (enter "4-line" for standard format with thermal parameters, or "2-line" for simplified format without thermal parameters - check your PCR file to see if each atom has 4 lines or 2 lines of data):', '4-line')
    print "Using atom format: %s" % atom_format
    
    folder_params = {}
    folders_to_process = [folder for folder in os.listdir(subfolder_path)
                         if os.path.isdir(os.path.join(subfolder_path, folder)) 
                         and folder not in ['output_figures', 'source_files', 'simulated_dataset']]
    folders_to_process.sort(key=natural_sort_key)  # Sort folders naturally

    for folder in folders_to_process:
        folder_path = os.path.join(subfolder_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.pcr'):
                pcr_path = os.path.join(folder_path, file)
                modified_params = read_modified_parameters(pcr_path, atom_format)
                folder_params[folder] = modified_params
                break

    if use_binary:
        new_param_filename = "%s_simulated_data_row_param.bin" % base_name_without_timestamp.replace(" ", "_")
        new_param_path = os.path.join(subfolder_path, new_param_filename)
        create_binary_file_from_dat(combined_dat_filepath, new_param_path, folder_params, lattice_type)
    else:
        new_param_filename = "%s_simulated_data_row_param.dat" % base_name_without_timestamp.replace(" ", "_")
        new_param_path = os.path.join(subfolder_path, new_param_filename)
        create_xyy_file_from_dat(combined_dat_filepath, new_param_path, folder_params, lattice_type)

    print("Parameter file has been created at %s" % new_param_path)

    for folder in os.listdir(subfolder_path):
        folder_path = os.path.join(subfolder_path, folder)
        if os.path.isdir(folder_path) and folder not in ['output_figures', 'source_files', 'simulated_dataset']:
            prf_files = [f for f in os.listdir(folder_path) if f.endswith('.prf')]
            if len(prf_files) == 1:
                prf_file_path = os.path.join(folder_path, prf_files[0])
                print "Processing PRF file {} from {}".format(prf_files[0], folder)  # Debug print
                if folder in interpolated_df.columns:
                    extract_and_plot_prf_data(prf_file_path, interpolated_df, folder, generate_figures, display_figures, generate_ycal_plots)
                create_rietveld_plots_from_prf(prf_file_path, generate_figures, display_figures, generate_rietveld_plots, generate_prf_files)
            elif len(prf_files) > 1:
                print "Multiple .prf files found in the folder {}. Please specify which file to use.".format(folder)
                for i, prf_file in enumerate(prf_files):
                    print "[{}] {}".format(i + 1, prf_file)
                file_index = int(raw_input("Enter the number corresponding to the .prf file: ").strip()) - 1
                if 0 <= file_index < len(prf_files):
                    prf_file_path = os.path.join(folder_path, prf_files[file_index])
                    print "Processing selected PRF file: {}".format(prf_files[file_index])
                    if folder in interpolated_df.columns:
                        extract_and_plot_prf_data(prf_file_path, interpolated_df, folder, generate_figures, display_figures)
                    create_rietveld_plots_from_prf(prf_file_path, generate_figures, display_figures)
                else:
                    print "Invalid selection."
            else:
                print "No .prf files found in the folder {}. Check if AutoFP refinement completed successfully.".format(folder)

    simulated_dataset_folder = os.path.join(subfolder_path, "simulated_dataset")
    if not os.path.exists(simulated_dataset_folder):
        os.makedirs(simulated_dataset_folder)

    for folder_name in os.listdir(subfolder_path):
        folder_path = os.path.join(subfolder_path, folder_name)
        if (os.path.isdir(folder_path) and 
            folder_name not in ['output_figures', 'source_files', 'simulated_dataset']):
            old_folder_path = os.path.join(subfolder_path, folder_name)
            
            if folder_name.startswith(base_name_without_timestamp):
                sample_number = len([f for f in os.listdir(simulated_dataset_folder) 
                                  if f.startswith("sample")]) + 1
                new_name = "sample{}".format(sample_number)
                new_folder_path = os.path.join(simulated_dataset_folder, new_name)
            else:
                new_folder_path = os.path.join(simulated_dataset_folder, folder_name)
                
            def safe_rename(old_folder_path, new_folder_path, retries=3, delay=5):
                for i in range(retries):
                    try:
                        if os.path.exists(new_folder_path):
                            # Skip if folder already exists with correct name
                            break
                        shutil.move(old_folder_path, new_folder_path)
                        return True
                    except Exception as e:
                        print("Retrying move due to error: {}".format(e))
                        time.sleep(delay)
                return False

            safe_rename(old_folder_path, new_folder_path)

    source_files_folder = os.path.join(subfolder_path, 'source_files')
    if not os.path.exists(source_files_folder):
        os.makedirs(source_files_folder)

    base_name = os.path.splitext(parameters.get('3>Name of .PCR file to be generated'))[0]
    input_files = [base_name + ".dat", base_name + ".pcr", base_name + "_vesta.cif"]
    for input_file in input_files:
        input_file_path = os.path.join(subfolder_path, input_file)
        if os.path.exists(input_file_path):
            shutil.move(input_file_path, source_files_folder)

    keep_samples = parameters.get('19>Make simulated_dataset folder and save its files (sample PCR, PRF files) (Y/N)?:', 'N').strip().upper()
    keep_analysis = parameters.get('20>Make output_figures folder and save its files (analysis plots, distributions) (Y/N)?:', 'N').strip().upper()

    if keep_samples == 'N':
        if os.path.exists(simulated_dataset_folder) and os.path.isdir(simulated_dataset_folder):
            shutil.rmtree(simulated_dataset_folder)
            print("Deleted folder: {}".format(simulated_dataset_folder))
    else:
        print("Keeping simulated_dataset folder as per user configuration.")

    output_figures_folder = os.path.join(subfolder_path, "output_figures")
    if keep_analysis == 'N':
        if os.path.exists(output_figures_folder) and os.path.isdir(output_figures_folder):
            shutil.rmtree(output_figures_folder)
            print("Deleted folder: {}".format(output_figures_folder))
    else:
        print("Keeping output_figures folder as per user configuration.")

if __name__ == "__main__":
    main()