#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import re
from collections import OrderedDict

def parse_arguments():
    args = {}
    
    named_args = True
    for arg in sys.argv[1:]:
        if not arg.startswith('--'):
            named_args = False
            break
    
    if named_args:
        for arg in sys.argv[1:]:
            if '=' in arg:
                name, value = arg.split('=', 1)
                name = name.lstrip('-')
                
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                args[name] = value
        
        args['input_file'] = args.get('input_file', 'inputs.txt')
        args['output_dir'] = args.get('output_dir', 'macro_inputs')
        args['count'] = int(args.get('count', '1'))
        
        args['zero'] = int(args.get('zero', '0'))
        args['lattice'] = int(args.get('lattice', '0'))
        args['biso'] = int(args.get('biso', '0'))
        args['scale'] = int(args.get('scale', '0'))
        args['u'] = int(args.get('u', '0'))
        args['v'] = int(args.get('v', '0'))
        args['w'] = int(args.get('w', '0'))
        
        args['zero_shifts'] = args.get('zero_shifts', '')
        args['lattice_shifts'] = args.get('lattice_shifts', '')
        args['biso_shifts'] = args.get('biso_shifts', '')
        args['scale_shifts'] = args.get('scale_shifts', '')
        args['u_shifts'] = args.get('u_shifts', '')
        args['v_shifts'] = args.get('v_shifts', '')
        args['w_shifts'] = args.get('w_shifts', '')
        
        args['disable_partial_rerun'] = 'disable_partial_rerun' in args
        
    else:
        if len(sys.argv) < 4:
            print("Usage: {} <input_file> <output_dir> <count>".format(sys.argv[0]))
            sys.exit(1)
        
        args['input_file'] = sys.argv[1]
        args['output_dir'] = sys.argv[2]
        args['count'] = int(sys.argv[3])
        
        args['zero'] = 0
        args['lattice'] = 0
        args['biso'] = 0
        args['scale'] = 0
        args['u'] = 0
        args['v'] = 0
        args['w'] = 0
        args['zero_shifts'] = ''
        args['lattice_shifts'] = ''
        args['biso_shifts'] = ''
        args['scale_shifts'] = ''
        args['u_shifts'] = ''
        args['v_shifts'] = ''
        args['w_shifts'] = ''
        args['disable_partial_rerun'] = False
    
    return args

def read_input_file(file_path):
    if not os.path.exists(file_path):
        print("Error: Input file '{}' not found.".format(file_path))
        sys.exit(1)
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        params = OrderedDict()
        param_key = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if '>' in line:
                param_key = line
                params[param_key] = {'value': '', 'line_index': i}
            elif param_key and i == params[param_key]['line_index'] + 1:
                params[param_key]['value'] = line
        
        return lines, params
    
    except Exception as e:
        print("Error reading input file: {}".format(e))
        sys.exit(1)

def parse_parameter_shifts(shift_str):
    if not shift_str or shift_str.strip() == '':
        return []
    
    # Remove quotes if present
    if shift_str.startswith('"') and shift_str.endswith('"'):
        shift_str = shift_str[1:-1]
    
    return [s.strip() for s in shift_str.split(',') if s.strip()]

def calculate_fine_tuned_values(params, param_key, shift_values, iterations):
    if param_key not in params:
        return []
    
    try:
        original_value = float(params[param_key]['value'].replace('%', ''))
    except ValueError:
        print(f"Warning: Could not parse original value for {param_key}")
        return []
    
    if not shift_values:
        return []
    
    try:
        shift_amount = float(shift_values[0].replace('%', ''))
    except ValueError:
        print(f"Warning: Could not parse shift value '{shift_values[0]}'")
        return []
    
    fine_tuned_values = []
    current_value = original_value
    
    for i in range(iterations):
        current_value -= shift_amount
        if '%' in params[param_key]['value']:
            fine_tuned_values.append(f"{current_value}%")
        else:
            fine_tuned_values.append(str(current_value))
    
    return fine_tuned_values

def create_fine_tuned_files(args, lines, params):
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
    
    param_mapping = {
        'zero': '7>Input the maximum absolute shift range for generating a random number around the zero parameter value (example: 0.03). You may ignore this part if you did not skip part 6>:',
        'lattice': '12>Input the percentage range for generating a random number around the lattice parameters with degrees of freedom (example: 3%). You may ignore this part if you did not skip part 10>:',
        'biso': '23>Input the percentage range for generating a random number around the Biso parameter with degrees of freedom (example: 3%):',
        'scale': '24>Input the percentage range for generating a random number around the scale factor parameter with degrees of freedom (example: 5%):',
        'u': '25>Input the percentage range for generating a random number around the U, parameter with degrees of freedom (example: 1%):',
        'v': '26>Input the percentage range for generating a random number around the V, parameter with degrees of freedom (example: 1%):',
        'w': '27>Input the percentage range for generating a random number around the W, parameter with degrees of freedom (example: 1%):'
    }
    
    shift_values = {
        'zero': parse_parameter_shifts(args['zero_shifts']),
        'lattice': parse_parameter_shifts(args['lattice_shifts']),
        'biso': parse_parameter_shifts(args['biso_shifts']),
        'scale': parse_parameter_shifts(args['scale_shifts']),
        'u': parse_parameter_shifts(args['u_shifts']),
        'v': parse_parameter_shifts(args['v_shifts']),
        'w': parse_parameter_shifts(args['w_shifts'])
    }
    
    fine_tuned_values = {}
    for param_name, param_key in param_mapping.items():
        if args[param_name] == 1 and shift_values[param_name]:
            fine_tuned_values[param_name] = calculate_fine_tuned_values(
                params, param_key, shift_values[param_name], args['count'])
    
    created_files = []
    for i in range(args['count']):
        modified_lines = lines.copy()
        
        for param_name, param_key in param_mapping.items():
            if param_name in fine_tuned_values and i < len(fine_tuned_values[param_name]):
                line_idx = params[param_key]['line_index'] + 1
                new_value = fine_tuned_values[param_name][i]
                
                if line_idx < len(modified_lines):
                    modified_lines[line_idx] = new_value + '\n'
        
        partial_rerun_key = '32>Allow partial re-run if 80/20 ratio not met? (Y/N)?:'
        if partial_rerun_key in params:
            line_idx = params[partial_rerun_key]['line_index'] + 1
            if line_idx < len(modified_lines):
                modified_lines[line_idx] = 'N\n'
        
        output_path = os.path.join(args['output_dir'], f"inputs_ft_{i+1}.txt")
        
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)
        
        created_files.append(output_path)
        print(f"Created file: {output_path}")
    
    return created_files

def main():
    args = parse_arguments()
    
    print(f"Input file: {args['input_file']}")
    print(f"Output directory: {args['output_dir']}")
    print(f"Number of files to create: {args['count']}")
    
    lines, params = read_input_file(args['input_file'])
    
    created_files = create_fine_tuned_files(args, lines, params)
    
    print(f"\nCreated {len(created_files)} fine-tuned input files.")
    return 0

if __name__ == "__main__":
    main()