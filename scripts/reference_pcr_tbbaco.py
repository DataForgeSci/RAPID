# -*- coding: utf-8 -*-

import os
import re
import sys

def find_reference_pcr_file(ref_pcr_name, root_dir):

    ref_path = os.path.join(root_dir, 'dat_vestacif_files', 'reference_pcr_format', ref_pcr_name)
    if os.path.exists(ref_path):
        print("Found reference PCR in reference_pcr_format folder: {}".format(ref_path))
        return ref_path
    
    ref_path = os.path.join(root_dir, 'dat_vestacif_files', ref_pcr_name)
    if os.path.exists(ref_path):
        print("Found reference PCR in dat_vestacif_files folder: {}".format(ref_path))
        return ref_path
    
    ref_path = os.path.join(os.getcwd(), ref_pcr_name)
    if os.path.exists(ref_path):
        print("Found reference PCR in current directory: {}".format(ref_path))
        return ref_path
    
    print("Warning: Reference PCR file '{}' not found".format(ref_pcr_name))
    return None

def extract_reference_parameters(ref_pcr_path):

    if not os.path.exists(ref_pcr_path):
        print("Error: Reference PCR file does not exist: {}".format(ref_pcr_path))
        return None
    
    params = {
        'comment': '',
        'lambda1': '',
        'lambda2': '',
        'ratio': '',
        'bkpos': '',
        'wdt': '',
        'thmin': '',
        'step': '',
        'thmax': '',
        'excluded_regions': [],
        'zero': '',
        'bg_coeffs': [],
        'space_group': '',
        'scale': '',
        'shape1': '',
        'u': '',
        'v': '',
        'w': '',
        'x': '',
        'y': '',
        'asy1': '',
        'asy2': '',
        'asy3': '',
        'asy4': '',
        'th1': '',
        'th2': '',
        'atoms': [],
        'display_codes': {},  # For special display directives like #color green, #nodisplay, etc.
        'npr1': '',  # First Npr value
        'npr2': ''  # Second Npr value
    }
    
    try:
        with open(ref_pcr_path, 'r') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                # Comment line
                if line.startswith('COMM'):
                    params['comment'] = line.strip()[4:].strip()
                
                elif '!Job Npr Nph Nba Nex Nsc Nor Dum Iwg Ilo Ias Res Ste Nre Cry Uni Cor Opt Aut' in line and i+1 < len(lines):
                    npr_line = lines[i+1].strip()
                    parts = npr_line.split()
                    if len(parts) >= 2:
                        params['npr1'] = parts[1]  # Second value is Npr
                
                elif '!Nat Dis Ang Pr1 Pr2 Pr3 Jbt Irf Isy Str Furth       ATZ    Nvk Npr More' in line and i+1 < len(lines):
                    npr_line = lines[i+1].strip()
                    parts = npr_line.split()
                    if len(parts) >= 14:
                        params['npr2'] = parts[13]  # 14th value is Npr
                
                elif '! Lambda1  Lambda2' in line and i+1 < len(lines):
                    lambda_line = lines[i+1].strip()
                    parts = lambda_line.split()
                    if len(parts) >= 10:
                        params['lambda1'] = parts[0]
                        params['lambda2'] = parts[1]
                        params['ratio'] = parts[2]
                        params['bkpos'] = parts[3]
                        params['wdt'] = parts[4]
                
                elif '!     Thmin       Step       Thmax' in line and i+1 < len(lines):
                    theta_line = lines[i+1].strip()
                    parts = theta_line.split()
                    if len(parts) >= 3:
                        params['thmin'] = parts[0]
                        params['step'] = parts[1]
                        params['thmax'] = parts[2]
                
                elif '! Excluded regions' in line:
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith('!'):
                        line_parts = lines[j].strip().split()
                        if len(line_parts) >= 2:
                            params['excluded_regions'].append((line_parts[0], line_parts[1]))
                        j += 1
                
                elif '!  Zero    Code    SyCos' in line and i+1 < len(lines):
                    zero_line = lines[i+1].strip()
                    parts = zero_line.split()
                    if parts:
                        params['zero'] = parts[0]
                
                elif '!   Background coefficients/codes' in line and i+1 < len(lines):
                    bg_line = lines[i+1].strip()
                    params['bg_coeffs'] = bg_line.split()
                
                elif '<--Space group symbol' in line:
                    params['space_group'] = line.strip().split('<--')[0].strip()
                
                elif re.search(r'!Atom\s+Typ\s+X\s+Y\s+Z\s+Biso', line):
                    j = i + 1
                    while j < len(lines) and not re.search(r'!-+>\s+Profile Parameters', lines[j]):
                        if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', lines[j]):
                            parts = lines[j].split()
                            if len(parts) >= 7:
                                atom_info = {
                                    'name': parts[0],
                                    'type': parts[1],
                                    'x': parts[2],
                                    'y': parts[3],
                                    'z': parts[4],
                                    'biso': parts[5],
                                    'occ': parts[6]
                                }
                                if '#' in lines[j]:
                                    display_directive = lines[j].split('#')[1].strip()
                                    params['display_codes'][parts[0]] = display_directive
                                
                                params['atoms'].append(atom_info)
                        j += 1
                
                elif '!  Scale          Shape1' in line and i+1 < len(lines):
                    scale_line = lines[i+1].strip()
                    parts = scale_line.split()
                    if len(parts) >= 2:
                        params['scale'] = parts[0]
                        params['shape1'] = parts[1]
                
                elif '!       U            V            W' in line and i+1 < len(lines):
                    uvw_line = lines[i+1].strip()
                    parts = uvw_line.split()
                    if len(parts) >= 7:
                        params['u'] = parts[0]
                        params['v'] = parts[1]
                        params['w'] = parts[2]
                        params['x'] = parts[3]
                        params['y'] = parts[4]
                
                elif '!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4' in line and i+1 < len(lines):
                    asy_line = lines[i+1].strip()
                    parts = asy_line.split()
                    if len(parts) >= 6:
                        params['asy1'] = parts[2]
                        params['asy2'] = parts[3]
                        params['asy3'] = parts[4]
                        if len(parts) > 5:
                            params['asy4'] = parts[5]
                
                elif '!  2Th1/TOF1    2Th2/TOF2' in line and i+1 < len(lines):
                    th_line = lines[i+1].strip()
                    parts = th_line.split()
                    if len(parts) >= 2:
                        params['th1'] = parts[0]
                        params['th2'] = parts[1]
        
        print("\nParameters extracted from reference PCR (tbbaco):")
        print("Comment: {}".format(params['comment']))
        print("Lambda1, Lambda2: {}, {}".format(params['lambda1'], params['lambda2']))
        print("Theta range: {}, {}, {}".format(params['thmin'], params['step'], params['thmax']))
        print("Zero parameter: {}".format(params['zero']))
        print("Scale factor: {}".format(params['scale']))
        print("U, V, W: {}, {}, {}".format(params['u'], params['v'], params['w']))
        print("Space group: {}".format(params['space_group']))
        print("Npr values: {} (first), {} (second)".format(params['npr1'], params['npr2']))
        print("Number of atoms: {}".format(len(params['atoms'])))
        
        return params
    
    except Exception as e:
        print("Error extracting parameters from reference PCR: {}".format(e))
        return None

def apply_parameters_to_pcr(pcr_file_path, params):

    if not params:
        print("Error: No parameters to apply")
        return False
    
    try:
        with open(pcr_file_path, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if line.startswith('COMM') and params['comment']:
                lines[i] = "COMM   {}\n".format(params['comment'])
            
            elif '!Job Npr Nba Nex Nsc Nor Iwg Ilo Res Ste Uni Cor Anm Int' in line and i+1 < len(lines) and params['npr1']:
                npr_parts = lines[i+1].split()
                if len(npr_parts) >= 2:
                    npr_parts[1] = params['npr1']  # Replace Npr value
                    lines[i+1] = '   ' + '   '.join(npr_parts) + '  !-> Patt#: 1\n'
            
            elif '!Irf Npr Jtyp  Nsp_Ref Ph_Shift for Pattern#  1' in line and i+1 < len(lines) and params['npr2']:
                npr_parts = lines[i+1].split()
                if len(npr_parts) >= 2:
                    npr_parts[1] = params['npr2']  # Replace Npr value
                    lines[i+1] = '   ' + '   '.join(npr_parts) + '\n'
            
            elif '! Lambda1  Lambda2' in line and i+1 < len(lines) and params['lambda1'] and params['lambda2']:
                lambda_parts = lines[i+1].split()
                if len(lambda_parts) >= 10:
                    lambda_parts[0] = params['lambda1']
                    lambda_parts[1] = params['lambda2']
                    if params['ratio']:
                        lambda_parts[2] = params['ratio']
                    if params['bkpos']:
                        lambda_parts[3] = params['bkpos']
                    if params['wdt']:
                        lambda_parts[4] = params['wdt']
                    lines[i+1] = ' ' + ' '.join(lambda_parts) + '\n'
            
            elif '!     Thmin       Step       Thmax' in line and i+1 < len(lines) and params['thmin'] and params['step'] and params['thmax']:
                lines[i+1] = "     {}   {}   {}   0.000   0.000\n".format(
                    params['thmin'], params['step'], params['thmax'])
            
            elif '! Excluded regions' in line and params['excluded_regions']:
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('!'):
                    lines.pop(j)
                
                for region in params['excluded_regions']:
                    lines.insert(j, '        {:<10}{:<10}\n'.format(region[0], region[1]))
                    j += 1
            
            elif '!  Zero    Code    SyCos' in line and i+1 < len(lines) and params['zero']:
                zero_parts = lines[i+1].split()
                if zero_parts:
                    zero_parts[0] = params['zero']
                    lines[i+1] = '  ' + '    '.join(zero_parts) + '\n'
            
            elif '!   Background coefficients/codes' in line and i+1 < len(lines) and params['bg_coeffs']:
                lines[i+1] = '      ' + '      '.join(params['bg_coeffs']) + '\n'
            
            elif '<--Space group symbol' in line and params['space_group']:
                lines[i] = params['space_group'] + ' ' * 19 + '<--Space group symbol\n'
            
            elif re.search(r'!Atom\s+Typ\s+X\s+Y\s+Z\s+Biso', line):
                j = i + 2  # Start from first atom line
                while j < len(lines) and not re.search(r'!-+>\s+Profile Parameters', lines[j]):
                    if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', lines[j]):
                        parts = lines[j].split()
                        if len(parts) >= 7:
                            if j+1 < len(lines):
                                lines[j+1] = "                  0.00     0.00     0.00     0.00      0.00\n"
                            j += 4
                        else:
                            j += 1
                    else:
                        j += 1
            
            elif '!  Scale        Shape1' in line and i+1 < len(lines) and params['scale']:
                scale_line_parts = lines[i+1].strip().split()
                scale_line_parts[0] = params['scale']
                if params['shape1'] and len(scale_line_parts) > 1:
                    scale_line_parts[1] = params['shape1']
                lines[i+1] = '  ' + ' '.join(scale_line_parts) + '\n'
            
            elif '!       U         V          W' in line and i+1 < len(lines) and params['u'] and params['v'] and params['w']:
                uvw_line = '     ' + params['u'] + ' ' * 4 + params['v'] + ' ' * 5 + params['w'] + ' ' * 5
                if params['x']:
                    uvw_line += params['x'] + ' ' * 5
                else:
                    uvw_line += '0.000000' + ' ' * 5
                
                if params['y']:
                    uvw_line += params['y'] + ' ' * 5
                else:
                    uvw_line += '0.000000' + ' ' * 5
                
                uvw_line += '0.000000' + ' ' * 5 + '0.000000' + ' ' * 7 + '0\n'
                lines[i+1] = uvw_line
            
            elif '!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4' in line and i+1 < len(lines) and params['asy1'] and params['asy2']:
                asy_parts = lines[i+1].split()
                if len(asy_parts) >= 6:
                    asy_parts[2] = params['asy1']
                    asy_parts[3] = params['asy2']
                    if params['asy3']:
                        asy_parts[4] = params['asy3']
                    if params['asy4'] and len(asy_parts) > 5:
                        asy_parts[5] = params['asy4']
                    lines[i+1] = '  ' + '  '.join(asy_parts) + '\n'
            
            # 2Theta plotting range
            elif '!  2Th1/TOF1    2Th2/TOF2' in line and i+1 < len(lines) and params['th1'] and params['th2']:
                th_parts = lines[i+1].split()
                if len(th_parts) >= 3:
                    th_parts[0] = params['th1']
                    th_parts[1] = params['th2']
                    lines[i+1] = '      ' + '     '.join(th_parts) + '\n'
        
        with open(pcr_file_path, 'w') as file:
            file.writelines(lines)
        
        print("Successfully applied parameters to PCR file from tbbaco.pcr")
        return True
    
    except Exception as e:
        print("Error applying parameters to PCR file: {}".format(e))
        return False

def fix_atom_labels_spacing(pcr_file):

    try:
        with open(pcr_file, 'r') as file:
            lines = file.readlines()
        atom_line_indices = []
        for i, line in enumerate(lines):
            if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+', line):
                atom_line_indices.append(i)
        
        for i in atom_line_indices:
            try:
                if '#' in lines[i]:
                    main_part, display_part = lines[i].split('#', 1)
                    parts = main_part.split()
                    display_code = '#' + display_part.strip()
                else:
                    parts = lines[i].split()
                    display_code = ''
                
                if len(parts) >= 7:  # Must have at least name, type, x, y, z, Biso, occ
                    try:
                        name = parts[0]
                        typ = parts[1]  # Preserve original capitalization
                        
                        # Check for invalid values and skip this line if found
                        if parts[2].startswith('<') or parts[3].startswith('<') or parts[4].startswith('<'):
                            print("Warning: Skipping atom line with invalid format: {}".format(lines[i].strip()))
                            continue
                        
                        _ = float(parts[2])  # Validate x is numeric
                        _ = float(parts[3])  # Validate y is numeric  
                        _ = float(parts[4])  # Validate z is numeric
                        _ = float(parts[5])  # Validate biso is numeric
                        _ = float(parts[6])  # Validate occ is numeric
                        
                        x_str, y_str, z_str = parts[2], parts[3], parts[4]
                        biso_str, occ_str = parts[5], parts[6]
                        
                        formatted_line = "{:<2} {:<2}      {}  {}  {}  {}   {}   0   0   2    0".format(
                            name, typ, x_str, y_str, z_str, biso_str, occ_str)
                        
                        if display_code:
                            formatted_line += "  " + display_code
                        
                        lines[i] = formatted_line + "\n"
                        
                    except ValueError as e:
                        print("Warning: Skipping atom line with invalid format: {} - Error: {}".format(lines[i].strip(), str(e)))
            except Exception as e:
                print("Error processing atom line {}: {}".format(i+1, str(e)))
        
        with open(pcr_file, 'w') as file:
            file.writelines(lines)
    
    except Exception as e:
        print("Error in fix_atom_labels_spacing: {}".format(e))

def apply_reference_params_extended(pcr_file_path, root_dir, copy_asymmetry=False, copy_scan_range=False):

    try:
        result = apply_reference_params(pcr_file_path, root_dir)
        if not result:
            return False
            
        if not copy_asymmetry and not copy_scan_range:
            return True
            
        ref_pcr_path = find_reference_pcr_file('tbbaco.pcr', root_dir)
        if not ref_pcr_path:
            print("Warning: Could not find reference PCR file tbbaco.pcr for additional parameters")
            return False
            
        asymmetry_lines = []
        scan_range_lines = []
        
        with open(ref_pcr_path, 'r') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                # Asymmetry parameters
                if '!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4' in line and copy_asymmetry:
                    if i+1 < len(lines) and i+2 < len(lines):
                        asymmetry_lines = [line, lines[i+1], lines[i+2]]
                        print("Found asymmetry parameters in reference PCR")
                
                if '!  2Th1/TOF1    2Th2/TOF2  Pattern to plot' in line and copy_scan_range:
                    if i+1 < len(lines):
                        scan_range_lines = [line, lines[i+1]]
                        print("Found scan range parameters in reference PCR")
        
        if asymmetry_lines or scan_range_lines:
            with open(pcr_file_path, 'r') as file:
                pcr_lines = file.readlines()
                
            if asymmetry_lines and copy_asymmetry:
                for i, line in enumerate(pcr_lines):
                    if '!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4' in line:
                        if i+1 < len(pcr_lines) and i+2 < len(pcr_lines):
                            # Replace the header and the two lines below it
                            pcr_lines[i] = asymmetry_lines[0]
                            pcr_lines[i+1] = asymmetry_lines[1]
                            pcr_lines[i+2] = asymmetry_lines[2]
                            print("Applied asymmetry parameters to PCR file")
                            break
            
            if scan_range_lines and copy_scan_range:
                for i, line in enumerate(pcr_lines):
                    if '!  2Th1/TOF1    2Th2/TOF2  Pattern to plot' in line:
                        if i+1 < len(pcr_lines):
                            # Replace the header and the line below it
                            pcr_lines[i] = scan_range_lines[0]
                            pcr_lines[i+1] = scan_range_lines[1]
                            print("Applied scan range parameters to PCR file")
                            break
            
            with open(pcr_file_path, 'w') as file:
                file.writelines(pcr_lines)
            
            print("Successfully applied additional parameters from reference PCR file")
        
        return True
        
    except Exception as e:
        print("Error applying extended reference PCR parameters: {}".format(str(e)))
        return False

def apply_reference_params(pcr_file_path, root_dir):

    try:
        ref_pcr_path = find_reference_pcr_file('tbbaco.pcr', root_dir)
        if not ref_pcr_path:
            print("Warning: Could not find reference PCR file tbbaco.pcr")
            return False
        
        params = extract_reference_parameters(ref_pcr_path)
        if not params:
            print("Warning: Failed to extract parameters from reference PCR")
            return False
        
        result = apply_parameters_to_pcr(pcr_file_path, params)
        
        fix_atom_labels_spacing(pcr_file_path)
        
        if result:
            print("\nReference PCR parameters applied from tbbaco.pcr:")
            for key in ['comment', 'lambda1', 'lambda2', 'thmin', 'step', 'thmax', 
                       'zero', 'scale', 'u', 'v', 'w', 'space_group', 'npr1', 'npr2']:
                if key in params and params[key]:
                    print("   {}: {}".format(key, params[key]))
            
            print("   Number of atoms from reference: {}".format(len(params['atoms'])))
            if params['atoms']:
                print("   Atom information from reference:")
                for i, atom in enumerate(params['atoms']):
                    display_code = params['display_codes'].get(atom['name'], '')
                    if i < 2:  # Show only first 2 atoms to avoid verbose output
                        print("      {}: ({}, {}, {}) Biso: {}, Occ: {} {}".format(
                            atom['name'], atom['x'], atom['y'], atom['z'], 
                            atom['biso'], atom['occ'], display_code))
                        print("        Biso codeword will be set to 0.00")
                if len(params['atoms']) > 2:
                    print("      ... and {} more atoms".format(len(params['atoms']) - 2))
        
        return result
    
    except Exception as e:
        print("Error in apply_reference_params: {}".format(e))
        return False

if __name__ == "__main__":
    print("This module is designed to be imported, not run directly.")