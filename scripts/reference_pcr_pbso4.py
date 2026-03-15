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
        'job': '1',         # Set job to 1 for refinement
        'lambda1': '',
        'lambda2': '',
        'ratio': '',
        'bkpos': '',
        'wdt': '',
        'cthm': '',
        'mur': '',
        'asylim': '',
        'rpolarz': '',
        'thmin': '',
        'step': '',
        'thmax': '',
        'exact_theta_line': '',  # Store the exact theta line from reference
        'excluded_regions': [],
        'zero': '',
        'zero_code': '0.0',      # Set to 0.0 as per template
        'bg_coeffs': [],
        'bg_codes': [],          # Set all to 0.00 as per template
        'space_group': '',
        'phase_name': 'PbSO4',   # Use correct capitalization
        'scale': '',
        'scale_code': '11.00000',  # Keep the same scale code
        'shape1': '',
        'bov': '',
        'str1': '',
        'str2': '',
        'str3': '',
        'u': '',
        'v': '',
        'w': '',
        'x': '',
        'y': '',
        'gausz': '',
        'lorsz': '',
        'u_code': '0.000',      # Set to 0.000 as per template
        'v_code': '0.000',      # Set to 0.000 as per template
        'w_code': '0.000',      # Set to 0.000 as per template
        'y_code': '0.000',      # Set to 0.000 as per template
        'a': '',
        'b': '',
        'c': '',
        'alpha': '',
        'beta': '',
        'gamma': '',
        'a_code': '0.00000',    # Set to 0.00000 as per template
        'b_code': '0.00000',    # Set to 0.00000 as per template
        'c_code': '0.00000',    # Set to 0.00000 as per template
        'asy1': '',
        'asy2': '',
        'asy3': '',
        'asy4': '',
        'asy1_code': '0.00',    # Set to 0.00 as per template
        'asy2_code': '0.00',    # Set to 0.00 as per template
        'asy3_code': '0.00',    # Set to 0.00 as per template
        'th1': '',
        'th2': '',
        'atoms': [],
        'atom_codes': {},       # Store atom position codes
        'atz': '1213.030',      # Default value if not found
        'display_codes': {},    # For special display directives like #color green, etc.
        'ins_value': '6',       # Use new INS value of 6
        'rmua': '0.000',        # Set to 0.000 as per template
        'rmub': '0.000',        # Set to 0.000 as per template
        'rmuc': '0.000',        # Set to 0.000 as per template
        'chi2': '4.202',        # Updated Chi2 value
        'r_bragg': '0.0000'     # Updated R_Bragg value
    }
    
    try:
        with open(ref_pcr_path, 'r') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                if line.startswith('!Job') and i+1 < len(lines):
                    job_line = lines[i+1].strip()
                    parts = job_line.split()
                    if parts and len(parts) > 0:
                        params['job'] = parts[0]
                
                if line.startswith('COMM'):
                    params['comment'] = line.strip()[4:].strip()
                
                elif '!  Data for PHASE number:' in line and i+2 < len(lines):
                    phase_line = lines[i+2].strip()
                    if phase_line:
                        params['phase_name'] = 'PbSO4'  # Force proper capitalization
                
                elif '! Lambda1  Lambda2' in line and i+1 < len(lines):
                    lambda_line = lines[i+1].strip()
                    parts = lambda_line.split()
                    if len(parts) >= 10:
                        params['lambda1'] = parts[0]
                        params['lambda2'] = parts[1]
                        params['ratio'] = parts[2]
                        params['bkpos'] = parts[3]
                        params['wdt'] = parts[4]
                        params['cthm'] = parts[5]
                        params['mur'] = parts[6]
                        params['asylim'] = parts[7]
                        params['rpolarz'] = parts[8]
                
                elif '!Ipr Ppl Ioc' in line and i+1 < len(lines):
                    ins_line = lines[i+1].strip()
                    parts = ins_line.split()
                    if len(parts) >= 8:
                        params['ins_value'] = parts[7]
                
                elif '!     Thmin       Step       Thmax' in line and i+1 < len(lines):
                    params['exact_theta_line'] = lines[i+1]
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
                    if parts and len(parts) > 1:
                        params['zero'] = parts[0]
                        params['zero_code'] = '0.0'
                
                elif '!   Background coefficients/codes' in line and i+1 < len(lines) and i+2 < len(lines):
                    bg_line = lines[i+1].strip()
                    params['bg_coeffs'] = bg_line.split()
                    params['bg_codes'] = ['0.00' for _ in range(len(params['bg_coeffs']))]
                
                elif '<--Space group symbol' in line:
                    params['space_group'] = line.strip().split('<--')[0].strip()
                
                elif re.search(r'!Nat\s+Dis\s+Ang.*ATZ', line) and i+1 < len(lines):
                    nat_line = lines[i+1].strip()
                    parts = nat_line.split()
                    if len(parts) > 7:
                        if parts[7] != '0' and parts[7] != '0.0' and parts[7] != '0.00':
                            params['atz'] = parts[7]
                
                elif '! Pr1    Pr2    Pr3   Brind.   Rmua   Rmub   Rmuc' in line and i+1 < len(lines):
                    pr_line = lines[i+1].strip()
                    parts = pr_line.split()
                    if len(parts) >= 7:
                        params['rmua'] = '0.000'
                        params['rmub'] = '0.000'
                        params['rmuc'] = '0.000'
                
                elif re.search(r'!Atom\s+Typ\s+X\s+Y\s+Z\s+Biso', line):
                    j = i + 2
                    while j < len(lines) and not re.search(r'!-+>\s+Profile Parameters', lines[j]):
                        atom_match = re.match(r'^([A-Za-z][A-Za-z0-9]*\d*)\s+([A-Za-z]+)\s+', lines[j])
                        if atom_match:
                            parts = lines[j].split()
                            if len(parts) >= 7:
                                atom_name = parts[0]
                                # Convert atom type to all uppercase
                                atom_type = parts[1].upper()
                                
                                atom_info = {
                                    'name': atom_name,
                                    'type': atom_type,
                                    'x': parts[2],
                                    'y': parts[3],
                                    'z': parts[4],
                                    'biso': parts[5],
                                    'occ': parts[6],
                                }
                                
                                if j+1 < len(lines):
                                    code_parts = lines[j+1].strip().split()
                                    if len(code_parts) >= 5:
                                        # Force all codes to 0.00
                                        atom_codes = {
                                            'x_code': '0.00',
                                            'y_code': '0.00',
                                            'z_code': '0.00',
                                            'biso_code': '0.00',
                                            'occ_code': '0.00'
                                        }
                                        params['atom_codes'][atom_name] = atom_codes
                                
                                if '#' in lines[j]:
                                    display_directive = lines[j].split('#')[1].strip()
                                    params['display_codes'][atom_name] = display_directive
                                
                                params['atoms'].append(atom_info)
                                j += 4
                            else:
                                j += 1
                        else:
                            j += 1
                
                elif re.search(r'![\s]*Scale[\s]+.*Strain-Model', line) and i+1 < len(lines) and i+2 < len(lines):
                    scale_line = lines[i+1].strip()
                    scale_parts = scale_line.split()
                    
                    if scale_parts and len(scale_parts) >= 1:
                        params['scale'] = scale_parts[0]
                        if len(scale_parts) >= 2:
                            params['shape1'] = scale_parts[1]
                        if len(scale_parts) >= 3:
                            params['bov'] = scale_parts[2]
                        if len(scale_parts) >= 4:
                            params['str1'] = scale_parts[3]
                        if len(scale_parts) >= 5:
                            params['str2'] = scale_parts[4]
                        if len(scale_parts) >= 6:
                            params['str3'] = scale_parts[5]
                    
                
                elif re.search(r'![\s]*U[\s]+V[\s]+W[\s]+.*Size-Model', line) and i+1 < len(lines) and i+2 < len(lines):
                    uvw_line = lines[i+1].strip()
                    uvw_parts = uvw_line.split()
                    
                    if uvw_parts and len(uvw_parts) >= 7:
                        params['u'] = uvw_parts[0]
                        params['v'] = uvw_parts[1]
                        params['w'] = uvw_parts[2]
                        params['x'] = uvw_parts[3]
                        params['y'] = uvw_parts[4]
                        if len(uvw_parts) >= 6:
                            params['gausz'] = uvw_parts[5]
                        if len(uvw_parts) >= 7:
                            params['lorsz'] = uvw_parts[6]
                    
                    params['u_code'] = '0.000'
                    params['v_code'] = '0.000'
                    params['w_code'] = '0.000'
                    params['x_code'] = '0.000'
                    params['y_code'] = '0.000'
                
                elif '!     a          b         c' in line and i+1 < len(lines) and i+2 < len(lines):
                    lattice_line = lines[i+1].strip()
                    lattice_parts = lattice_line.split()
                    
                    if lattice_parts and len(lattice_parts) >= 6:
                        params['a'] = lattice_parts[0]
                        params['b'] = lattice_parts[1]
                        params['c'] = lattice_parts[2]
                        params['alpha'] = lattice_parts[3]
                        params['beta'] = lattice_parts[4]
                        params['gamma'] = lattice_parts[5]
                    
                    params['a_code'] = '0.00000'
                    params['b_code'] = '0.00000'
                    params['c_code'] = '0.00000'
                    params['alpha_code'] = '0.00000'
                    params['beta_code'] = '0.00000'
                    params['gamma_code'] = '0.00000'
                
                elif re.search(r'![\s]*Pref1[\s]+Pref2[\s]+Asy1', line) and i+1 < len(lines) and i+2 < len(lines):
                    asy_line = lines[i+1].strip()
                    asy_parts = asy_line.split()
                    
                    if asy_parts and len(asy_parts) >= 6:
                        params['asy1'] = asy_parts[2]
                        params['asy2'] = asy_parts[3]
                        params['asy3'] = asy_parts[4]
                        if len(asy_parts) > 5:
                            params['asy4'] = asy_parts[5]
                    
                    params['asy1_code'] = '0.00'
                    params['asy2_code'] = '0.00'
                    params['asy3_code'] = '0.00'
                
                elif '!  2Th1/TOF1    2Th2/TOF2' in line and i+1 < len(lines):
                    th_line = lines[i+1].strip()
                    parts = th_line.split()
                    if len(parts) >= 2:
                        params['th1'] = parts[0]
                        params['th2'] = parts[1]
        
        print("\nParameters extracted from reference PCR (pbso4):")
        print("Comment: {}".format(params['comment']))
        print("Lambda1, Lambda2: {}, {}".format(params['lambda1'], params['lambda2']))
        print("Theta range: {}, {}, {}".format(params['thmin'], params['step'], params['thmax']))
        print("Zero parameter: {}".format(params['zero']))
        print("Scale factor: {}".format(params['scale']))
        print("U, V, W: {}, {}, {}".format(params['u'], params['v'], params['w']))
        print("Space group: {}".format(params['space_group']))
        print("ATZ value: {}".format(params['atz']))
        print("Number of atoms: {}".format(len(params['atoms'])))
        print("INS value: {}".format(params['ins_value']))
        print("Chi2 value: {}".format(params['chi2']))
        print("R_Bragg value: {}".format(params['r_bragg']))
        
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
            if "Current global Chi2" in line:
                lines[i] = "! Current global Chi2 (Bragg contrib.) =      {}  \n".format(params['chi2'])
                break
                
        for i, line in enumerate(lines):
            if line.startswith('!Job') and i+1 < len(lines) and params['job']:
                job_parts = lines[i+1].split()
                if job_parts and len(job_parts) > 0:
                    job_parts[0] = params['job']
                    lines[i+1] = '   ' + '   '.join(job_parts) + '\n'
            
            elif line.startswith('COMM') and params['comment']:
                lines[i] = "COMM   {}\n".format(params['comment'])
            
            elif '!  Data for PHASE number:' in line and i+2 < len(lines) and params['phase_name']:
                lines[i] = lines[i].replace("Current R_Bragg for Pattern#  1:    36.86",
                                           "Current R_Bragg for Pattern#  1:    {}".format(params['r_bragg']))
                lines[i+2] = "{}\n".format(params['phase_name'])
            
            elif '!Ipr Ppl Ioc' in line and i+1 < len(lines) and params['ins_value']:
                ins_parts = lines[i+1].split()
                if len(ins_parts) >= 8:
                    ins_parts[7] = params['ins_value']
                    lines[i+1] = '   ' + '   '.join(ins_parts) + '\n'
            
            elif '! Lambda1  Lambda2' in line and i+1 < len(lines) and params['lambda1'] and params['lambda2']:
                lambda_line = " {} {} {} {} {} {} {} {} {} {}\n".format(
                    params['lambda1'], 
                    params['lambda2'],
                    params['ratio'] if params['ratio'] else "1.00000", 
                    params['bkpos'] if params['bkpos'] else "70.000",
                    params['wdt'] if params['wdt'] else "6.0000",
                    params['cthm'] if params['cthm'] else "0.0000",
                    params['mur'] if params['mur'] else "0.0000",
                    params['asylim'] if params['asylim'] else "160.00",
                    params['rpolarz'] if params['rpolarz'] else "0.0000",
                    params['rpolarz'] if params['rpolarz'] else "0.0000"  # 2nd-muR
                )
                lines[i+1] = lambda_line
            
            elif '!     Thmin       Step       Thmax' in line and i+1 < len(lines):
                if params['exact_theta_line'] and params['exact_theta_line'].strip():
                    lines[i+1] = params['exact_theta_line']
                    print(">>> THETA PARAMETERS UPDATED USING EXACT REFERENCE LINE")
                    print("    New line: {}".format(params['exact_theta_line'].strip()))
                elif params['thmin'] and params['step'] and params['thmax']:
                    lines[i+1] = "     {}   {}   {}   0.000   0.000\n".format(
                        params['thmin'], params['step'], params['thmax'])
                    print(">>> THETA PARAMETERS UPDATED: {} {} {}".format(
                        params['thmin'], params['step'], params['thmax']))
                else:
                    lines[i+1] = "     10.0000   0.050000   155.4500   0.000   0.000\n"
                    print(">>> THETA PARAMETERS SET TO SAFE DEFAULT VALUES")
            
            elif '! Excluded regions' in line and params['excluded_regions']:
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('!'):
                    lines.pop(j)
                
                for region in params['excluded_regions']:
                    lines.insert(j, "        {}      {}    \n".format(region[0], region[1]))
                    j += 1
            
            elif '!  Zero    Code    SyCos' in line and i+1 < len(lines) and params['zero']:
                lines[i+1] = "  {}    {}  0.00000    0.0  0.00000    0.0 0.000000    0.00   0\n".format(
                    params['zero'], params['zero_code'])
            
            elif '!   Background coefficients/codes' in line and i+1 < len(lines) and i+2 < len(lines) and params['bg_coeffs']:
                if len(params['bg_coeffs']) >= 6:
                    lines[i+1] = "      {}      {}      {}      {}      {}      {}\n".format(
                        params['bg_coeffs'][0], params['bg_coeffs'][1], params['bg_coeffs'][2],
                        params['bg_coeffs'][3], params['bg_coeffs'][4], params['bg_coeffs'][5])
                    
                    lines[i+2] = "        0.00        0.00        0.00        0.00        0.00        0.00\n"
            
            elif '<--Space group symbol' in line and params['space_group']:
                lines[i] = params['space_group'] + ' ' * 19 + '<--Space group symbol\n'
            
            elif re.search(r'!Nat\s+Dis\s+Ang.*ATZ', line) and i+1 < len(lines):
                nat_line = lines[i+1].split()
                if len(nat_line) >= 3:
                    nat_count = nat_line[0] if len(nat_line) > 0 else "5"
                    lines[i+1] = "   {}   0   0   0   0   0   0       {}   0   0\n".format(
                        nat_count, params['atz'])
            
            elif '! Pr1    Pr2    Pr3   Brind.   Rmua   Rmub   Rmuc' in line and i+1 < len(lines):
                pr_parts = lines[i+1].split()
                if len(pr_parts) >= 7:
                    pr_parts[4] = params['rmua']
                    pr_parts[5] = params['rmub']
                    pr_parts[6] = params['rmuc']
                    lines[i+1] = "  {}  {}  {}  {}  {}  {}  {}\n".format(
                        pr_parts[0], pr_parts[1], pr_parts[2], pr_parts[3],
                        pr_parts[4], pr_parts[5], pr_parts[6])
            
            elif re.search(r'![\s]*Scale[\s]+.*Strain-Model', line) and i+1 < len(lines) and i+2 < len(lines) and params['scale']:
                lines[i+1] = " {} 0.00000 0.00000 0.00000 0.00000 0.00000 0\n".format(params['scale'])
                
                lines[i+2] = "    {}     0.000     0.000     0.000     0.000     0.000\n".format(params['scale_code'])
                
                print(">>> SCALE PARAMETER UPDATED: {}".format(params['scale']))
            
            elif re.search(r'![\s]*U[\s]+V[\s]+W[\s]+.*Size-Model', line) and i+1 < len(lines) and i+2 < len(lines) and params['u'] and params['v'] and params['w']:
                lines[i+1] = "     {}    {}    {}    {}    {}    {}    {}    0\n".format(
                    params['u'], 
                    params['v'], 
                    params['w'],
                    params['x'] if params['x'] else "0.000000",
                    params['y'] if params['y'] else "0.091739",  # Use value from reference
                    params['gausz'] if params['gausz'] else "0.000000",
                    params['lorsz'] if params['lorsz'] else "0.000000"
                )
                
                lines[i+2] = "      {}      {}      {}      {}      {}      {}      {}\n".format(
                    params['u_code'], params['v_code'], params['w_code'], 
                    params['u_code'], params['v_code'], params['u_code'], params['v_code']
                )
                
                print(">>> U, V, W PARAMETERS UPDATED: {} {} {}".format(
                    params['u'], params['v'], params['w']))
                
            elif '!     a          b         c        alpha      beta       gamma      #Cell Info' in line and i+1 < len(lines) and i+2 < len(lines):
                if params['a'] and params['b'] and params['c']:
                    lines[i+1] = "   {}   {}   {}  {}  {}  {}   \n".format(
                        params['a'], params['b'], params['c'],
                        params['alpha'], params['beta'], params['gamma']
                    )
                    
                    lines[i+2] = "    {}    {}    {}    {}    {}    {}\n".format(
                        params.get('a_code', '0.00000'),
                        params.get('b_code', '0.00000'),
                        params.get('c_code', '0.00000'),
                        params.get('alpha_code', '0.00000'),
                        params.get('beta_code', '0.00000'),
                        params.get('gamma_code', '0.00000')
                    )
            
            elif re.search(r'![\s]*Pref1[\s]+Pref2[\s]+Asy1', line) and i+1 < len(lines) and i+2 < len(lines):
                if 'S_L' not in line and 'D_L' not in line:
                    lines[i] = "!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4      S_L      D_L\n"
                
                if params['asy1'] and params['asy2'] and params['asy3']:
                    lines[i+1] = "  0.00000  0.00000  {}  {} {}  {}  0.00000  0.00000\n".format(
                        params['asy1'], params['asy2'], params['asy3'], params.get('asy4', '0.00000')
                    )
                    
                    lines[i+2] = "     0.00     0.00   {}   {}   {}     0.00     0.00     0.00\n".format(
                        params.get('asy1_code', '0.00'),
                        params.get('asy2_code', '0.00'),
                        params.get('asy3_code', '0.00')
                    )
            
            elif '!  2Th1/TOF1    2Th2/TOF2' in line and i+1 < len(lines) and params['th1'] and params['th2']:
                lines[i+1] = "      {}     {}     1\n".format(params['th1'], params['th2'])
        
        with open(pcr_file_path, 'w') as file:
            file.writelines(lines)
        
        print("Successfully applied parameters to PCR file from pbso4.pcr")
        return True
    
    except Exception as e:
        print("Error applying parameters to PCR file: {}".format(e))
        return False

def fix_atom_labels_spacing(pcr_file, atom_codes=None, display_codes=None):

    if atom_codes is None:
        atom_codes = {}
    if display_codes is None:
        display_codes = {}
        
    try:
        with open(pcr_file, 'r') as file:
            lines = file.readlines()

        atom_section_start = -1
        atom_section_end = -1
        
        for i, line in enumerate(lines):
            if re.search(r'!Atom\s+Typ\s+X\s+Y\s+Z\s+Biso', line):
                atom_section_start = i + 2  # Skip the header and beta11 line
            if atom_section_start > 0 and re.search(r'!-+>\s+Profile Parameters', line):
                atom_section_end = i
                break
        
        if atom_section_start == -1 or atom_section_end == -1:
            print("Warning: Could not locate atom section in PCR file")
            return
        
        for i in range(atom_section_start, atom_section_end, 4):  # Process in groups of 4 lines
            atom_match = re.match(r'^([A-Za-z][A-Za-z0-9]*\d*)\s+([A-Za-z]+)\s+', lines[i])
            if atom_match:
                try:
                    parts = lines[i].split()
                    if len(parts) >= 7:  # Must have at least name, type, x, y, z, Biso, occ
                        atom_name = parts[0]
                        
                        display_code = ""
                        if atom_name in display_codes:
                            display_code = "#" + display_codes[atom_name]
                        elif '#' in lines[i]:
                            main_part, display_part = lines[i].split('#', 1)
                            display_code = "#" + display_part.strip()
                        
                        try:
                            name = parts[0]
                            typ = parts[1].upper()
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])
                            biso = float(parts[5])
                            occ = float(parts[6])
                            
                            if len(name) == 2:  # Two character element (Pb, Si, etc.)
                                atom_format = "{} {}      {:.5f}  {:.5f}  {:.5f}  {:.5f}   {:.5f}   0   0   2    0"
                            else:  # One character element with number (O1, O2, etc.)
                                atom_format = "{} {}       {:.5f}  {:.5f}  {:.5f}  {:.5f}   {:.5f}   0   0   2    0"
                            
                            formatted_line = atom_format.format(name, typ, x, y, z, biso, occ)
                            
                            if display_code:
                                formatted_line += "  " + display_code
                            
                            lines[i] = formatted_line + "\n"
                            
                            atom_code = atom_codes.get(name, {})
                            
                            if atom_code and i+1 < len(lines) and i+3 < len(lines):
                                x_code = atom_code.get('x_code', '0.00')
                                y_code = atom_code.get('y_code', '0.00')
                                z_code = atom_code.get('z_code', '0.00')
                                biso_code = atom_code.get('biso_code', '0.00')
                                occ_code = atom_code.get('occ_code', '0.00')
                                
                                lines[i+1] = "                  {}     {}     {}     {}      {}\n".format(
                                    x_code, y_code, z_code, biso_code, occ_code
                                )
                                
                                lines[i+2] = "      0.00000  0.00000  0.00000  0.00000  0.00000   0.00000\n"
                                lines[i+3] = "         0.00     0.00     0.00     0.00     0.00      0.00\n"
                            else:
                                lines[i+1] = "                  0.00     0.00     0.00     0.00      0.00\n"
                                lines[i+2] = "      0.00000  0.00000  0.00000  0.00000  0.00000   0.00000\n"
                                lines[i+3] = "         0.00     0.00     0.00     0.00     0.00      0.00\n"
                            
                        except ValueError as e:
                            print("Warning: Could not parse atom coordinates in line {}: {}".format(i+1, str(e)))
                except Exception as e:
                    print("Error processing atom line {}: {}".format(i+1, str(e)))
        
        with open(pcr_file, 'w') as file:
            file.writelines(lines)
    
    except Exception as e:
        print("Error in fix_atom_labels_spacing: {}".format(e))

def apply_reference_params(pcr_file_path, root_dir):

    try:
        ref_pcr_path = find_reference_pcr_file('pbso4.pcr', root_dir)
        if not ref_pcr_path:
            print("Warning: Could not find reference PCR file pbso4.pcr")
            return False
        
        params = extract_reference_parameters(ref_pcr_path)
        if not params:
            print("Warning: Failed to extract parameters from reference PCR")
            return False
        
        result = apply_parameters_to_pcr(pcr_file_path, params)
        
        fix_atom_labels_spacing(pcr_file_path, params.get('atom_codes', {}), params.get('display_codes', {}))
        
        if result:
            print("\nReference PCR parameters applied from pbso4.pcr:")
            for key in ['comment', 'lambda1', 'lambda2', 'thmin', 'step', 'thmax', 
                       'zero', 'scale', 'u', 'v', 'w', 'space_group', 'atz', 'ins_value',
                       'chi2', 'r_bragg']:
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
                if len(params['atoms']) > 2:
                    print("      ... and {} more atoms".format(len(params['atoms']) - 2))
        
        return result
    
    except Exception as e:
        print("Error in apply_reference_params: {}".format(e))
        return False

if __name__ == "__main__":
    print("This module is designed to be imported, not run directly.")