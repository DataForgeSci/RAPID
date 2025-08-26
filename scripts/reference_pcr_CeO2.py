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
        'cthm': '',
        'mur': '',
        'asylim': '',
        'rpolarz': '',
        'thmin': '',
        'step': '',
        'thmax': '',
        'excluded_regions': [],
        'zero': '',
        'zero_code': '',
        'bg_coeffs': [],
        'bg_codes': [],
        'space_group': '',
        'phase_name': '',
        'scale': '',
        'scale_code': '',
        'shape1': '',
        'bov': '',
        'u': '',
        'v': '',
        'w': '',
        'x': '',
        'y': '',
        'u_code': '',
        'v_code': '',
        'w_code': '',
        'x_code': '',
        'y_code': '',
        'asy1': '',
        'asy2': '',
        'asy3': '',
        'asy4': '',
        'th1': '',
        'th2': '',
        'a': '',
        'b': '',
        'c': '',
        'alpha': '',
        'beta': '',
        'gamma': '',
        'a_code': '',
        'b_code': '',
        'c_code': '',
        'alpha_code': '',
        'beta_code': '',
        'gamma_code': '',
        'atoms': [],
        'atom_codes': {},
        'atz': '',
        'display_codes': {},  # For special display directives
        'chi2': '',
        'r_bragg': '',
        'ncy': '',
        'rpa': '',
        'num_refined': ''
    }
    
    try:
        with open(ref_pcr_path, 'r') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                # Chi2 value
                if "Current global Chi2" in line:
                    chi2_match = re.search(r"=\s+([\d.]+)", line)
                    if chi2_match:
                        params['chi2'] = chi2_match.group(1).strip()
                
                elif line.startswith('COMM'):
                    params['comment'] = line.strip()[4:].strip()
                
                elif '!  Data for PHASE number:' in line and i+2 < len(lines):
                    # Extract R_Bragg value from this line
                    r_bragg_match = re.search(r"Current R_Bragg for Pattern#\s+\d+:\s+([\d.]+)", line)
                    if r_bragg_match:
                        params['r_bragg'] = r_bragg_match.group(1).strip()
                    
                    params['phase_name'] = lines[i+2].strip()
                
                elif '!Number of refined parameters' in line and i-1 >= 0:
                    params['num_refined'] = lines[i-1].strip()
                
                elif '!Mat Pcr NLI Rpa Sym Sho' in line and i+1 < len(lines):
                    rpa_line = lines[i+1].strip().split()
                    if len(rpa_line) >= 4:
                        params['rpa'] = rpa_line[3]
                
                elif '!NCY  Eps  R_at  R_an  R_pr  R_gl' in line and i+1 < len(lines):
                    ncy_line = lines[i+1].strip().split()
                    if ncy_line:
                        params['ncy'] = ncy_line[0]
                
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
                    if len(parts) >= 2:
                        params['zero'] = parts[0]
                        params['zero_code'] = parts[1]
                
                elif '!   Background coefficients/codes' in line and i+1 < len(lines) and i+2 < len(lines):
                    bg_line = lines[i+1].strip()
                    bg_code_line = lines[i+2].strip()
                    params['bg_coeffs'] = bg_line.split()
                    params['bg_codes'] = bg_code_line.split()
                
                elif '<--Space group symbol' in line:
                    params['space_group'] = line.strip().split('<--')[0].strip()
                
                elif re.search(r'!Nat\s+Dis\s+Ang.*ATZ', line) and i+1 < len(lines):
                    nat_line = lines[i+1].strip()
                    parts = nat_line.split()
                    if len(parts) > 7:
                        params['atz'] = parts[7]
                
                elif re.search(r'!Atom\s+Typ\s+X\s+Y\s+Z\s+Biso', line):
                    j = i + 1
                    while j < len(lines) and not re.search(r'!-+>\s+Profile Parameters', lines[j]):
                        # Match atom line pattern
                        atom_match = re.match(r'^([A-Za-z0-9]+)\s+([A-Za-z]+)\s+', lines[j])
                        if atom_match:
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
                                
                                if j+1 < len(lines):
                                    code_line = lines[j+1].strip()
                                    code_parts = code_line.split()
                                    if len(code_parts) >= 5:  # Expecting at least 5 code values
                                        atom_code = {
                                            'x_code': code_parts[0],
                                            'y_code': code_parts[1],
                                            'z_code': code_parts[2],
                                            'biso_code': code_parts[3],
                                            'occ_code': code_parts[4]
                                        }
                                        params['atom_codes'][parts[0]] = atom_code
                                
                                params['atoms'].append(atom_info)
                            j += 4
                        else:
                            j += 1
                
                elif re.search(r'![\s]*Scale.*Strain-Model', line) and i+1 < len(lines) and i+2 < len(lines):
                    scale_line = lines[i+1].strip()
                    code_line = lines[i+2].strip()
                    
                    scale_parts = scale_line.split()
                    code_parts = code_line.split()
                    
                    if scale_parts:
                        params['scale'] = scale_parts[0]
                        if len(scale_parts) > 1:
                            params['shape1'] = scale_parts[1]
                        if len(scale_parts) > 2:
                            params['bov'] = scale_parts[2]
                    
                    if code_parts:
                        params['scale_code'] = code_parts[0]
                
                elif re.search(r'![\s]*U[\s]+.*Size-Model', line) and i+1 < len(lines) and i+2 < len(lines):
                    uvw_line = lines[i+1].strip()
                    uvw_code_line = lines[i+2].strip()
                    
                    uvw_parts = uvw_line.split()
                    code_parts = uvw_code_line.split()
                    
                    if len(uvw_parts) >= 7:
                        params['u'] = uvw_parts[0]
                        params['v'] = uvw_parts[1]
                        params['w'] = uvw_parts[2]
                        params['x'] = uvw_parts[3]
                        params['y'] = uvw_parts[4]
                    
                    if len(code_parts) >= 7:
                        params['u_code'] = code_parts[0]
                        params['v_code'] = code_parts[1]
                        params['w_code'] = code_parts[2]
                        params['x_code'] = code_parts[3]
                        params['y_code'] = code_parts[4]
                
                elif '!     a          b         c        alpha      beta       gamma      #Cell Info' in line and i+1 < len(lines) and i+2 < len(lines):
                    lattice_line = lines[i+1].strip()
                    lattice_code_line = lines[i+2].strip()
                    
                    lattice_parts = lattice_line.split()
                    code_parts = lattice_code_line.split()
                    
                    if len(lattice_parts) >= 6:
                        params['a'] = lattice_parts[0]
                        params['b'] = lattice_parts[1]
                        params['c'] = lattice_parts[2]
                        params['alpha'] = lattice_parts[3]
                        params['beta'] = lattice_parts[4]
                        params['gamma'] = lattice_parts[5]
                    
                    if len(code_parts) >= 6:
                        params['a_code'] = code_parts[0]
                        params['b_code'] = code_parts[1]
                        params['c_code'] = code_parts[2]
                        params['alpha_code'] = code_parts[3]
                        params['beta_code'] = code_parts[4]
                        params['gamma_code'] = code_parts[5]
                
                elif re.search(r'![\s]*Pref1[\s]+Pref2[\s]+Asy1', line) and i+1 < len(lines):
                    asy_line = lines[i+1].strip()
                    asy_parts = asy_line.split()
                    
                    if len(asy_parts) >= 6:
                        params['asy1'] = asy_parts[2]
                        params['asy2'] = asy_parts[3]
                        params['asy3'] = asy_parts[4]
                        if len(asy_parts) > 5:
                            params['asy4'] = asy_parts[5]
                
                elif '!  2Th1/TOF1    2Th2/TOF2' in line and i+1 < len(lines):
                    th_line = lines[i+1].strip()
                    parts = th_line.split()
                    if len(parts) >= 2:
                        params['th1'] = parts[0]
                        params['th2'] = parts[1]
        
        print("\nParameters extracted from reference PCR (CeO2):")
        print("Comment: {}".format(params['comment']))
        print("Chi2: {}".format(params['chi2']))
        print("R_Bragg: {}".format(params['r_bragg']))
        print("Number of refined parameters: {}".format(params['num_refined']))
        print("NCY value: {}".format(params['ncy']))
        print("RPa value: {}".format(params['rpa']))
        print("Lambda1, Lambda2: {}, {}".format(params['lambda1'], params['lambda2']))
        print("Theta range: {}, {}, {}".format(params['thmin'], params['step'], params['thmax']))
        print("Zero parameter: {} (code: {})".format(params['zero'], params['zero_code']))
        print("Scale factor: {} (code: {})".format(params['scale'], params['scale_code']))
        print("U, V, W: {}, {}, {} (codes: {}, {}, {})".format(
            params['u'], params['v'], params['w'], 
            params['u_code'], params['v_code'], params['w_code']))
        print("Lattice parameters: {}, {}, {} (codes: {}, {}, {})".format(
            params['a'], params['b'], params['c'],
            params['a_code'], params['b_code'], params['c_code']))
        print("Space group: {}".format(params['space_group']))
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
            if "Current global Chi2" in line and params['chi2']:
                lines[i] = "! Current global Chi2 (Bragg contrib.) =      {}    \n".format(params['chi2'])
                break
                
        for i, line in enumerate(lines):
            if '!Number of refined parameters' in line:
                if re.search(r'^\s+!\s+!Number', line):
                    lines[i] = re.sub(r'^\s+!\s+', '      {}    '.format(params['num_refined']), line)
            
            elif '!Mat Pcr NLI Rpa Sym Sho' in line and i+1 < len(lines) and params['rpa']:
                mat_parts = lines[i+1].split()
                if len(mat_parts) >= 4:
                    mat_parts[3] = params['rpa']
                    lines[i+1] = '   ' + '   '.join(mat_parts) + '\n'
            
            elif '!NCY  Eps  R_at  R_an  R_pr  R_gl' in line and i+1 < len(lines) and params['ncy']:
                ncy_parts = lines[i+1].split()
                if ncy_parts:
                    ncy_parts[0] = params['ncy']
                    lines[i+1] = ' ' + '  '.join(ncy_parts) + '\n'
            
            elif line.startswith('COMM') and params['comment']:
                lines[i] = "COMM   {}\n".format(params['comment'])
            
            elif '!  Data for PHASE number:' in line and params['r_bragg']:
                r_bragg_pattern = r"(Current R_Bragg for Pattern#\s+\d+:)\s+[\d.]+"
                updated_line = re.sub(r_bragg_pattern, r"\1    {}".format(params['r_bragg']), line)
                lines[i] = updated_line
            
            elif '!  Data for PHASE number:' in line and i+2 < len(lines) and params['phase_name']:
                lines[i+2] = "{}\n".format(params['phase_name'])
            
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
                    if params['cthm']:
                        lambda_parts[5] = params['cthm']
                    if params['mur']:
                        lambda_parts[6] = params['mur']
                    if params['asylim']:
                        lambda_parts[7] = params['asylim']
                    if params['rpolarz']:
                        lambda_parts[8] = params['rpolarz']
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
                    if params['zero_code']:
                        zero_parts[1] = params['zero_code']
                    lines[i+1] = '  ' + '  '.join(zero_parts) + '\n'
            
            elif '!   Background coefficients/codes' in line and i+1 < len(lines) and i+2 < len(lines) and params['bg_coeffs']:
                if len(params['bg_coeffs']) >= 6:
                    lines[i+1] = "      " + "      ".join(params['bg_coeffs']) + "\n"
                    
                    if params['bg_codes'] and len(params['bg_codes']) >= 6:
                        lines[i+2] = "        " + "        ".join(params['bg_codes']) + "\n"
            
            elif '<--Space group symbol' in line and params['space_group']:
                lines[i] = params['space_group'] + ' ' * 19 + '<--Space group symbol\n'
            
            elif re.search(r'!Nat\s+Dis\s+Ang.*ATZ', line) and i+1 < len(lines) and params['atz']:
                nat_parts = lines[i+1].split()
                if len(nat_parts) >= 8:
                    nat_parts[7] = params['atz']
                    lines[i+1] = '   ' + '   '.join(nat_parts) + '\n'
            
            elif re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+[-+]?\d*\.\d+', line):
                parts = line.split()
                if len(parts) >= 7:
                    atom_name = parts[0]
                    if atom_name in params['atom_codes'] and i+1 < len(lines):
                        code_line = lines[i+1].strip()
                        code_parts = code_line.split()
                        if len(code_parts) >= 5:
                            code_parts[3] = params['atom_codes'][atom_name]['biso_code']
                            lines[i+1] = "                  " + "     ".join(code_parts) + "\n"
            
            elif re.search(r'![\s]*Scale.*Strain-Model', line) and i+1 < len(lines) and i+2 < len(lines) and params['scale']:
                scale_parts = lines[i+1].split()
                code_parts = lines[i+2].split()
                
                if scale_parts:
                    scale_parts[0] = params['scale']
                    if params['shape1'] and len(scale_parts) > 1:
                        scale_parts[1] = params['shape1']
                    if params['bov'] and len(scale_parts) > 2:
                        scale_parts[2] = params['bov']
                    lines[i+1] = ' ' + ' '.join(scale_parts) + '\n'
                
                if code_parts and params['scale_code']:
                    code_parts[0] = params['scale_code']
                    lines[i+2] = '      ' + '     '.join(code_parts) + '\n'
            
            elif '!-------> Profile Parameters for Pattern #' in line and not '----> Phase #' in line:
                lines[i] = "!-------> Profile Parameters for Pattern #   1  ----> Phase #   1\n"
            
            elif re.search(r'![\s]*U[\s]+.*Size-Model', line) and i+1 < len(lines) and i+2 < len(lines) and params['u'] and params['v'] and params['w']:
                uvw_parts = lines[i+1].split()
                code_parts = lines[i+2].split()
                
                if len(uvw_parts) >= 7:
                    uvw_parts[0] = params['u']
                    uvw_parts[1] = params['v']
                    uvw_parts[2] = params['w']
                    if params['x']:
                        uvw_parts[3] = params['x']
                    if params['y']:
                        uvw_parts[4] = params['y']
                    lines[i+1] = '     ' + '    '.join(uvw_parts) + '\n'
                
                if len(code_parts) >= 4 and params['u_code'] and params['v_code'] and params['w_code']:
                    code_parts[0] = params['u_code']
                    code_parts[1] = params['v_code']
                    code_parts[2] = params['w_code']
                    if params['x_code']:
                        code_parts[3] = params['x_code']
                    lines[i+2] = '      ' + '        '.join(code_parts) + '\n'
            
            elif '!     a          b         c        alpha      beta       gamma      #Cell Info' in line and i+1 < len(lines) and i+2 < len(lines):
                lattice_parts = lines[i+1].split()
                code_parts = lines[i+2].split()
                
                if len(lattice_parts) >= 6 and params['a'] and params['b'] and params['c']:
                    lattice_parts[0] = params['a']
                    lattice_parts[1] = params['b']
                    lattice_parts[2] = params['c']
                    lattice_parts[3] = params['alpha']
                    lattice_parts[4] = params['beta']
                    lattice_parts[5] = params['gamma']
                    lines[i+1] = '   ' + '   '.join(lattice_parts) + '\n'
                
                if len(code_parts) >= 6 and params['a_code'] and params['b_code'] and params['c_code']:
                    code_parts[0] = params['a_code']
                    code_parts[1] = params['b_code']
                    code_parts[2] = params['c_code']
                    code_parts[3] = params['alpha_code']
                    code_parts[4] = params['beta_code']
                    code_parts[5] = params['gamma_code']
                    lines[i+2] = '    ' + '    '.join(code_parts) + '\n'
            
            elif re.search(r'![\s]*Pref1[\s]+Pref2[\s]+Asy1', line) and i+1 < len(lines) and params['asy1'] and params['asy2']:
                asy_parts = lines[i+1].split()
                if len(asy_parts) >= 6:
                    asy_parts[2] = params['asy1']
                    asy_parts[3] = params['asy2']
                    if params['asy3']:
                        asy_parts[4] = params['asy3']
                    if params['asy4'] and len(asy_parts) > 5:
                        asy_parts[5] = params['asy4']
                    lines[i+1] = '  ' + '  '.join(asy_parts) + '\n'
            
            elif '!  2Th1/TOF1    2Th2/TOF2' in line and i+1 < len(lines) and params['th1'] and params['th2']:
                th_parts = lines[i+1].split()
                if len(th_parts) >= 3:
                    th_parts[0] = params['th1']
                    th_parts[1] = params['th2']
                    lines[i+1] = '      ' + '     '.join(th_parts) + '\n'
        
        with open(pcr_file_path, 'w') as file:
            file.writelines(lines)
        
        print("Successfully applied parameters to PCR file from CeO2.pcr")
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
            if re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+[-+]?\d*\.\d+', line):
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
                        
                        formatted_line = "{:<4} {:<4}      {:.5f}  {:.5f}  {:.5f}  {:.5f}   {:.5f}   0   0   0    0  \n".format(
                            name, typ, x, y, z, biso, occ)
                        
                        lines[i] = formatted_line
                    except ValueError as e:
                        print("Warning: Skipping atom line with invalid coordinates: {}".format(lines[i].strip()))
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
            
        ref_pcr_path = find_reference_pcr_file('CeO2.pcr', root_dir)
        if not ref_pcr_path:
            print("Warning: Could not find reference PCR file CeO2.pcr for additional parameters")
            return False
            
        asymmetry_lines = []
        scan_range_lines = []
        
        with open(ref_pcr_path, 'r') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                if '!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4' in line and copy_asymmetry:
                    if i+1 < len(lines) and i+2 < len(lines):
                        asymmetry_lines = [line, lines[i+1], lines[i+2]]
                        print("Found asymmetry parameters in reference PCR")
                
                if '!  2Th1/TOF1    2Th2/TOF2' in line and copy_scan_range:
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
                            pcr_lines[i] = asymmetry_lines[0]
                            pcr_lines[i+1] = asymmetry_lines[1]
                            pcr_lines[i+2] = asymmetry_lines[2]
                            print("Applied asymmetry parameters to PCR file")
                            break
            
            if scan_range_lines and copy_scan_range:
                for i, line in enumerate(pcr_lines):
                    if '!  2Th1/TOF1    2Th2/TOF2' in line:
                        if i+1 < len(pcr_lines):
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
        ref_pcr_path = find_reference_pcr_file('CeO2.pcr', root_dir)
        if not ref_pcr_path:
            print("Warning: Could not find reference PCR file CeO2.pcr")
            return False
        
        params = extract_reference_parameters(ref_pcr_path)
        if not params:
            print("Warning: Failed to extract parameters from reference PCR")
            return False
        
        result = apply_parameters_to_pcr(pcr_file_path, params)
        
        fix_atom_labels_spacing(pcr_file_path)
        
        if result:
            print("\nReference PCR parameters applied from CeO2.pcr:")
            for key in ['comment', 'lambda1', 'lambda2', 'thmin', 'step', 'thmax', 
                       'zero', 'scale', 'u', 'v', 'w', 'space_group', 'chi2', 
                       'num_refined', 'ncy', 'rpa', 'r_bragg']:
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