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

def find_cif_file(cif_name, root_dir):
    cif_path = os.path.join(root_dir, 'dat_vestacif_files', cif_name)
    if os.path.exists(cif_path):
        print("Found CIF file: {}".format(cif_path))
        return cif_path

    cif_path = os.path.join(os.getcwd(), cif_name)
    if os.path.exists(cif_path):
        print("Found CIF file in current directory: {}".format(cif_path))
        return cif_path

    print("Warning: CIF file '{}' not found".format(cif_name))
    return None

def parse_cif_file(cif_path):
    if not os.path.exists(cif_path):
        print("Error: CIF file does not exist: {}".format(cif_path))
        return None

    cif_data = {
        'a': None,
        'b': None,
        'c': None,
        'alpha': 90.0,
        'beta': 90.0,
        'gamma': 90.0,
        'space_group': '',
        'atoms': []
    }

    try:
        with open(cif_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        a_match = re.search(r'_cell_length_a\s+([\d.]+)', content)
        b_match = re.search(r'_cell_length_b\s+([\d.]+)', content)
        c_match = re.search(r'_cell_length_c\s+([\d.]+)', content)
        alpha_match = re.search(r'_cell_angle_alpha\s+([\d.]+)', content)
        beta_match = re.search(r'_cell_angle_beta\s+([\d.]+)', content)
        gamma_match = re.search(r'_cell_angle_gamma\s+([\d.]+)', content)

        if a_match:
            cif_data['a'] = float(a_match.group(1))
        if b_match:
            cif_data['b'] = float(b_match.group(1))
        if c_match:
            cif_data['c'] = float(c_match.group(1))
        if alpha_match:
            cif_data['alpha'] = float(alpha_match.group(1))
        if beta_match:
            cif_data['beta'] = float(beta_match.group(1))
        if gamma_match:
            cif_data['gamma'] = float(gamma_match.group(1))

        sg_match = re.search(r"_space_group_name_H-M_alt\s+'([^']+)'", content)
        if sg_match:
            cif_data['space_group'] = sg_match.group(1).strip()

        in_atom_loop = False
        atom_columns = []

        for i, line in enumerate(lines):
            line = line.strip()

            if '_atom_site_label' in line:
                in_atom_loop = True
                atom_columns = []
                j = i
                while j < len(lines) and lines[j].strip().startswith('_atom_site'):
                    col_name = lines[j].strip()
                    atom_columns.append(col_name)
                    j += 1
                continue

            if in_atom_loop and line and not line.startswith('_') and not line.startswith('loop'):
                # Check if this is an atom data line (starts with element label)
                if re.match(r'^[A-Za-z]+\d*\s+', line):
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            label = parts[0]
                            occ = float(re.sub(r'\([^)]*\)', '', parts[1]))
                            x = float(re.sub(r'\([^)]*\)', '', parts[2]))
                            y = float(re.sub(r'\([^)]*\)', '', parts[3]))
                            z = float(re.sub(r'\([^)]*\)', '', parts[4]))

                            element_type = parts[-1] if len(parts) > 7 else label.rstrip('0123456789')

                            uiso = 0.0
                            if len(parts) > 6:
                                try:
                                    uiso = float(re.sub(r'\([^)]*\)', '', parts[6]))
                                except:
                                    pass

                            atom = {
                                'name': label,
                                'type': element_type,
                                'x': x,
                                'y': y,
                                'z': z,
                                'occ': occ,
                                'uiso': uiso
                            }
                            cif_data['atoms'].append(atom)
                        except (ValueError, IndexError) as e:
                            print("Warning: Could not parse atom line: {} - {}".format(line, str(e)))
                elif line.startswith('#') or line.startswith('loop') or line.startswith('data'):
                    in_atom_loop = False

        print("\nParsed CIF file:")
        print("  Lattice: a={:.6f}, b={:.6f}, c={:.6f}".format(
            cif_data['a'], cif_data['b'], cif_data['c']))
        print("  Angles: alpha={:.2f}, beta={:.4f}, gamma={:.2f}".format(
            cif_data['alpha'], cif_data['beta'], cif_data['gamma']))
        print("  Space group: {}".format(cif_data['space_group']))
        print("  Number of atoms: {}".format(len(cif_data['atoms'])))

        return cif_data

    except Exception as e:
        print("Error parsing CIF file: {}".format(e))
        import traceback
        traceback.print_exc()
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
        'thmin': '',
        'step': '',
        'thmax': '',
        'excluded_regions': [],
        'zero': '',
        'bg_coeffs': [],
        'scale': '',
        'shape1': '',
        'bov': '',
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
        'npr1': '',
        'npr2': ''
    }

    try:
        with open(ref_pcr_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if line.startswith('COMM'):
                params['comment'] = line.strip()[4:].strip()

            elif '! Lambda1  Lambda2' in line and i+1 < len(lines):
                lambda_line = lines[i+1].strip()
                parts = lambda_line.split()
                if len(parts) >= 6:
                    params['lambda1'] = parts[0]
                    params['lambda2'] = parts[1]
                    params['ratio'] = parts[2]
                    params['bkpos'] = parts[3]
                    params['wdt'] = parts[4]
                    if len(parts) >= 6:
                        params['cthm'] = parts[5]

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
                        try:
                            float(line_parts[0])
                            float(line_parts[1])
                            params['excluded_regions'].append((line_parts[0], line_parts[1]))
                        except:
                            pass
                    j += 1

            elif '!  Zero    Code    SyCos' in line and i+1 < len(lines):
                zero_line = lines[i+1].strip()
                parts = zero_line.split()
                if parts:
                    params['zero'] = parts[0]

            elif '!   Background coefficients/codes' in line and i+1 < len(lines):
                bg_line = lines[i+1].strip()
                params['bg_coeffs'] = bg_line.split()

            elif '!  Scale          Shape1' in line and i+1 < len(lines):
                scale_line = lines[i+1].strip()
                parts = scale_line.split()
                if len(parts) >= 3:
                    params['scale'] = parts[0]
                    params['shape1'] = parts[1]
                    params['bov'] = parts[2]

            elif '!       U            V            W' in line and i+1 < len(lines):
                uvw_line = lines[i+1].strip()
                parts = uvw_line.split()
                if len(parts) >= 5:
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
                    params['asy4'] = parts[5]

            elif '!  2Th1/TOF1    2Th2/TOF2' in line and i+1 < len(lines):
                th_line = lines[i+1].strip()
                parts = th_line.split()
                if len(parts) >= 2:
                    params['th1'] = parts[0]
                    params['th2'] = parts[1]

        print("\nExtracted reference parameters from Ba.pcr:")
        print("  Lambda: {}, {}".format(params['lambda1'], params['lambda2']))
        print("  Theta range: {} to {} (step {})".format(params['thmin'], params['thmax'], params['step']))
        print("  Zero: {}".format(params['zero']))
        print("  Scale: {}".format(params['scale']))
        print("  U, V, W: {}, {}, {}".format(params['u'], params['v'], params['w']))

        return params

    except Exception as e:
        print("Error extracting parameters: {}".format(e))
        return None

def create_pcr_from_cif(output_pcr_path, cif_data, ref_params, dat_filename='Ba.dat'):
    if not cif_data or not ref_params:
        print("Error: Missing CIF data or reference parameters")
        return False

    content = []

    # Header
    content.append("COMM   {}\n".format(os.path.basename(output_pcr_path)))
    content.append("! Current global Chi2 (Bragg contrib.) =      1.000\n")
    content.append("NPATT      1       1 <- Flags for patterns (1:refined, 0: excluded)\n")
    content.append("W_PAT   1.000\n")
    content.append("!Nph Dum Ias Nre Cry Opt Aut\n")
    content.append("   1   0   0   0   0   0   1\n")
    content.append("!Job Npr Nba Nex Nsc Nor Iwg Ilo Res Ste Uni Cor Anm Int\n")
    content.append("   0   5   0   2   0   1   0   0   0   0   0   0   0   0  !-> Patt#: 1\n")
    content.append("!\n")
    content.append("!File names of data(patterns) files\n")
    content.append("{}\n".format(dat_filename))
    content.append("!\n")
    content.append("!Mat Pcr NLI Rpa Sym Sho\n")
    content.append("   0   1   0  -1   0   0\n")
    content.append("!Ipr Ppl Ioc Ls1 Ls2 Ls3 Prf Ins Hkl Fou Ana\n")
    content.append("   0   0   1   0   4   0   3  10   0   0   0  !-> Patt#: 1\n")
    content.append("!\n")

    content.append("! Lambda1  Lambda2    Ratio    Bkpos    Wdt    Cthm     muR   AsyLim   Rpolarz  2nd-muR -> Patt# 1\n")
    content.append(" {} {}  {}   {} {}  {}  0.0000   60.00    0.0000  0.0000\n".format(
        ref_params.get('lambda1', '1.540560'),
        ref_params.get('lambda2', '1.540560'),
        ref_params.get('ratio', '0.00000'),
        ref_params.get('bkpos', '25.000'),
        ref_params.get('wdt', '15.0000'),
        ref_params.get('cthm', '0.9100')
    ))
    content.append("!\n")

    content.append("!NCY  Eps  R_at  R_an  R_pr  R_gl\n")
    content.append(" 10  0.10  1.00  1.00  1.00  1.00\n")

    content.append("!     Thmin       Step       Thmax    PSD    Sent0  -> Patt#: 1\n")
    content.append("     {}   {}   {}   0.000   0.000\n".format(
        ref_params.get('thmin', '5.1200'),
        ref_params.get('step', '0.020006'),
        ref_params.get('thmax', '119.9800')
    ))
    content.append("!\n")

    content.append("! Excluded regions (LowT  HighT) for Pattern#  1\n")
    for region in ref_params.get('excluded_regions', [('1.00', '2.00'), ('120.00', '160.00')]):
        content.append("        {}        {}\n".format(region[0], region[1]))
    content.append("! \n")
    content.append("!\n")

    num_atoms = len(cif_data['atoms'])
    num_params = 6 + 1 + 4 + 1 + 1 + 1 + 3 + 2 + num_atoms * 3
    content.append("      {}    !Number of refined parameters\n".format(num_params))
    content.append("!\n")

    content.append("!  Zero    Code    SyCos    Code   SySin    Code  Lambda     Code MORE ->Patt# 1\n")
    content.append("  {}   0.0  0.00000    0.0  0.00000    0.0 0.000000    0.00   0\n".format(
        ref_params.get('zero', '0.00802')
    ))

    content.append("!   Background coefficients/codes  for Pattern#  1  (Polynomial of 6th degree)\n")
    bg_coeffs = ref_params.get('bg_coeffs', ['1730.568', '-2032.394', '1746.730', '-820.944', '189.804', '-17.223'])
    content.append("    {}   {}    {}    {}     {}     {}\n".format(*bg_coeffs[:6]))
    content.append("        0.00        0.00        0.00        0.00        0.00        0.00\n")

    content.append("!-------------------------------------------------------------------------------\n")
    content.append("!  Data for PHASE number:   1  ==> Current R_Bragg for Pattern#  1:   0.0000\n")
    content.append("!-------------------------------------------------------------------------------\n")
    content.append("Ba\n")
    content.append("!\n")

    content.append("!Nat Dis Ang Jbt Isy Str Furth        ATZ     Nvk More\n")
    content.append("  {}   0   0   0   0   0   0       4158.5508   0   0\n".format(num_atoms))
    content.append("!Contributions (0/1) of this phase to the  1 patterns\n")
    content.append(" 1\n")
    content.append("!Irf Npr Jtyp  Nsp_Ref Ph_Shift for Pattern#  1\n")
    content.append("   0   5    0      0      0\n")
    content.append("! Pr1    Pr2    Pr3   Brind.   Rmua   Rmub   Rmuc     for Pattern#  1\n")
    content.append("  0.000  0.000  1.000  1.000  1.000  1.000  1.000\n")
    content.append("!\n")
    content.append("!\n")

    sg = cif_data.get('space_group', 'P 21/m')
    content.append("{}                   <--Space group symbol\n".format(sg))

    content.append("!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes\n")

    code_base = 211  
    for idx, atom in enumerate(cif_data['atoms']):
        biso = atom['uiso'] * 78.9568 if atom['uiso'] > 0 else 0.0

        occ = atom['occ']
        if abs(atom['y'] - 0.25) < 0.001:
            occ = 0.50000

        name = atom['name']
        typ = atom['type']
        x_code = code_base + idx * 3
        y_code = code_base + idx * 3 + 1 if abs(atom['y'] - 0.25) > 0.001 else 0  # Don't refine y on mirror
        z_code = code_base + idx * 3 + 2

        content.append("{:<6} {:<6}  {:>8.5f}  {:>8.5f}  {:>8.5f}  {:>7.5f}   {:>7.5f}   0   0   0    0  \n".format(
            name, typ, atom['x'], atom['y'], atom['z'], biso, occ
        ))
        content.append("                  0.00     0.00     0.00     0.00      0.00\n")

    content.append("!-------> Profile Parameters for Pattern #   1  ----> Phase #   1\n")
    content.append("!  Scale          Shape1      Bov      Str1      Str2      Str3   Strain-Model\n")
    content.append(" {}   {}   {}   0.00000   0.00000   0.00000       0\n".format(
        ref_params.get('scale', '0.3676107E-03'),
        ref_params.get('shape1', '0.20408'),
        ref_params.get('bov', '0.76810')
    ))
    content.append("       0.00000     0.000     0.000     0.000     0.000     0.000\n")

    content.append("!       U            V            W             X            Y         GauSiz      LorSiz    Size-Model\n")
    content.append("     {}    {}     {}     {}     {}     0.000000     0.000000       0\n".format(
        ref_params.get('u', '0.070671'),
        ref_params.get('v', '-0.043468'),
        ref_params.get('w', '0.025587'),
        ref_params.get('x', '0.006154'),
        ref_params.get('y', '0.000000')
    ))
    content.append("         0.00         0.00         0.00         0.00         0.00         0.00         0.00\n")

    content.append("!     a          b         c        alpha      beta       gamma      #Cell Info\n")
    content.append("   {:>9.6f}  {:>9.6f}   {:>9.6f}  {:>9.6f} {:>9.6f}  {:>9.6f}   \n".format(
        cif_data['a'], cif_data['b'], cif_data['c'],
        cif_data['alpha'], cif_data['beta'], cif_data['gamma']
    ))
    content.append("    0.00000    0.00000    0.00000    0.00000    0.00000    0.00000\n")

    content.append("!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4  \n")
    content.append("  1.00000  0.00000  {}  {}  {}  {}\n".format(
        ref_params.get('asy1', '0.05055'),
        ref_params.get('asy2', '0.01455'),
        ref_params.get('asy3', '0.00000'),
        ref_params.get('asy4', '0.00000')
    ))
    content.append("     0.00     0.00     0.00     0.00     0.00     0.00\n")

    content.append("!  2Th1/TOF1    2Th2/TOF2  Pattern to plot\n")
    content.append("       {}     {}       1\n".format(
        ref_params.get('th1', '5.120'),
        ref_params.get('th2', '119.980')
    ))

    try:
        with open(output_pcr_path, 'w') as f:
            f.writelines(content)
        print("\nSuccessfully created PCR file: {}".format(output_pcr_path))
        return True
    except Exception as e:
        print("Error writing PCR file: {}".format(e))
        return False

def apply_cif_parameters_to_pcr(pcr_file_path, cif_data):

    if not cif_data:
        print("Error: No CIF data to apply")
        return False

    try:
        with open(pcr_file_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if '!     a          b         c        alpha      beta       gamma' in line:
                if i+1 < len(lines):
                    lines[i+1] = "   {:>9.6f}  {:>9.6f}   {:>9.6f}  {:>9.6f} {:>9.6f}  {:>9.6f}   \n".format(
                        cif_data['a'], cif_data['b'], cif_data['c'],
                        cif_data['alpha'], cif_data['beta'], cif_data['gamma']
                    )
                    print("Updated lattice parameters from CIF")

        atom_idx = 0
        for i, line in enumerate(lines):
            if re.search(r'!Atom\s+Typ\s+X\s+Y\s+Z\s+Biso', line):
                j = i + 1
                while j < len(lines) and atom_idx < len(cif_data['atoms']):
                    if re.match(r'^[A-Za-z]+\d*\s+[A-Za-z]+\s+', lines[j]):
                        atom = cif_data['atoms'][atom_idx]
                        biso = atom['uiso'] * 78.9568 if atom['uiso'] > 0 else 0.0
                        occ = 0.50000 if abs(atom['y'] - 0.25) < 0.001 else atom['occ']

                        parts = lines[j].split()
                        if len(parts) >= 7:
                            lines[j] = "{:<6} {:<6}  {:>8.5f}  {:>8.5f}  {:>8.5f}  {:>7.5f}   {:>7.5f}   0   0   0    0  \n".format(
                                atom['name'], atom['type'],
                                atom['x'], atom['y'], atom['z'],
                                biso, occ
                            )
                        atom_idx += 1
                        j += 2  
                    else:
                        j += 1
                break

        with open(pcr_file_path, 'w') as f:
            f.writelines(lines)

        print("Applied {} atom positions from CIF to PCR".format(atom_idx))
        return True

    except Exception as e:
        print("Error applying CIF parameters: {}".format(e))
        return False

def apply_reference_params(pcr_file_path, root_dir, cif_name='Ba_vesta.cif'):

    try:
        cif_path = find_cif_file(cif_name, root_dir)
        if not cif_path:
            print("Warning: CIF file not found, using PCR file as-is")
            return False

        cif_data = parse_cif_file(cif_path)
        if not cif_data:
            print("Warning: Failed to parse CIF file")
            return False

        ref_pcr_path = find_reference_pcr_file('Ba.pcr', root_dir)
        if not ref_pcr_path:
            print("Warning: Reference PCR not found, using defaults")
        else:
            ref_params = extract_reference_parameters(ref_pcr_path)

        result = apply_cif_parameters_to_pcr(pcr_file_path, cif_data)

        return result

    except Exception as e:
        print("Error in apply_reference_params: {}".format(e))
        return False

def generate_ba_pcr(output_path, root_dir, dat_filename='Ba.dat', cif_name='Ba_vesta.cif'):
    cif_path = find_cif_file(cif_name, root_dir)
    if not cif_path:
        print("Error: Cannot find CIF file '{}'".format(cif_name))
        return False

    cif_data = parse_cif_file(cif_path)
    if not cif_data:
        print("Error: Cannot parse CIF file")
        return False

    ref_pcr_path = find_reference_pcr_file('Ba.pcr', root_dir)
    ref_params = {}
    if ref_pcr_path:
        ref_params = extract_reference_parameters(ref_pcr_path)

    if not ref_params:
        print("Warning: Using default instrumental parameters")
        ref_params = {
            'lambda1': '1.540560',
            'lambda2': '1.540560',
            'ratio': '0.00000',
            'bkpos': '25.000',
            'wdt': '15.0000',
            'cthm': '0.9100',
            'thmin': '5.1200',
            'step': '0.020006',
            'thmax': '119.9800',
            'excluded_regions': [('1.00', '2.00'), ('120.00', '160.00')],
            'zero': '0.00802',
            'bg_coeffs': ['1730.568', '-2032.394', '1746.730', '-820.944', '189.804', '-17.223'],
            'scale': '0.3676107E-03',
            'shape1': '0.20408',
            'bov': '0.76810',
            'u': '0.070671',
            'v': '-0.043468',
            'w': '0.025587',
            'x': '0.006154',
            'y': '0.000000',
            'asy1': '0.05055',
            'asy2': '0.01455',
            'asy3': '0.00000',
            'asy4': '0.00000',
            'th1': '5.120',
            'th2': '119.980'
        }

    return create_pcr_from_cif(output_path, cif_data, ref_params, dat_filename)

if __name__ == "__main__":
    print("Reference PCR script for Ba (Monoclinic)")
    print("=" * 50)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    print("Root directory: {}".format(root_dir))

    output_path = os.path.join(root_dir, 'dat_vestacif_files', 'test_Ba.pcr')

    if generate_ba_pcr(output_path, root_dir):
        print("\nTest PCR file created successfully!")
        print("Output: {}".format(output_path))
    else:
        print("\nFailed to create test PCR file")
