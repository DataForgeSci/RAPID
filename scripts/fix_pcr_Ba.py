# -*- coding: utf-8 -*-

import os
import re
import sys

REFERENCE_VALUES = {
    'npr': 5,              # Npr value (line 8 and 48)
    'ncy': 10,             # Number of cycles
    'rpa': -1,             # Rpa value (line 14)

    'ratio': '0.00000',
    'bkpos': '25.000',
    'wdt': '15.0000',
    'cthm': '0.9100',

    'thmin': '5.1200',
    'step': '0.020006',
    'thmax': '119.9800',

    'excluded_regions': [('1.00', '2.00'), ('120.00', '160.00')],

    'zero': '0.00802',
    'zero_code': '0.0',

    'bg_coeffs': ['1730.568', '-2032.394', '1746.730', '-820.944', '189.804', '-17.223'],
    'bg_codes': ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00'],

    'scale': '0.3676107E-03',
    'shape1': '0.20408',
    'bov': '0.76810',
    'scale_code': '0.00',
    'shape1_code': '0.000',
    'bov_code': '0.000',

    'u': '0.070671',
    'v': '-0.043468',
    'w': '0.025587',
    'x': '0.006154',
    'y': '0.000000',
    'u_code': '0.00',
    'v_code': '0.00',
    'w_code': '0.00',
    'x_code': '0.00',

    'asy1': '0.05055',
    'asy2': '0.01455',
    'asy3': '0.00000',
    'asy4': '0.00000',
    'asy1_code': '0.00',
    'asy2_code': '0.00',

    'th1': '5.120',
    'th2': '119.980',

    'atz': '4158.5508',

    'biso': '0.00000',   
    'n_t': '0',          
}

def convert_4line_to_2line(lines):

    new_lines = []
    i = 0
    in_atoms = False

    while i < len(lines):
        line = lines[i]

        if '!Atom   Typ' in line and 'X' in line and 'Y' in line and 'Z' in line:
            in_atoms = True
            new_lines.append(line)
            # Skip the beta header line if present
            if i+1 < len(lines) and 'beta11' in lines[i+1]:
                i += 2
                continue
            i += 1
            continue

        if in_atoms and '!-------> Profile Parameters' in line:
            in_atoms = False
            new_lines.append(line)
            i += 1
            continue

        if in_atoms:
            if re.match(r'^[A-Za-z]+\d*\s+[A-Za-z]+\s+', line):
                parts = line.split()
                if len(parts) >= 7:
                    name = parts[0]
                    typ = parts[1]
                    x = parts[2]
                    y = parts[3]
                    z = parts[4]
                    biso = REFERENCE_VALUES['biso']
                    occ = parts[6]

                    try:
                        y_val = float(y)
                        on_mirror = abs(y_val - 0.25) < 0.001
                    except:
                        on_mirror = False

                    if on_mirror:
                        occ = '0.50000'

                    atom_line = "{:<6} {:<6}  {:>8}  {:>8}  {:>8}  {:>7}   {:>7}   0   0   {}    0  \n".format(
                        name, typ, x, y, z, biso, occ, REFERENCE_VALUES['n_t']
                    )
                    new_lines.append(atom_line)

                    codes_line = "                  0.00     0.00     0.00     0.00      0.00\n"
                    new_lines.append(codes_line)

                    i += 4
                    continue
            else:
                new_lines.append(line)
                i += 1
                continue
        else:
            new_lines.append(line)

        i += 1

    return new_lines

def fix_pcr_file(pcr_path):

    if not os.path.exists(pcr_path):
        print("Error: PCR file not found: {}".format(pcr_path))
        return False

    try:
        with open(pcr_path, 'r') as f:
            lines = f.readlines()

        has_beta = any('beta11' in line for line in lines)

        if has_beta:
            print("Converting 4-line to 2-line atom format...")
            lines = convert_4line_to_2line(lines)

        new_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            if '!Job Npr Nba Nex Nsc Nor Iwg Ilo Res Ste Uni Cor Anm Int' in line:
                new_lines.append(line)
                if i+1 < len(lines):
                    parts = lines[i+1].split()
                    if len(parts) >= 2:
                        parts[1] = str(REFERENCE_VALUES['npr'])
                    new_lines.append('   ' + '   '.join(parts[:14]) + '  !-> Patt#: 1\n')
                    i += 2
                    continue

            elif '!Mat Pcr NLI Rpa Sym Sho' in line:
                new_lines.append(line)
                if i+1 < len(lines):
                    new_lines.append('   0   1   0  {}   0   0\n'.format(REFERENCE_VALUES['rpa']))
                    i += 2
                    continue

            elif '!NCY  Eps  R_at  R_an  R_pr  R_gl' in line:
                new_lines.append(line)
                new_lines.append(' {}  0.10  1.00  1.00  1.00  1.00\n'.format(REFERENCE_VALUES['ncy']))
                i += 2
                continue

            elif '! Lambda1  Lambda2' in line:
                new_lines.append(line)
                new_lines.append(' 1.540560 1.540560  {}   {} {}  {}  0.0000   60.00    0.0000  0.0000\n'.format(
                    REFERENCE_VALUES['ratio'],
                    REFERENCE_VALUES['bkpos'],
                    REFERENCE_VALUES['wdt'],
                    REFERENCE_VALUES['cthm']
                ))
                i += 2
                continue

            elif '!     Thmin       Step       Thmax' in line:
                new_lines.append(line)
                new_lines.append('     {}   {}   {}   0.000   0.000\n'.format(
                    REFERENCE_VALUES['thmin'],
                    REFERENCE_VALUES['step'],
                    REFERENCE_VALUES['thmax']
                ))
                i += 2
                continue

            elif '! Excluded regions' in line:
                new_lines.append(line)
                for region in REFERENCE_VALUES['excluded_regions']:
                    new_lines.append('        {}        {}\n'.format(region[0], region[1]))
                # Skip old excluded region lines
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('!'):
                    i += 1
                continue

            elif '!  Zero    Code    SyCos' in line:
                new_lines.append(line)
                new_lines.append('  {}   {}  0.00000    0.0  0.00000    0.0 0.000000    0.00   0\n'.format(
                    REFERENCE_VALUES['zero'],
                    REFERENCE_VALUES['zero_code']
                ))
                i += 2
                continue

            elif '!   Background coefficients/codes' in line:
                new_lines.append(line)
                bg = REFERENCE_VALUES['bg_coeffs']
                codes = REFERENCE_VALUES['bg_codes']
                new_lines.append('    {}   {}    {}    {}     {}     {}\n'.format(*bg))
                new_lines.append('       {}       {}       {}       {}       {}       {}\n'.format(*codes))
                i += 3
                continue

            elif line.strip() == 'Ba' or line.strip() == 'Ba, VARY xyz':
                new_lines.append('Ba\n')
                i += 1
                continue

            elif '!Nat Dis Ang Jbt Isy Str Furth' in line:
                new_lines.append(line)
                if i+1 < len(lines):
                    parts = lines[i+1].split()
                    # Keep Nat (number of atoms) but fix ATZ
                    nat = parts[0] if len(parts) > 0 else '21'
                    new_lines.append('  {}   0   0   0   0   0   0       {}   0   0\n'.format(
                        nat, REFERENCE_VALUES['atz']
                    ))
                    i += 2
                    continue

            elif '!Irf Npr Jtyp  Nsp_Ref Ph_Shift for Pattern#  1' in line:
                new_lines.append(line)
                new_lines.append('   0   {}    0      0      0\n'.format(REFERENCE_VALUES['npr']))
                i += 2
                continue

            elif '!  Scale' in line and 'Shape1' in line and 'Bov' in line:
                new_lines.append(line)
                new_lines.append(' {}   {}   {}   0.00000   0.00000   0.00000       0\n'.format(
                    REFERENCE_VALUES['scale'],
                    REFERENCE_VALUES['shape1'],
                    REFERENCE_VALUES['bov']
                ))
                new_lines.append('      {}   {}   {}     0.000     0.000     0.000\n'.format(
                    REFERENCE_VALUES['scale_code'],
                    REFERENCE_VALUES['shape1_code'],
                    REFERENCE_VALUES['bov_code']
                ))
                i += 3
                continue

            elif '!       U' in line and 'V' in line and 'W' in line and 'GauSiz' in line:
                new_lines.append(line)
                new_lines.append('     {}    {}     {}     {}     {}     0.000000     0.000000       0\n'.format(
                    REFERENCE_VALUES['u'],
                    REFERENCE_VALUES['v'],
                    REFERENCE_VALUES['w'],
                    REFERENCE_VALUES['x'],
                    REFERENCE_VALUES['y']
                ))
                new_lines.append('       {}       {}       {}       {}         0.00         0.00         0.00\n'.format(
                    REFERENCE_VALUES['u_code'],
                    REFERENCE_VALUES['v_code'],
                    REFERENCE_VALUES['w_code'],
                    REFERENCE_VALUES['x_code']
                ))
                i += 3
                continue

            elif '!     a          b         c        alpha      beta       gamma' in line:
                new_lines.append(line)
                # Keep the lattice values line as-is (next line)
                if i+1 < len(lines):
                    new_lines.append(lines[i+1])
                # Replace the codes line with all zeros
                new_lines.append('    0.00000    0.00000    0.00000    0.00000    0.00000    0.00000\n')
                i += 3
                continue

            elif '!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4' in line:
                new_lines.append(line)
                new_lines.append('  1.00000  0.00000  {}  {}  {}  {}\n'.format(
                    REFERENCE_VALUES['asy1'],
                    REFERENCE_VALUES['asy2'],
                    REFERENCE_VALUES['asy3'],
                    REFERENCE_VALUES['asy4']
                ))
                new_lines.append('     0.00     0.00   {}   {}     0.00     0.00\n'.format(
                    REFERENCE_VALUES['asy1_code'],
                    REFERENCE_VALUES['asy2_code']
                ))
                i += 3
                continue

            elif '!  2Th1/TOF1    2Th2/TOF2' in line:
                new_lines.append(line)
                new_lines.append('       {}     {}       1\n'.format(
                    REFERENCE_VALUES['th1'],
                    REFERENCE_VALUES['th2']
                ))
                i += 2
                continue

            else:
                new_lines.append(line)

            i += 1

        with open(pcr_path, 'w') as f:
            f.writelines(new_lines)

        print("Successfully fixed PCR file: {}".format(pcr_path))
        return True

    except Exception as e:
        print("Error fixing PCR file: {}".format(e))
        import traceback
        traceback.print_exc()
        return False

def fix_all_pcr_in_folder(folder_path):

    if not os.path.isdir(folder_path):
        print("Error: Folder not found: {}".format(folder_path))
        return False

    pcr_files = [f for f in os.listdir(folder_path) if f.endswith('.pcr')]

    if not pcr_files:
        print("No PCR files found in: {}".format(folder_path))
        return False

    print("Found {} PCR files to fix".format(len(pcr_files)))

    for pcr_file in pcr_files:
        pcr_path = os.path.join(folder_path, pcr_file)
        print("\nFixing: {}".format(pcr_file))
        fix_pcr_file(pcr_path)

    return True

def fix_ba_pcr(pcr_path):

    return fix_pcr_file(pcr_path)

def fix_ba_pcr_with_root(pcr_path, root_dir):

    return fix_pcr_file(pcr_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_pcr_Ba.py <pcr_file_or_folder>")
        print("Example: python fix_pcr_Ba.py Ba.pcr")
        print("Example: python fix_pcr_Ba.py ./data/Ba_20251201_051922/")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isfile(target):
        fix_pcr_file(target)
    elif os.path.isdir(target):
        fix_all_pcr_in_folder(target)
    else:
        print("Error: {} is not a valid file or folder".format(target))
        sys.exit(1)
