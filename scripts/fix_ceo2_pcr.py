# -*- coding: utf-8 -*-

import os
import re
import sys

def extract_reference_values(ref_pcr_path):

    values = {
        'zero': '',
        'bg_coeffs': [],
        'scale': '',
        'u': '',
        'v': '',
        'w': ''
    }
    
    try:
        with open(ref_pcr_path, 'r') as file:
            lines = file.readlines()
            
        for i, line in enumerate(lines):
            if '!  Zero    Code    SyCos' in line and i + 1 < len(lines):
                zero_parts = lines[i + 1].split()
                if zero_parts:
                    values['zero'] = zero_parts[0]
            
            elif '!   Background coefficients/codes' in line and i + 1 < len(lines):
                bg_parts = lines[i + 1].split()
                if len(bg_parts) >= 6:
                    values['bg_coeffs'] = bg_parts[:6]
            
            elif '!  Scale' in line and 'Strain-Model' in line and i + 1 < len(lines):
                scale_parts = lines[i + 1].split()
                if scale_parts:
                    values['scale'] = scale_parts[0]
            
            elif '!       U' in line and 'Size-Model' in line and i + 1 < len(lines):
                uvw_parts = lines[i + 1].split()
                if len(uvw_parts) >= 3:
                    values['u'] = uvw_parts[0]
                    values['v'] = uvw_parts[1]
                    values['w'] = uvw_parts[2]
            
        
        return values
        
    except Exception as e:
        print("Error extracting reference values: {}".format(e))
        return None

def extract_cif_values_from_pcr(pcr_file_path):

    cif_values = {
        'phase_name': '',
        'space_group': '',
        'atom_biso': {},
        'a': '',
        'b': '',
        'c': '',
        'alpha': '',
        'beta': '',
        'gamma': ''
    }
    
    try:
        with open(pcr_file_path, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if '!  Data for PHASE number:' in line and i + 2 < len(lines):
                cif_values['phase_name'] = lines[i + 2].strip()
            
            elif '<--Space group symbol' in line:
                cif_values['space_group'] = line.strip().split('<--')[0].strip()
            
            elif '!     a          b         c        alpha      beta       gamma      #Cell Info' in line and i + 1 < len(lines):
                lattice_parts = lines[i + 1].split()
                if len(lattice_parts) >= 6:
                    cif_values['a'] = lattice_parts[0]
                    cif_values['b'] = lattice_parts[1]
                    cif_values['c'] = lattice_parts[2]
                    cif_values['alpha'] = lattice_parts[3]
                    cif_values['beta'] = lattice_parts[4]
                    cif_values['gamma'] = lattice_parts[5]
            
            elif re.match(r'^[A-Za-z0-9]+\s+[A-Za-z0-9]+\s+[-+]?\d*\.\d+', line):
                parts = line.split()
                if len(parts) >= 6:
                    atom_name = parts[0]
                    cif_values['atom_biso'][atom_name] = parts[5]
        
        return cif_values
        
    except Exception as e:
        print("Error extracting CIF values: {}".format(e))
        return {}

def fix_ceo2_pcr(pcr_file_path):

    try:
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(pcr_file_path)))
        
        import sys
        sys.path.insert(0, os.path.join(root_dir, 'scripts'))
        
        from reference_pcr_CeO2 import find_reference_pcr_file
        
        ref_pcr_path = find_reference_pcr_file('CeO2.pcr', root_dir)
        if not ref_pcr_path:
            print("Error: Could not find reference CeO2.pcr")
            return False
        
        ref_values = extract_reference_values(ref_pcr_path)
        if not ref_values:
            print("Error: Could not extract values from reference PCR")
            return False
        
        cif_values = extract_cif_values_from_pcr(pcr_file_path)
        
        with open(pcr_file_path, 'r') as file:
            lines = file.readlines()
        
        modifications = []
        
        new_lines = []
        i = 0
        skip_o1_atom = False
        while i < len(lines):
            if i < len(lines) and lines[i].strip().startswith("O1 O"):
                skip_o1_atom = True
                i += 4  # Skip a total of 4 lines
                modifications.append("Removed O1 atom and its beta lines")
                continue
            
            new_lines.append(lines[i])
            i += 1
        
        lines = new_lines
        
        ce_biso = cif_values['atom_biso'].get('Ce', '0.03938')
        o_biso = cif_values['atom_biso'].get('O', '0.84886')
        
        solution_pcr = """COMM   CeO2_.pcr
! Current global Chi2 (Bragg contrib.) =      21.94    
NPATT      1       1 <- Flags for patterns (1:refined, 0: excluded)
W_PAT   1.000
!Nph Dum Ias Nre Cry Opt Aut
   1   0   0   0   0   0   1
!Job Npr Nba Nex Nsc Nor Iwg Ilo Res Ste Uni Cor Anm Int
   0   7   0   2   0   1   0   0   0   0   0   0   0   0  !-> Patt#: 1
!
!File names of data(patterns) files
CeO2.dat
!
!Mat Pcr NLI Rpa Sym Sho
   0   1   0   0   0   0
!Ipr Ppl Ioc Ls1 Ls2 Ls3 Prf Ins Hkl Fou Ana
   0   0   1   0   4   0   3  10   0   0   0  !-> Patt#: 1
!
! Lambda1  Lambda2    Ratio    Bkpos    Wdt    Cthm     muR   AsyLim   Rpolarz  2nd-muR -> Patt# 1
 1.540560 1.544390  0.50000   25.000 15.0000  0.9100  0.0000   60.00    0.0000  0.0000
!
!NCY  Eps  R_at  R_an  R_pr  R_gl
 10  0.10  1.00  1.00  1.00  1.00
!     Thmin       Step       Thmax    PSD    Sent0  -> Patt#: 1
    25.1243   0.025006   142.9743   0.000   0.000
!
! Excluded regions (LowT  HighT) for Pattern#  1
        1.00       20.00
      100.00      160.00
! 
!
       0    !Number of refined parameters
!
!  Zero    Code    SyCos    Code   SySin    Code  Lambda     Code MORE ->Patt# 1
 {}    0.0  0.00000    0.0  0.00000    0.0 0.000000    0.00   0
!   Background coefficients/codes  for Pattern#  1  (Polynomial of 6th degree)
      {}     {}      {}     {}       {}       0.000
        0.00        0.00        0.00        0.00        0.00        0.00
!-------------------------------------------------------------------------------
!  Data for PHASE number:   1  ==> Current R_Bragg for Pattern#  1:     4.75
!-------------------------------------------------------------------------------
{}
!
!Nat Dis Ang Jbt Isy Str Furth        ATZ     Nvk More
   2   0   0   0   0   0   0        688.2855   0   0
!Contributions (0/1) of this phase to the  1 patterns
 1
!Irf Npr Jtyp  Nsp_Ref Ph_Shift for Pattern#  1
   0   7    0      0      0
! Pr1    Pr2    Pr3   Brind.   Rmua   Rmub   Rmuc     for Pattern#  1
  0.000  0.000  1.000  1.000  1.000  1.000  1.000
!
{}                 <--Space group symbol
!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes
Ce     Ce      0.00000  0.00000  0.00000  {}   0.02083   0   0   0    0  
                  0.00     0.00     0.00     0.00      0.00
O      O       0.25000  0.25000  0.25000  {}   0.04167   0   0   0    0  
                  0.00     0.00     0.00     0.00      0.00
!-------> Profile Parameters for Pattern #  1
!  Scale        Shape1      Bov      Str1      Str2      Str3   Strain-Model
 {}   0.00000   0.00000   0.00000   0.00000   0.00000       0
     0.00000     0.000     0.000     0.000     0.000     0.000
!       U         V          W           X          Y        GauSiz   LorSiz Size-Model
   {}  {}   {}   0.074822   0.000000   0.000000   0.000000    0
      0.000      0.000      0.000      0.000      0.000      0.000      0.000
!     a          b         c        alpha      beta       gamma      #Cell Info
   {}   {}   {}  {}  {}  {}   
    0.00000    0.00000    0.00000    0.00000    0.00000    0.00000
!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4      S_L      D_L
  1.00000  0.00000  0.00000  0.07230  0.00000  0.00000  0.00000  0.00000
     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00
!  2Th1/TOF1    2Th2/TOF2  Pattern to plot
      25.124     100.000       1""".format(
            ref_values['zero'],  # Zero from reference
            ref_values['bg_coeffs'][0], ref_values['bg_coeffs'][1], ref_values['bg_coeffs'][2],  # Background from reference
            ref_values['bg_coeffs'][3], ref_values['bg_coeffs'][4],
            cif_values['phase_name'],  # Phase name from CIF
            cif_values['space_group'],  # Space group from CIF
            ce_biso,  # Biso from CIF calculation
            o_biso,   # Biso from CIF calculation
            ref_values['scale'],  # Scale from reference
            ref_values['u'], ref_values['v'], ref_values['w'],  # U, V, W from reference
            cif_values['a'], cif_values['b'], cif_values['c'],  # Lattice parameters from CIF
            cif_values['alpha'], cif_values['beta'], cif_values['gamma']  # Angles from CIF
        )
        
        with open(pcr_file_path, 'w') as file:
            file.write(solution_pcr)
        
        print("Successfully fixed CeO2 PCR file using dynamic values")
        print("- Zero: {} (from reference)".format(ref_values['zero']))
        print("- Background: {} (from reference)".format(ref_values['bg_coeffs']))
        print("- Biso values: Ce={}, O={} (from CIF)".format(ce_biso, o_biso))
        print("- Scale: {} (from reference)".format(ref_values['scale']))
        print("- U,V,W: {},{},{} (from reference)".format(ref_values['u'], ref_values['v'], ref_values['w']))
        print("- Lattice: a={}, b={}, c={} (from CIF)".format(cif_values['a'], cif_values['b'], cif_values['c']))
        print("- Angles: alpha={}, beta={}, gamma={} (from CIF)".format(cif_values['alpha'], cif_values['beta'], cif_values['gamma']))
        print("- Phase name: {} (from CIF)".format(cif_values['phase_name']))
        print("- Space group: {} (from CIF)".format(cif_values['space_group']))
        
        return True
    
    except Exception as e:
        print("Error fixing PCR file: {}".format(e))
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_ceo2_pcr.py <path_to_pcr_file>")
        sys.exit(1)
    
    pcr_file_path = sys.argv[1]
    if not os.path.exists(pcr_file_path):
        print("Error: PCR file '{}' does not exist".format(pcr_file_path))
        sys.exit(1)
    
    fix_ceo2_pcr(pcr_file_path)