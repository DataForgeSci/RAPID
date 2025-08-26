#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
import sys
import glob
from subprocess import Popen, PIPE

def run_autofp_on_pcr(pcr_path):

    print("\nProcessing PCR file: {}".format(pcr_path))
    
    curr_dir = os.getcwd()
    pcr_abs_path = os.path.abspath(pcr_path)
    
    autofp_dir = os.path.abspath(os.path.join(curr_dir, '..', 'autofp-1.3.5'))
    
    try:
        os.chdir(autofp_dir)
        
        autofp_script = 'autofp_fs_unselect_GUI_suppressed.py'
        with open(autofp_script, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if "file_path = r'" in line:
                lines[i] = "    file_path = r'{}'\n".format(pcr_abs_path)
                break
        
        with open(autofp_script, 'w') as f:
            f.writelines(lines)
        
        cmd = [sys.executable, 'autofp_fs_unselect_GUI_suppressed.py']
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
            
    except Exception as e:
        print("Error running AutoFP on {}: {}".format(pcr_path, str(e)))
    finally:
        if os.path.exists('inputs.txt'):
            os.remove('inputs.txt')
        os.chdir(curr_dir)

def main():

    if len(sys.argv) < 2:
        print("Usage: python2.7 Rietveld_Refinement_step2_MI.py <identified_material_folder>")
        sys.exit(1)

    identified_dir = sys.argv[1]
    if not os.path.exists(identified_dir):
        print("Error: Identified material folder not found: {}".format(identified_dir))
        sys.exit(1)

    refine_root = os.path.join(identified_dir, "Rietveld_Refinement")
    if not os.path.exists(refine_root):
        print("Error: Refinement directory not found: {}".format(refine_root))
        sys.exit(1)

    cnn_dir = os.path.join(refine_root, "CNN_ML_refinement")
    if not os.path.exists(cnn_dir):
        print("Error: CNN refinement directory not found: {}".format(cnn_dir))
        sys.exit(1)

    cnn_pcr = os.path.join(cnn_dir, "cnn_refined.pcr")
    if not os.path.exists(cnn_pcr):
        print("Error: CNN PCR file not found: {}".format(cnn_pcr))
        sys.exit(1)

    print("\nRunning AutoFP refinement...")
    run_autofp_on_pcr(cnn_pcr)
    
    out_files = glob.glob(os.path.join(cnn_dir, "*.out"))
    if not out_files:
        print("\nWarning: No .out file was generated. Refinement may have failed.")
        sys.exit(1)
    else:
        print("\nAutoFP refinement completed successfully.")
        print("Output files generated:")
        for out_file in out_files:
            print("  - {}".format(os.path.basename(out_file)))

if __name__ == "__main__":
    main()