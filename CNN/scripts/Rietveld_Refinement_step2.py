#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

def read_ml_inputs(ml_file):

    if not os.path.isfile(ml_file):
        print("Error: {} not found. Exiting.".format(ml_file))
        sys.exit(1)

    lines = []
    with open(ml_file, 'r') as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            lines.append(t)
            if len(lines) == 5:
                break

    if len(lines) < 5:
        print("Error: ML_inputs file must have at least 5 non-comment lines.")
        sys.exit(1)

    line1 = lines[0].strip()
    model_name = "default_model"
    if ";" in line1:
        parts = line1.split(";", 1)
        model_name = parts[1].strip() if len(parts) > 1 else "default_model"

    dataset_line = lines[2].strip()
    dataset_folders = []
    if ";" in dataset_line:
        parts = dataset_line.split(";", 1)
        leftover = parts[1].strip()
        splitted = leftover.split(',')
        for ds in splitted:
            ds_clean = ds.strip()
            if ds_clean:
                dataset_folders.append(ds_clean)
    else:
        dataset_folders.append(dataset_line)

    if not dataset_folders:
        print("No dataset folder name found on line3.")
        sys.exit(1)

    experimental_line = lines[3].strip()
    experiment_base = os.path.splitext(experimental_line)[0]

    return model_name, dataset_folders, experiment_base

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
        from subprocess import Popen, PIPE
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        
        # Print any output
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
        print("Usage: python Rietveld_Refinement_step2.py <path_to_ML_inputs>")
        sys.exit(1)

    ml_inputs = sys.argv[1]
    model_name, dataset_folders, experiment_base = read_ml_inputs(ml_inputs)

    for dataset_name in dataset_folders:
        print("\nProcessing refinement step2 for dataset: {}".format(dataset_name))
        
        refine_root = os.path.join(
            "saved_models",
            model_name,
            "refinement_result",
            dataset_name,
            experiment_base,
            "Rietveld_Refinement"
        )
        
        if not os.path.exists(refine_root):
            print("Error: Refinement directory not found: {}".format(refine_root))
            continue  # Skip to next dataset instead of exiting

        cnn_pcr = os.path.join(refine_root, "CNN_ML_refinement", "cnn_refined.pcr")
        soln_pcr = os.path.join(refine_root, "solution_refinement", "*.pcr")
        
        soln_pcr_files = glob(soln_pcr)
        if not soln_pcr_files:
            print("Error: No solution PCR file found in {}".format(os.path.dirname(soln_pcr)))
            continue  # Skip to next dataset
        soln_pcr = soln_pcr_files[0]  # Take the first match

        if not os.path.exists(cnn_pcr):
            print("Error: CNN PCR file not found: {}".format(cnn_pcr))
            continue  # Skip to next dataset
        if not os.path.exists(soln_pcr):
            print("Error: Solution PCR file not found: {}".format(soln_pcr))
            continue  # Skip to next dataset

        print("\nRunning AutoFP refinement on PCR pairs...")
        
        print("\nProcessing CNN refined PCR...")
        run_autofp_on_pcr(cnn_pcr)
        
        print("\nProcessing solution PCR...")
        run_autofp_on_pcr(soln_pcr)
        
        print("\nAutoFP refinement completed for dataset: {}".format(dataset_name))

if __name__ == "__main__":
    main()