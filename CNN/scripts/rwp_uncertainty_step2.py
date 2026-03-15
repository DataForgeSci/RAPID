#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
import sys
import re
from subprocess import Popen, PIPE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.dirname(SCRIPT_DIR)

MATERIALS = {
    'CeO2': {
        'model_folder': 'model_CeO2_100epochs_digit_zero0p0075_lp0p005_May13_2025',
    },
    'pbso4': {
        'model_folder': 'model_pbso4_100epochs_digit_finetune_zero0p05_May07_2025',
    },
    'tbbaco': {
        'model_folder': 'model_tbbaco_100epochs_digit_June29_2025',
    },
}


def get_output_dir(material_name):
    if material_name not in MATERIALS:
        raise ValueError("Unknown material: {}. Available: {}".format(
            material_name, ', '.join(MATERIALS.keys())))
    model_folder = MATERIALS[material_name]['model_folder']
    return os.path.join(CNN_DIR, 'saved_models', 'backup', model_folder,
                        'uncertainty_results', 'rwp_uncertainty')


def run_autofp_on_pcr(pcr_path):
    print("\nProcessing PCR file: {}".format(pcr_path))

    curr_dir = os.getcwd()
    pcr_abs_path = os.path.abspath(pcr_path)

    autofp_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'autofp-1.3.5'))

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

        return True
    except Exception as e:
        print("Error running AutoFP: {}".format(str(e)))
        return False
    finally:
        if os.path.exists('inputs.txt'):
            os.remove('inputs.txt')
        os.chdir(curr_dir)


def extract_rwp(out_path):
    if not os.path.exists(out_path):
        return None
    with open(out_path, 'r') as f:
        content = f.read()
    match = re.search(r'=> Conventional Rietveld Rp,Rwp,Re and Chi2:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
    if match:
        return {'Rp': float(match.group(1)), 'Rwp': float(match.group(2)),
                'Re': float(match.group(3)), 'Chi2': float(match.group(4))}
    return None


def main():
    if len(sys.argv) < 3:
        print("Usage: python2.7 rwp_uncertainty_step2.py <material> <start_idx> [end_idx]")
        print("Available materials: {}".format(', '.join(MATERIALS.keys())))
        sys.exit(1)

    material = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3]) if len(sys.argv) > 3 else start_idx

    print("=" * 60)
    print("RWP UNCERTAINTY STEP 2 - {}".format(material))
    print("=" * 60)

    try:
        output_dir = get_output_dir(material)
    except ValueError as e:
        print("Error: {}".format(str(e)))
        sys.exit(1)

    print("Output directory: {}".format(output_dir))

    results = []

    for idx in range(start_idx, end_idx + 1):
        print("\nProcessing pattern {}...".format(idx))

        pattern_dir = os.path.join(output_dir, 'pattern_{:04d}'.format(idx))
        cnn_dir = os.path.join(pattern_dir, 'CNN_ML_refinement')
        pcr_path = os.path.join(cnn_dir, 'cnn_refined.pcr')

        if not os.path.exists(pcr_path):
            print("  Error: PCR file not found: {}".format(pcr_path))
            continue

        success = run_autofp_on_pcr(pcr_path)

        if success:
            out_path = os.path.join(cnn_dir, 'cnn_refined.out')
            rwp_data = extract_rwp(out_path)
            if rwp_data:
                print("  Rwp = {:.2f}%".format(rwp_data['Rwp']))
                results.append({'idx': idx, 'rwp': rwp_data['Rwp']})

                # Save result
                with open(os.path.join(pattern_dir, 'rwp_result.dat'), 'w') as f:
                    f.write("Rwp = {:.4f}\n".format(rwp_data['Rwp']))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY - {}".format(material))
    print("=" * 60)
    if results:
        rwp_values = [r['rwp'] for r in results]
        mean_rwp = sum(rwp_values) / len(rwp_values)
        variance = sum((x - mean_rwp)**2 for x in rwp_values) / len(rwp_values)
        std_rwp = variance ** 0.5
        print("Patterns processed: {}".format(len(results)))
        print("Mean Rwp: {:.4f}%".format(mean_rwp))
        print("Std Rwp:  {:.4f}%".format(std_rwp))

        # Save summary
        summary_path = os.path.join(output_dir, 'rwp_summary_{}_to_{}.dat'.format(start_idx, end_idx))
        with open(summary_path, 'w') as f:
            f.write("Material: {}\n".format(material))
            f.write("Patterns: {} to {}\n".format(start_idx, end_idx))
            f.write("Patterns processed: {}\n".format(len(results)))
            f.write("Mean Rwp: {:.4f}%\n".format(mean_rwp))
            f.write("Std Rwp:  {:.4f}%\n".format(std_rwp))
            f.write("\nIndividual Rwp values:\n")
            for r in results:
                f.write("  Pattern {:04d}: {:.4f}%\n".format(r['idx'], r['rwp']))
        print("Summary saved to: {}".format(summary_path))

    print("=" * 60)


if __name__ == "__main__":
    main()
