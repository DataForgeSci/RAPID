import re
import os
import sys
import numpy as np
import pandas as pd
import subprocess

def update_autofp_script(new_pcr_file_path, autofp_script_path):
    if not os.path.exists(autofp_script_path):
        raise IOError("The file '{}' does not exist.".format(autofp_script_path))
    
    with open(autofp_script_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("file_path = r'"):
            lines[i] = "    file_path = r'{}'\n".format(new_pcr_file_path)
            break

    with open(autofp_script_path, 'w') as file:
        file.writelines(lines)

def run_autofp_on_pcr(pcr_file_name, script_dir, subfolder_name, suppress_gui):
    # Debug print for input parameters
    print "PCR file name: %s" % pcr_file_name
    print "Script directory: %s" % script_dir
    print "Subfolder name: %s" % subfolder_name
    print "Display GUI: %s" % suppress_gui

    pcr_subfolder = os.path.splitext(pcr_file_name)[0]
    
    new_pcr_file_path = os.path.join(script_dir, '..', 'data', subfolder_name, pcr_subfolder, pcr_file_name)
    
    new_pcr_file_path = os.path.normpath(new_pcr_file_path)
    print "Full PCR file path: %s" % new_pcr_file_path

    if not os.path.exists(new_pcr_file_path):
        raise IOError("PCR file not found at path: %s" % new_pcr_file_path)

    autofp_dir = os.path.join(script_dir, '..', 'autofp-1.3.5')

    if suppress_gui == 'N':  # Suppress GUI if N
        autofp_script_path = os.path.join(autofp_dir, 'autofp_fs_unselect_GUI_suppressed.py')
        print "Using the suppressed GUI script."
    else:  # Do not suppress GUI if Y
        autofp_script_path = os.path.join(autofp_dir, 'autofp_fs_unselect_GUI_notsuppressed.py')
        print "Using the non-suppressed GUI script."

    update_autofp_script(new_pcr_file_path, autofp_script_path)

    sys.path.insert(0, autofp_dir)
    os.chdir(autofp_dir)

    setting_path = os.path.join(autofp_dir, "setting.txt")
    setting_default_path = os.path.join(autofp_dir, "setting_default.txt")
    
    if not os.path.exists(setting_path):
        print "setting.txt not found, loading setting_default.txt"
        if os.path.exists(setting_default_path):
            with open(setting_default_path, "r") as default_file:
                with open(setting_path, "w") as setting_file:
                    setting_file.write(default_file.read())
        else:
            raise IOError("Both setting.txt and setting_default.txt are missing")

    strategy_path = os.path.join(autofp_dir, "strategy")
    print "Strategy path: %s" % strategy_path
    if not os.path.exists(strategy_path):
        raise IOError("Strategy path is missing: %s" % strategy_path)

    process = subprocess.Popen([sys.executable, autofp_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print output.decode().strip()
    
    stderr = process.stderr.read().decode()
    if stderr:
        print stderr

def read_input_parameters(file_path):
    parameters = {}
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                if '>' in lines[i]:
                    key = lines[i].strip()
                    if i + 1 < len(lines) and '>' not in lines[i + 1]:
                        value = lines[i + 1].strip()
                        parameters[key] = value
    except Exception as e:
        print "Error reading input parameters from {}: {}".format(file_path, e)
    
    print "Read {} parameters from {}".format(len(parameters), file_path)
    for key in sorted(parameters.keys())[:5]:  # Show first 5 parameters to avoid clutter
        print "  {}: {}".format(key, parameters[key])
    
    return parameters

def validate_and_correct_pcr_format(pcr_file_path):
    with open(pcr_file_path, 'r') as file:
        lines = file.readlines()

    corrected_lines = []
    for line in lines:
        if re.match(r'^\s*!\s*a\s+b\s+c\s+alpha\s+beta\s+gamma\s+#\s*Cell\s*Info', line):
            corrected_line = '!     a          b         c        alpha      beta       gamma      #Cell Info\n'
            corrected_lines.append(corrected_line)
        elif re.match(r'^\s*!\s*Zero\s+Code\s+SyCos\s+Code\s+SySin\s+Code\s+Lambda\s+Code', line):
            corrected_line = '!  Zero    Code    SyCos    Code   SySin    Code  Lambda     Code MORE ->Patt# 1\n'
            corrected_lines.append(corrected_line)
        elif re.match(r'^\s*!\s*Background\s+coefficients/codes', line):
            corrected_line = '!   Background coefficients/codes  for Pattern#  1  (Polynomial of 6th degree)\n'
            corrected_lines.append(corrected_line)
        elif re.match(r'^\s*2Theta\s+Yobs\s+Ycal\s+Yobs-Ycal\s+Backg\s+Posr\s+\(hkl\)\s+K', line):
            corrected_line = '  2Theta     Yobs     Ycal     Yobs-Ycal     Backg      Posr   (hkl)       K\n'
            corrected_lines.append(corrected_line)
        else:
            corrected_lines.append(line)

    with open(pcr_file_path, 'w') as file:
        file.writelines(corrected_lines)

    print("Validated and corrected PCR file:", pcr_file_path)

def main():
    if len(sys.argv) != 2:
        print "Usage: python step3_runautofp_fixall.py <path_to_inputs.txt>"
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    if not os.path.exists(input_file_path):
        print "Error: The file '{}' does not exist.".format(input_file_path)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print "Debug: Script directory: %s" % script_dir
    print "Debug: Working directory: %s" % working_dir
    print "Debug: Input file path: %s" % input_file_path
    
    parameters = read_input_parameters(input_file_path)
    
    subfolder_param = '5>Enter the name of the subfolder to be created in the data directory (the purpose of the subfolder is to organize and store related data files and refinement results for analysis):'
    base_subfolder_name = parameters.get(subfolder_param)
    
    if base_subfolder_name is None:
        print "Warning: Could not find subfolder name parameter in inputs file."
        print "Looking for parameter key: {}".format(subfolder_param)
        print "Available parameter keys:"
        for key in sorted(parameters.keys()):
            print "  {}".format(key)
        
        working_dir_name = os.path.basename(working_dir)
        if "data" in working_dir:
            data_parent = os.path.dirname(working_dir)
            possible_folders = [d for d in os.listdir(data_parent) 
                               if os.path.isdir(os.path.join(data_parent, d)) 
                               and d != "backup"]
            
            if possible_folders:
                base_subfolder_name = possible_folders[0].split('_')[0]
                print "Using subfolder name from existing directory: {}".format(base_subfolder_name)
            else:
                input_dir = os.path.basename(os.path.dirname(input_file_path))
                if "_" in input_dir:
                    base_subfolder_name = input_dir.split('_')[0]
                    print "Using subfolder name from input file path: {}".format(base_subfolder_name)
                else:
                    base_subfolder_name = "sample"
                    print "Using default subfolder name: {}".format(base_subfolder_name)
    
    input_filename = os.path.basename(input_file_path)
    timestamp = input_filename.replace('inputs_', '').replace('.txt', '')
    
    subfolder_name = base_subfolder_name + "_" + timestamp
    print "Using subfolder name: {}".format(subfolder_name)
    
    subfolder_path = os.path.join(working_dir, 'data', subfolder_name)
    print "Debug: Subfolder path: %s" % subfolder_path
    
    if not os.path.exists(subfolder_path):
        print "Error: Subfolder path does not exist: {}".format(subfolder_path)
        
        data_dir = os.path.join(working_dir, 'data')
        matching_folders = [d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d)) 
                           and d.startswith(base_subfolder_name)]
        
        if matching_folders:
            subfolder_name = matching_folders[0]
            subfolder_path = os.path.join(working_dir, 'data', subfolder_name)
            print "Found alternative subfolder: {}".format(subfolder_path)
        else:
            print "No matching subfolder found. Available folders in data directory:"
            for d in os.listdir(data_dir):
                if os.path.isdir(os.path.join(data_dir, d)):
                    print "  {}".format(d)
            sys.exit(1)

    parameters = read_input_parameters(input_file_path)

    suppress_gui = parameters.get("15>Display AutoFP GUI interface (Y/N)?", 'Y').strip().upper()

    if suppress_gui not in ['Y', 'N']:
        raise ValueError("Invalid value for '15>Display AutoFP GUI interface (Y/N)?' in inputs.txt. Expected 'Y' or 'N'.")

    pcr_files = []
    for folder in os.listdir(subfolder_path):
        folder_path = os.path.join(subfolder_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".pcr"):
                    full_pcr_path = os.path.join(folder_path, file)
                    pcr_files.append((file, full_pcr_path)) 
                    print "Found PCR file: %s at %s" % (file, full_pcr_path)

    for pcr_file, pcr_full_path in pcr_files:
        print "Processing PCR file: %s" % pcr_file
        run_autofp_on_pcr(pcr_file, script_dir, subfolder_name, suppress_gui)
        validate_and_correct_pcr_format(pcr_full_path) 

if __name__ == "__main__":
    main()
