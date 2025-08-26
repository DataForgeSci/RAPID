# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import time
import threading

def list_macro_input_files(macro_inputs_dir, pattern=None):
    
    if pattern:
        import fnmatch
        files = [f for f in os.listdir(macro_inputs_dir) if f.lower().endswith('.txt') and fnmatch.fnmatch(f, pattern)]
    else:
        files = [f for f in os.listdir(macro_inputs_dir) if f.lower().endswith('.txt')]
    
    files = sorted(files)
    
    if not files:
        print("No .txt files found in {}".format(macro_inputs_dir))
        return []
    
    print("\nAvailable input files in '{}':".format(macro_inputs_dir))
    for i, f in enumerate(files, start=1):
        print("{}. {}".format(i, f))
    
    return files

def confirm_process_all_files(files_count):
    while True:
        confirm = raw_input("\nDo you want to process all {} input files? (Y/N): ".format(files_count)).strip().upper()
        if confirm in ['Y', 'N']:
            return confirm == 'Y'
        print("Invalid input. Please enter 'Y' or 'N'.")

def write_inputs_txt_from_file(source_file, target_file):
    with open(source_file, 'r') as sf, open(target_file, 'w') as tf:
        tf.write(sf.read())

def monitor_and_handle_pause(process):
    time.sleep(0.5)
    
    try:
        process.stdin.write(b'\n')
        process.stdin.flush()
    except:
        pass

def run_data_augmentation(root_dir, non_blocking=False):

    data_aug_bat = os.path.join(root_dir, "data_augmentation.bat")
    if not os.path.exists(data_aug_bat):
        print("Error: data_augmentation.bat not found in the root directory.")
        return False

    print("Running data_augmentation.bat...")
    
    if non_blocking:
        process = subprocess.Popen(
            [data_aug_bat], 
            cwd=root_dir, 
            shell=True, 
            stdin=subprocess.PIPE,
        )
        
        pause_handler = threading.Thread(target=monitor_and_handle_pause, args=(process,))
        pause_handler.daemon = True
        pause_handler.start()
        
        process.wait()
    else:
        process = subprocess.Popen([data_aug_bat], cwd=root_dir, shell=True)
        process.wait()

    if process.returncode != 0:
        print("Error: data_augmentation.bat encountered an error.")
        return False
    
    print("data_augmentation.bat completed successfully.")
    return True

def main():
    process_all = "--process_all" in sys.argv
    non_blocking = "--non_blocking" in sys.argv  # New flag for non-blocking operation

    process_pattern = None
    for arg in sys.argv:
        if arg.startswith("--process_pattern="):
            process_pattern = arg.split("=", 1)[1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # 'scripts' is a subfolder of root
    macro_inputs_dir = os.path.join(root_dir, "macro_inputs")

    if not os.path.exists(macro_inputs_dir):
        print("Error: 'macro_inputs' folder does not exist in the root directory.")
        sys.exit(1)

    files = list_macro_input_files(macro_inputs_dir, process_pattern)
    if not files:
        print("No input files available. Exiting.")
        sys.exit(0)

    if process_all or process_pattern:
        confirmation = raw_input("\nDo you want to process all {} matching files? (Y/N): ".format(len(files))).strip().upper()
        if confirmation != "Y":
            print("Operation cancelled. Exiting.")
            sys.exit(0)
            
        root_inputs_path = os.path.join(root_dir, "inputs.txt")
        for idx, filename in enumerate(files, start=1):
            file_path = os.path.join(macro_inputs_dir, filename)
            
            print("\n=== Processing File {} of {}: {} ===".format(idx, len(files), filename))
            
            write_inputs_txt_from_file(file_path, root_inputs_path)
            print("inputs.txt updated for Session {}.".format(idx))
            
            success = run_data_augmentation(root_dir, non_blocking)
            if not success:
                print("Session {} failed. Moving to the next file.".format(idx))
                continue
            else:
                print("Session {} completed successfully.".format(idx))
        
        print("\nAll files have been processed.")
    
    else:
        sessions = []
        used_files = {}  # Dictionary: filename -> session_number in which it's used
        
        while True:
            chosen_file, file_path = select_input_file(files, used_files, macro_inputs_dir)
            display_file_content(file_path)
            confirm = confirm_file_choice()
            if confirm == 'Y':
                session_number = len(sessions) + 1
                sessions.append(file_path)
                used_files[chosen_file] = session_number
                print("This file is added as session {}.".format(session_number))
                add_more = ask_add_more_sessions()
                if add_more == 'N':
                    break
                else:
                    continue
            else:
                print("File not used. Please select another file.")
                continue

        if not sessions:
            print("No sessions selected. Exiting.")
            sys.exit(0)

        start_now = confirm_start_sessions()
        if start_now == 'N':
            print("Not starting sessions. Exiting.")
            sys.exit(0)

        root_inputs_path = os.path.join(root_dir, "inputs.txt")
        for idx, session_file in enumerate(sessions, start=1):
            print("\n=== Processing Session {} ===".format(idx))
            write_inputs_txt_from_file(session_file, root_inputs_path)
            print("inputs.txt updated for Session {}.".format(idx))

            success = run_data_augmentation(root_dir, non_blocking)
            if not success:
                print("Session {} failed. Moving to the next session.".format(idx))
                continue
            else:
                print("Session {} completed successfully.".format(idx))

        print("\nAll chosen sessions have been processed.")

def select_input_file(files, used_files, macro_inputs_dir):
    while True:
        print("\nPlease select a file for this session:")
        for i, f in enumerate(files, start=1):
            if f in used_files:
                print("{}. {} (already used in session {})".format(i, f, used_files[f]))
            else:
                print("{}. {}".format(i, f))

        choice = raw_input("Enter the index of the input file for this session: ").strip()
        if not choice.isdigit():
            print("Invalid input. Please enter a number.")
            continue
        idx = int(choice)
        if idx < 1 or idx > len(files):
            print("Invalid index. Please choose a valid number from the list.")
            continue
        chosen_file = files[idx-1]
        file_path = os.path.join(macro_inputs_dir, chosen_file)
        return chosen_file, file_path

def display_file_content(file_path):
    print("\n--- Content of {} ---".format(file_path))
    with open(file_path, 'r') as f:
        content = f.read()
    print(content)
    print("--- End of file content ---\n")

def confirm_file_choice():
    while True:
        confirm = raw_input("Use this file for this session? (Y/N): ").strip().upper()
        if confirm in ['Y','N']:
            return confirm
        print("Invalid input. Please enter 'Y' or 'N'.")

def confirm_start_sessions():
    while True:
        confirm = raw_input("Start the macro sessions with the chosen inputs? (Y/N): ").strip().upper()
        if confirm in ['Y','N']:
            return confirm
        print("Invalid input. Please enter 'Y' or 'N'.")

def ask_add_more_sessions():
    while True:
        add_more = raw_input("Do you want to add another session? (Y/N): ").strip().upper()
        if add_more in ['Y', 'N']:
            return add_more
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

if __name__ == "__main__":
    main()