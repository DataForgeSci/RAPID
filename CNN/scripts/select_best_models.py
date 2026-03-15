#!/usr/bin/env python


import os
import re
import shutil
import glob
import sys

def extract_rwp_from_report(report_path):

    try:
        with open(report_path, 'r') as f:
            content = f.read()
            lines = content.splitlines()

            if re.search(r'CNN_ML.*?\bNA\b|\bN/A\b', content, re.IGNORECASE):
                print("Found NA/N/A value for CNN_ML Rwp in {}".format(report_path))
                return float('inf')

            for line in lines:
                line = line.strip()
                if line.startswith('CNN_ML'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            if parts[2].lower() in ('na', 'n/a', '--', 'none', 'null'):
                                print("Found non-numeric value '{}' for CNN_ML Rwp in {}".format(parts[2], report_path))
                                return float('inf')

                            return float(parts[2])
                        except ValueError:
                            if len(parts) >= 4:
                                try:
                                    return float(parts[3])
                                except ValueError:
                                    pass

            header_line = None
            for i, line in enumerate(lines):
                if "RefinementType" in line and "Rwp" in line:
                    header_line = line
                    break

            if header_line:
                header_parts = header_line.split()
                try:
                    rwp_index = header_parts.index("Rwp(%)")
                    for line in lines:
                        if line.strip().startswith("CNN_ML"):
                            parts = line.split()
                            if len(parts) > rwp_index:
                                if parts[rwp_index].lower() in ('na', 'n/a', '--', 'none', 'null'):
                                    print("Found non-numeric value '{}' for CNN_ML Rwp in {}".format(parts[rwp_index], report_path))
                                    return float('inf')
                                try:
                                    return float(parts[rwp_index])
                                except ValueError:
                                    pass
                except ValueError:
                    pass  # Rwp not found in header

            cnn_ml_line_pattern = re.compile(r'CNN_ML\s+.*?(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', re.IGNORECASE)
            matches = cnn_ml_line_pattern.findall(content)
            if matches and len(matches[0]) >= 2:
                try:
                    return float(matches[0][1])
                except ValueError:
                    pass

        print("Warning: Could not extract Rwp from {}".format(report_path))
        return float('inf')
    except Exception as e:
        print("Error processing {}: {}".format(report_path, str(e)))
        return float('inf')

def find_analysis_report(folder_path):
    common_patterns = [
        os.path.join(folder_path, "refinement_result", "*", "*", "Rietveld_Refinement", "output_analysis", "analysis_report.dat"),
        os.path.join(folder_path, "refinement_result", "*", "Rietveld_Refinement", "output_analysis", "analysis_report.dat"),
        os.path.join(folder_path, "**", "output_analysis", "analysis_report.dat")
    ]

    for pattern in common_patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    # Fallback: search recursively (Python 2 compatible)
    report_paths = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f == "analysis_report.dat":
                report_paths.append(os.path.join(root, f))

    if not report_paths:
        print("Warning: No analysis_report.dat found in {}".format(folder_path))
        return None

    for path in report_paths:
        if "output_analysis" in path:
            return path

    return report_paths[0]

def rename_with_rwp(folder_path, rwp_value):
    folder_dir = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)

    rwp_str = "{:.2f}".format(rwp_value).replace('.', 'p')

    new_name = "{}_RWP_{}".format(folder_name, rwp_str)
    new_path = os.path.join(folder_dir, new_name)

    os.rename(folder_path, new_path)
    print("Renamed: {} -> {}".format(folder_path, new_path))

    return new_path

def get_folder_groups(folders):
    folder_groups = {}

    for folder in folders:
        match = re.match(r'(model_.*?)_(\d+)$', folder)
        if match:
            base_name = match.group(1)  # e.g., "model_pbso4_0P03"
            if base_name not in folder_groups:
                folder_groups[base_name] = []
            folder_groups[base_name].append(folder)

    return folder_groups

def main():
    current_dir = os.getcwd()

    print("=" * 70)
    print("CNN Model Best Performance Finder")
    print("=" * 70)
    print("This script will:")
    print("1. Find all model_* folders in the current directory")
    print("2. Extract the CNN_ML Rwp value from each model's analysis_report.dat")
    print("3. Keep the 5 models with lowest Rwp values and rename them to include their Rwp")
    print("4. Delete the remaining models\n")

    model_folders = [d for d in os.listdir(current_dir)
                     if os.path.isdir(os.path.join(current_dir, d)) and d.startswith("model_")]

    if not model_folders:
        print("No model folders found in the current directory.")
        return

    print("Found {} model folders".format(len(model_folders)))

    folder_groups = get_folder_groups(model_folders)

    if not folder_groups:
        print("No folders matching the expected naming pattern (model_*_NUMBER)")
        return

    print("Grouped into {} base name patterns".format(len(folder_groups)))

    for base_name, folders in folder_groups.items():
        print("\nProcessing {} folders with base name '{}'".format(len(folders), base_name))

        folder_rwp_map = {}

        for folder in folders:
            folder_path = os.path.join(current_dir, folder)
            report_path = find_analysis_report(folder_path)
            if report_path:
                rwp = extract_rwp_from_report(report_path)
                folder_rwp_map[folder] = rwp
                print("Folder: {}, Rwp: {:.2f}".format(folder, rwp))
            else:
                folder_rwp_map[folder] = float('inf')

        sorted_folders = sorted(folder_rwp_map.items(), key=lambda x: x[1])

        folders_to_keep = sorted_folders[:min(5, len(sorted_folders))]
        folders_to_delete = sorted_folders[min(5, len(sorted_folders)):]

        kept_folders = []
        for folder, rwp in folders_to_keep:
            folder_path = os.path.join(current_dir, folder)
            if rwp < float('inf'):  # Only rename if we have a valid Rwp
                new_path = rename_with_rwp(folder_path, rwp)
                kept_folders.append(os.path.basename(new_path))
            else:
                kept_folders.append(folder)

        for folder, _ in folders_to_delete:
            folder_path = os.path.join(current_dir, folder)
            print("Deleting folder: {}".format(folder))
            shutil.rmtree(folder_path)

        print("\nSummary for base name '{}':".format(base_name))
        print("Kept {} folders with lowest Rwp values:".format(len(kept_folders)))
        for folder in kept_folders:
            print("  - {}".format(folder))
        print("Deleted {} folders with higher Rwp values.".format(len(folders_to_delete)))

if __name__ == "__main__":
    try:
        main()
        print("\nScript completed successfully!")
    except Exception as e:
        print("\nError: {}".format(str(e)))
        sys.exit(1)
