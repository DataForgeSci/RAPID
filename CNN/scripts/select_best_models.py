#!/usr/bin/env python3


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
                print(f"Found NA/N/A value for CNN_ML Rwp in {report_path}")
                return float('inf')
            
            for line in lines:
                line = line.strip()
                if line.startswith('CNN_ML'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            if parts[2].lower() in ('na', 'n/a', '--', 'none', 'null'):
                                print(f"Found non-numeric value '{parts[2]}' for CNN_ML Rwp in {report_path}")
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
                                    print(f"Found non-numeric value '{parts[rwp_index]}' for CNN_ML Rwp in {report_path}")
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
                    
        print(f"Warning: Could not extract Rwp from {report_path}")
        return float('inf')
    except Exception as e:
        print(f"Error processing {report_path}: {str(e)}")
        return float('inf')

def find_analysis_report(folder_path):
    common_patterns = [
        os.path.join(folder_path, "refinement_result", "*", "*", "Rietveld_Refinement", "output_analysis", "analysis_report.dat"),
        os.path.join(folder_path, "refinement_result", "*", "Rietveld_Refinement", "output_analysis", "analysis_report.dat"),
        os.path.join(folder_path, "**", "output_analysis", "analysis_report.dat")
    ]
    
    for pattern in common_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    
    report_paths = glob.glob(os.path.join(folder_path, "**", "analysis_report.dat"), recursive=True)
    
    if not report_paths:
        print(f"Warning: No analysis_report.dat found in {folder_path}")
        return None
    
    for path in report_paths:
        if "output_analysis" in path:
            return path
    
    return report_paths[0]

def rename_with_rwp(folder_path, rwp_value):
    folder_dir = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)
    
    rwp_str = f"{rwp_value:.2f}".replace('.', 'p')
    
    new_name = f"{folder_name}_RWP_{rwp_str}"
    new_path = os.path.join(folder_dir, new_name)
    
    os.rename(folder_path, new_path)
    print(f"Renamed: {folder_path} -> {new_path}")
    
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
    
    print(f"Found {len(model_folders)} model folders")
    
    folder_groups = get_folder_groups(model_folders)
    
    if not folder_groups:
        print("No folders matching the expected naming pattern (model_*_NUMBER)")
        return
    
    print(f"Grouped into {len(folder_groups)} base name patterns")
    
    for base_name, folders in folder_groups.items():
        print(f"\nProcessing {len(folders)} folders with base name '{base_name}'")
        
        folder_rwp_map = {}
        
        for folder in folders:
            folder_path = os.path.join(current_dir, folder)
            report_path = find_analysis_report(folder_path)
            if report_path:
                rwp = extract_rwp_from_report(report_path)
                folder_rwp_map[folder] = rwp
                print(f"Folder: {folder}, Rwp: {rwp:.2f}")
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
            print(f"Deleting folder: {folder}")
            shutil.rmtree(folder_path)
        
        print(f"\nSummary for base name '{base_name}':")
        print(f"Kept {len(kept_folders)} folders with lowest Rwp values:")
        for folder in kept_folders:
            print(f"  - {folder}")
        print(f"Deleted {len(folders_to_delete)} folders with higher Rwp values.")

if __name__ == "__main__":
    try:
        main()
        print("\nScript completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)