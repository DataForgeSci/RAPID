# update_ml_input.py
import sys
import re

def update_ml_input(file_path, dataset_name):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if line.strip().startswith("N;"):
            lines[i] = f"N;{dataset_name}\n"
            break
    
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print(f"Successfully updated {file_path} with dataset {dataset_name}")
    return True

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python update_ml_input.py <file_path> <dataset_name>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    dataset_name = sys.argv[2]
    
    success = update_ml_input(file_path, dataset_name)
    if not success:
        print(f"Failed to update {file_path}")
        sys.exit(1)