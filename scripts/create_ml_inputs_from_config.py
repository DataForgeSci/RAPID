# create_ml_inputs_from_config.py
import os
import sys
import re

def create_ml_inputs_files(count, base_file_path):
    try:
        with open(base_file_path, 'r') as f:
            content = f.readlines()
        
        model_line_idx = None
        base_model_name = None
        
        for i, line in enumerate(content):
            if line.strip().startswith('y;'):
                model_line_idx = i
                parts = line.strip().split(';')
                if len(parts) > 1:
                    full_model_name = parts[1]
                    base_model_name = re.sub(r'_\d+$', '', full_model_name)
                break
        
        if not base_model_name:
            print("Error: Could not find model name in template file")
            return False
            
        output_dir = os.path.dirname(base_file_path)
        
        for i in range(1, count + 1):
            new_file_path = os.path.join(output_dir, f"ML_inputs_{i}.txt")
            
            new_content = content.copy()
            if model_line_idx is not None:
                new_content[model_line_idx] = f"y;{base_model_name}_{i}\n"
            
            with open(new_file_path, 'w') as f:
                f.writelines(new_content)
                
            print(f"Created {new_file_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating ML_inputs files: {e}")
        return False

def get_ml_inputs_count(config_file_path):
    try:
        with open(config_file_path, 'r') as f:
            lines = f.readlines()
        
        line_45_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('45>'):
                line_45_idx = i
                break
                
        if line_45_idx >= 0 and line_45_idx + 1 < len(lines):
            count_str = lines[line_45_idx + 1].strip()
            try:
                count = int(count_str)
                return count
            except ValueError:
                print(f"Warning: Invalid count value '{count_str}' in config file")
                return 1
        
        return 1  # Default value
    
    except Exception as e:
        print(f"Error reading config file: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_ml_inputs_from_config.py <config_file_path> <ml_inputs_template_path>")
        sys.exit(1)
        
    config_file_path = sys.argv[1]
    ml_inputs_template_path = sys.argv[2]
    
    count = get_ml_inputs_count(config_file_path)
    print(f"Creating {count} ML_inputs files")
    
    success = create_ml_inputs_files(count, ml_inputs_template_path)
    
    if not success:
        print("Failed to create ML_inputs files")
        sys.exit(1)