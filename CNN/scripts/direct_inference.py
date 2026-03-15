#!/usr/bin/env python
import os
import sys
import argparse
import shutil
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from material_identifier import (
    load_material_catalog,
    detect_dat_format,
    read_xrd_pattern,
    predict_parameters,
    IDENTIFIED_DIR
)
def parse_arguments():
    parser = argparse.ArgumentParser(description="Direct inference with specified material model")
    parser.add_argument("--file", required=True, help="Path to XRD pattern file")
    parser.add_argument("--material", required=True, help="Material name (e.g., CeO2, pbso4, tbbaco)")
    return parser.parse_args()
def main():
    args = parse_arguments()
    input_file = args.file
    material_name = args.material
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    catalog = load_material_catalog()
    print(f"Loaded catalog with {len(catalog)} materials")
    if material_name not in catalog:
        print(f"Error: Material '{material_name}' not found in catalog")
        print(f"Available materials: {list(catalog.keys())}")
        sys.exit(1)
    dat_format = detect_dat_format(input_file)
    ttheta_values, intensity_values = read_xrd_pattern(input_file, dat_format)
    if not ttheta_values or not intensity_values:
        print("Error: Failed to read pattern data")
        sys.exit(1)
    print(f"\n**DIRECT INFERENCE**: Using {material_name} model (skipping identification)")
    output_dir = os.path.join(IDENTIFIED_DIR, f"unknown_{material_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    source_dir = os.path.join(output_dir, "source_files")
    os.makedirs(source_dir, exist_ok=True)
    shutil.copy2(input_file, os.path.join(source_dir, os.path.basename(input_file)))
    print("\n=== Starting Parameter Prediction and Refinement ===")
    predicted_params = predict_parameters(material_name, ttheta_values, intensity_values, catalog, output_dir)
    print("\nDirect inference completed successfully!")
    print(f"Results saved to: {output_dir}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
