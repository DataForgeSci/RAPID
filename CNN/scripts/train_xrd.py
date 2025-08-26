import os
import sys
import glob
import shutil
import time
from datetime import datetime
import re
import glob

import numpy as np
import torch
import torch.nn as nn
import matplotlib
# If running headless (no GUI), uncomment:
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import CNN, get_loader, train_loop, parse_info_file
from feature_importance import run_feature_importance_analysis
from utils import CNN, get_loader, train_loop, parse_info_file, parse_dat_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_directories():
    dirs = [
        'data/train_data',
        'data/experimental_data',
        'saved_models'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_model(model_path, output_dim=8, param_info=None, omit_background=False):

    if param_info:
        if omit_background and param_info.get('has_background', False):
            output_dim = param_info['total_params'] - 1
            print(f"Adjusted output_dim to {output_dim} due to omit_background=True")
        else:
            output_dim = param_info['total_params']
    
    model = CNN(output_dim=output_dim, param_info=param_info).to(device)
    
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.eval()
    return model

def train_subfolder(row_path, model, base_model_name, omit_background=False, use_digit_scaling=True, use_adaptive_scaling=False):

    from utils import get_loader, train_loop, WeightedSmoothL1Loss

    output_file = os.path.join('data', 'train_data', 'cnn_training_results.dat')
    log_file = open(output_file, 'w')
    original_stdout = sys.stdout

    try:
        sys.stdout = Logger(log_file)

        if not os.path.exists(row_path):
            print(f"Data file not found: {row_path}. Skipping training.")
            # Return empty times if we didn't train
            return model, "", "", None, None

        print(f"\nUsing training data: {row_path}")
        print(f"(Base model used: {base_model_name})")
        print(f"Omit background parameter: {omit_background}")
        print(f"Digit-based parameter scaling: {use_digit_scaling}")
        print(f"Adaptive reference-based scaling: {use_adaptive_scaling}")

        dataset_name = os.path.basename(os.path.dirname(row_path))
        print(f"Dataset name: {dataset_name}")

        base_name = os.path.basename(row_path).split('_row_param')[0]
        info_file = os.path.join(os.path.dirname(row_path), f"{base_name}_2theta_param_info.dat")
        if os.path.exists(info_file):
            print(f"Found info file: {info_file}")
        else:
            info_file = None
            print(f"No info file found, will use default parameters")

        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\nTraining start time: {start_time}")

        train_loader, val_loader, param_info = get_loader(row_path, 
                                                         omit_background=omit_background, 
                                                         info_file=info_file,
                                                         use_digit_scaling=use_digit_scaling,
                                                         use_adaptive_scaling=use_adaptive_scaling)
        
        for _, y in train_loader:
            actual_param_count = y.size(1)
            break
        
        current_output_dim = getattr(model, 'output_dim', model.lin.out_features)
        
        model_has_bg = getattr(model, 'has_background', None)
        
        needs_recreate = (current_output_dim != actual_param_count) or (model_has_bg and omit_background)
        
        if needs_recreate:
            reason = "dimensions don't match" if current_output_dim != actual_param_count else "background handling mismatch"
            print(f"Model needs to be recreated because {reason}:")
            print(f"  - Model output dimension: {current_output_dim}, Data dimension: {actual_param_count}")
            print(f"  - Model has_background: {model_has_bg}, omit_background: {omit_background}")
            
            modified_param_info = param_info
            if omit_background and param_info and param_info['has_background']:
                modified_param_info = {key: value.copy() if isinstance(value, list) else value 
                                      for key, value in param_info.items()}
                modified_param_info['has_background'] = False
                print("Modified param_info to set has_background=False for CNN model initialization")
            
            model = CNN(output_dim=actual_param_count, param_info=modified_param_info).to(device)
            
            if base_model_name and base_model_name != "Scratch":
                print(f"Attempting to initialize with weights from previous model where possible")
                
                prev_model_path = glob.glob(os.path.join('saved_models', base_model_name, f"{base_model_name}.pth"))
                if prev_model_path:
                    try:
                        prev_state_dict = torch.load(prev_model_path[0], weights_only=True)
                        
                        own_state = model.state_dict()
                        for name, param in prev_state_dict.items():
                            if name in own_state:
                                if 'out.' in name:
                                    continue
                                if own_state[name].shape == param.shape:
                                    own_state[name].copy_(param)
                                    print(f"Copied weights for: {name}")
                                else:
                                    print(f"Skipped due to shape mismatch: {name}")
                                    
                        print(f"Initialized compatible weights from previous model")
                    except Exception as e:
                        print(f"Error initializing from previous model: {e}")
        else:
            print(f"Model output dimension matches dataset ({actual_param_count} parameters)")
        
        print("Using optimizer: Adam(lr=1e-3, weight_decay=1e-4)")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        print("\nStarting training with enhanced display...\n")
        
        param_names = None
        if param_info:
            param_types = param_info['param_types']
            param_names = []
            
            for i, param_type in enumerate(param_types):
                if param_type == 'zero':
                    param_names.append("Zero")
                elif param_type == 'background':
                    if not omit_background:  # Skip background if omitting
                        param_names.append("Background")
                elif 'lattice parameter' in param_type:
                    lattice_label = param_type.split()[-1] if len(param_type.split()) > 2 else ""
                    param_names.append(f"Lattice{' ' + lattice_label if lattice_label else ''}")
                elif 'biso' in param_type:
                    atom_type = param_type.split('_')[1] if '_' in param_type else ""
                    param_names.append(f"Biso{' ' + atom_type if atom_type else ''}")
                elif 'scale factor' in param_type:
                    param_names.append("Scale")
                elif 'u parameter' in param_type:
                    param_names.append("U")
                elif 'v parameter' in param_type:
                    param_names.append("V")
                elif 'w parameter' in param_type:
                    param_names.append("W")
                else:
                    param_names.append(f"Param{i+1}")
        
        if not param_names:
            if omit_background:
                param_names = ['Zero', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
            else:
                param_names = ['Zero', 'Background', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
            
            # If we have more parameters than names, add generic names
            while len(param_names) < actual_param_count:
                param_names.append(f'Param{len(param_names)+1}')
        
        if len(param_names) != actual_param_count:
            print(f"Adjusting parameter names to match actual parameter count ({actual_param_count})")
            
            if len(param_names) < actual_param_count:
                for i in range(len(param_names), actual_param_count):
                    param_names.append(f"Param{i+1}")
            else:
                param_names = param_names[:actual_param_count]
                
            print(f"Updated parameter names: {param_names}")
        
        total_epochs = 100
        
        metrics_history = train_loop(
            model, train_loader, val_loader, optimizer,
            epochs=total_epochs, 
            criterion=None,  # Let train_loop create appropriate weighted loss
            m=nn.Identity(), 
            verbose=True, 
            param_names=param_names,
            dataset_name=dataset_name,
            patience=7,  # Research-based patience
            min_improvement=1e-4,  # Research-based improvement threshold
            use_weighted_loss=True,  # Explicitly enable weighted loss
            zero_pct_error_threshold=0.3,
            omit_background=omit_background,
            param_info=param_info  # Pass parameter info for loss weighting
        )

        end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\nTraining completed.")
        print(f"Training start time: {start_time}")
        print(f"Training end time: {end_time}")
        
        metrics_file = os.path.join('data', 'train_data', 'training_metrics.npz')
        if os.path.exists(metrics_file):
            try:
                os.remove(metrics_file)
                print(f"Removed temporary metrics file: {metrics_file}")
            except Exception as e:
                print(f"Warning: Could not remove metrics file: {e}")
        
        return model, start_time, end_time, metrics_history, param_info

    finally:
        sys.stdout = original_stdout
        log_file.close()

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.__stdout__
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# def train_subfolder(row_path, model, base_model_name, omit_background=False):
#     from utils import get_loader, train_loop, WeightedSmoothL1Loss

#     output_file = os.path.join('data', 'train_data', 'cnn_training_results.dat')
#     log_file = open(output_file, 'w')
#     original_stdout = sys.stdout

#     try:
#         sys.stdout = Logger(log_file)

#         if not os.path.exists(row_path):
#             print(f"Data file not found: {row_path}. Skipping training.")
#             # Return empty times if we didn't train
#             return model, "", "", None, None

#         print(f"\nUsing training data: {row_path}")
#         print(f"(Base model used: {base_model_name})")
#         print(f"Omit background parameter: {omit_background}")

#         # Extract dataset name from row_path
#         dataset_name = os.path.basename(os.path.dirname(row_path))
#         print(f"Dataset name: {dataset_name}")

#         # Try to find the info file
#         base_name = os.path.basename(row_path).split('_row_param')[0]
#         info_file = os.path.join(os.path.dirname(row_path), f"{base_name}_2theta_param_info.dat")
#         if os.path.exists(info_file):
#             print(f"Found info file: {info_file}")
#         else:
#             info_file = None
#             print(f"No info file found, will use default parameters")

#         # Start time
#         start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#         print(f"\nTraining start time: {start_time}")

#         # Load the data with parameter info
#         train_loader, val_loader, param_info = get_loader(row_path, omit_background=omit_background, info_file=info_file)
        
#         # Check if the model's output dimension matches the data
#         for _, y in train_loader:
#             actual_param_count = y.size(1)
#             break
        
#         current_output_dim = getattr(model, 'output_dim', model.lin.out_features)
#         if current_output_dim != actual_param_count:
#             print(f"Model output dimension ({current_output_dim}) doesn't match dataset ({actual_param_count})")
#             print(f"Creating a new model with the correct output dimension")
            
#             # Create a new model with the correct output dimension and parameter info
#             model = CNN(output_dim=actual_param_count, param_info=param_info).to(device)
            
#             # If we had a pre-trained model, try to copy compatible weights
#             if base_model_name and base_model_name != "Scratch":
#                 print(f"Attempting to initialize with weights from previous model where possible")
                
#                 # Try to load the previous model's weights
#                 prev_model_path = glob.glob(os.path.join('saved_models', base_model_name, f"{base_model_name}.pth"))
#                 if prev_model_path:
#                     try:
#                         prev_state_dict = torch.load(prev_model_path[0], weights_only=True)
                        
#                         # Copy compatible weights
#                         own_state = model.state_dict()
#                         for name, param in prev_state_dict.items():
#                             if name in own_state:
#                                 # Skip output layers that might have different dimensions
#                                 if 'out.' in name:
#                                     continue
#                                 # Copy parameter if shapes match
#                                 if own_state[name].shape == param.shape:
#                                     own_state[name].copy_(param)
#                                     print(f"Copied weights for: {name}")
#                                 else:
#                                     print(f"Skipped due to shape mismatch: {name}")
                                    
#                         print(f"Initialized compatible weights from previous model")
#                     except Exception as e:
#                         print(f"Error initializing from previous model: {e}")
#         else:
#             print(f"Model output dimension matches dataset ({actual_param_count} parameters)")
        
#         # Use Adam optimizer for better stability
#         print("Using optimizer: Adam(lr=1e-3, weight_decay=1e-4)")
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
#         print("\nStarting training with enhanced display...\n")
        
#         # Get parameter names from param_info if available
#         param_names = None
#         if param_info:
#             param_types = param_info['param_types']
#             param_names = []
            
#             # Convert parameter types to readable names
#             for i, param_type in enumerate(param_types):
#                 if param_type == 'zero':
#                     param_names.append("Zero")
#                 elif param_type == 'background':
#                     param_names.append("Background")
#                 elif 'lattice parameter' in param_type:
#                     # Get lattice parameter label (a, b, c, etc.)
#                     lattice_label = param_type.split()[-1] if len(param_type.split()) > 2 else ""
#                     param_names.append(f"Lattice{' ' + lattice_label if lattice_label else ''}")
#                 elif 'biso' in param_type:
#                     # Get atom type if available (Biso_Si, Biso_O, etc.)
#                     atom_type = param_type.split('_')[1] if '_' in param_type else ""
#                     param_names.append(f"Biso{' ' + atom_type if atom_type else ''}")
#                 elif 'scale factor' in param_type:
#                     param_names.append("Scale")
#                 elif 'u parameter' in param_type:
#                     param_names.append("U")
#                 elif 'v parameter' in param_type:
#                     param_names.append("V")
#                 elif 'w parameter' in param_type:
#                     param_names.append("W")
#                 else:
#                     param_names.append(f"Param{i+1}")
        
#         # If no param_names, use defaults
#         if not param_names:
#             if omit_background:
#                 param_names = ['Zero', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
#             else:
#                 param_names = ['Zero', 'Background', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
            
#             # If we have more parameters than names, add generic names
#             while len(param_names) < actual_param_count:
#                 param_names.append(f'Param{len(param_names)+1}')
        
#         # Verify that param_names has the right length and fix if needed
#         if len(param_names) != actual_param_count:
#             print(f"Adjusting parameter names to match actual parameter count ({actual_param_count})")
            
#             if len(param_names) < actual_param_count:
#                 # Add generic names for extra parameters
#                 for i in range(len(param_names), actual_param_count):
#                     param_names.append(f"Param{i+1}")
#             else:
#                 # Truncate to match actual count
#                 param_names = param_names[:actual_param_count]
                
#             print(f"Updated parameter names: {param_names}")
        
#         total_epochs = 2000
        
#         # Run training loop with enhanced display
#         metrics_history = train_loop(
#             model, train_loader, val_loader, optimizer,
#             epochs=total_epochs, 
#             criterion=None,  # Let train_loop create appropriate weighted loss
#             m=nn.Identity(), 
#             verbose=True, 
#             param_names=param_names,
#             dataset_name=dataset_name,
#             patience=7,  # Research-based patience
#             min_improvement=1e-4,  # Research-based improvement threshold
#             use_weighted_loss=True,  # Explicitly enable weighted loss
#             zero_pct_error_threshold=0.3,
#             omit_background=omit_background,
#             param_info=param_info  # Pass parameter info for loss weighting
#         )

#         # End time
#         end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#         print(f"\nTraining completed.")
#         print(f"Training start time: {start_time}")
#         print(f"Training end time: {end_time}")
        
#         # Save metrics for plotting later
#         metrics_file = os.path.join('data', 'train_data', 'training_metrics.npz')
#         if os.path.exists(metrics_file):
#             try:
#                 os.remove(metrics_file)
#                 print(f"Removed temporary metrics file: {metrics_file}")
#             except Exception as e:
#                 print(f"Warning: Could not remove metrics file: {e}")
        
#         return model, start_time, end_time, metrics_history, param_info

#     finally:
#         sys.stdout = original_stdout
#         log_file.close()

def post_training_logic(model, model_name_prefix, used_row_path, do_refine_files, 
                       base_folder_name="", start_time="", end_time="", metrics_history=None,
                       omit_background=False, param_info=None, use_digit_scaling=True, use_adaptive_scaling=False):

    source_log = os.path.join('data', 'train_data', 'cnn_training_results.dat')
    if not os.path.exists(source_log):
        print(f"Warning: Training log not found at {source_log}")
        return None

    new_folder_name = model_name_prefix
    new_folder_path = os.path.join('saved_models', new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    folder_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    training_result_dir = os.path.join(new_folder_path, 'training_result')
    os.makedirs(training_result_dir, exist_ok=True)

    dataset_name = os.path.basename(os.path.dirname(used_row_path))
    dataset_result_dir = os.path.join(training_result_dir, dataset_name)
    os.makedirs(dataset_result_dir, exist_ok=True)

    target_log_name = f"cnn_training_results_{folder_time}.dat"
    target_log = os.path.join(dataset_result_dir, target_log_name)
    
    try:
        shutil.copy2(source_log, target_log)
    except Exception as e:
        print(f"Warning: Could not copy training log: {e}")
        target_log = None

    if param_info:
        param_info_path = os.path.join(dataset_result_dir, f"parameter_info_{folder_time}.dat")
        try:
            with open(param_info_path, 'w') as f:
                f.write("Parameter Information:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total parameters: {param_info['total_params']}\n")
                f.write("Parameter counts by type:\n")
                for param_type, count in param_info['param_counts'].items():
                    if count > 0:
                        f.write(f"  {param_type}: {count}\n")
                f.write("\nParameter types in order:\n")
                for i, param_type in enumerate(param_info['param_types']):
                    f.write(f"  {i+1}: {param_type}\n")
                f.write(f"\nBackground parameter: {'Present' if param_info['has_background'] else 'Absent/Omitted'}\n")
            print(f"Saved parameter info to {param_info_path}")
        except Exception as e:
            print(f"Warning: Could not save parameter info: {e}")

    final_pth = os.path.join(new_folder_path, f"{model_name_prefix}.pth")
    torch.save(model.state_dict(), final_pth)
    print(f"\nSaved/updated model => {final_pth}")

    if metrics_history and 'train_loss' in metrics_history and len(metrics_history['train_loss']) > 0:
        epochs = range(1, len(metrics_history['train_loss'])+1)
        
        train_loss = metrics_history['train_loss']
        val_loss = metrics_history['val_loss']
        train_mae = metrics_history['train_mae']
        val_mae = metrics_history['val_mae']
        train_rmse = metrics_history['train_rmse']
        val_rmse = metrics_history['val_rmse']
        
        plt.figure(figsize=(10,6))
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CNN Train/Validation Loss vs. Epoch')
        plt.grid(True)
        plt.legend()
        loss_plot_path = os.path.join(dataset_result_dir, f"cnn_training_loss_vs_epoch_{folder_time}.png")
        plt.savefig(loss_plot_path)
        plt.close()
        
        plt.figure(figsize=(10,6))
        plt.plot(epochs, train_mae, label='Train MAE')
        plt.plot(epochs, val_mae, label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('CNN Train/Validation MAE vs. Epoch')
        plt.grid(True)
        plt.legend()
        mae_plot_path = os.path.join(dataset_result_dir, f"cnn_training_mae_vs_epoch_{folder_time}.png")
        plt.savefig(mae_plot_path)
        plt.close()
        
        plt.figure(figsize=(10,6))
        plt.plot(epochs, train_rmse, label='Train RMSE')
        plt.plot(epochs, val_rmse, label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('Root Mean Squared Error')
        plt.title('CNN Train/Validation RMSE vs. Epoch')
        plt.grid(True)
        plt.legend()
        rmse_plot_path = os.path.join(dataset_result_dir, f"cnn_training_rmse_vs_epoch_{folder_time}.png")
        plt.savefig(rmse_plot_path)
        plt.close()
        
        print(f"\nSaved loss/MAE/RMSE plots to {dataset_result_dir}")
    else:
        print("No metrics history available for plotting or metrics history is empty.")
        
        # Parse the log file for metrics if no history provided
        train_loss, val_loss = [], []
        if target_log and os.path.exists(target_log):
            with open(target_log, 'r') as f:
                for line in f:
                    line = line.strip()
                    if "train acc/loss" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                train_loss.append(float(parts[3]))
                            except:
                                pass
                    elif "test acc/loss" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                val_loss.append(float(parts[3]))
                            except:
                                pass
        
        if train_loss and val_loss:
            epochs = range(1, len(train_loss)+1)
            plt.figure(figsize=(10,6))
            plt.plot(epochs, train_loss, label='Train Loss', linestyle='-', color='blue')
            plt.plot(epochs, val_loss, label='Validation Loss', linestyle='-', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('CNN Train/Validation Loss vs. Epoch')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(dataset_result_dir, f"cnn_training_loss_vs_epoch_{folder_time}.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\nSaved loss plot to {dataset_result_dir}")

    td_file = os.path.join(new_folder_path, 'train_data_log.dat')
    old_log = ""
    if base_folder_name:
        old_folder_path = os.path.join('saved_models', base_folder_name)
        old_td = os.path.join(old_folder_path, 'train_data_log.dat')
        if os.path.exists(old_td):
            with open(old_td, 'r') as f:
                old_log = f.read()

    with open(td_file, 'w') as f:
        f.write(old_log)
        f.write(f"model name: {model_name_prefix}\n")
        if base_folder_name:
            f.write(f"used base model: {base_folder_name}\n")
        else:
            f.write("used base model: None\n")
        f.write(f"training data path: {used_row_path}\n")
        f.write(f"omit background parameter: {omit_background}\n")
        f.write(f"digit-based parameter scaling: {use_digit_scaling}\n")
        f.write(f"adaptive reference-based scaling: {use_adaptive_scaling}\n")
        
        if param_info:
            f.write(f"crystal structure: ")
            lattice_count = param_info['param_counts']['lattice parameter']
            if lattice_count == 1:
                f.write("cubic (1 lattice parameter)\n")
            elif lattice_count == 2:
                f.write("tetragonal or hexagonal (2 lattice parameters)\n")
            elif lattice_count == 3:
                f.write("orthorhombic (3 lattice parameters)\n")
            elif lattice_count == 4:
                f.write("monoclinic (4 lattice parameters)\n")
            elif lattice_count == 6:
                f.write("triclinic (6 lattice parameters)\n")
            else:
                f.write(f"unknown ({lattice_count} lattice parameters)\n")
            
            f.write(f"total parameters: {param_info['total_params']}\n")
            f.write(f"Biso parameters: {param_info['param_counts']['biso']}\n")
        
        if start_time:
            f.write(f"start time: {start_time}\n")
        f.write(f"time finished: {folder_time}\n")
        f.write("\n")

    if do_refine_files:
        do_refinement(do_refine_files, model_name_prefix, dataset_name, omit_background, param_info,
                     use_digit_scaling, use_adaptive_scaling)

    try:
        if os.path.exists(source_log):
            os.remove(source_log)
            print("Cleaned up temporary training log file.")
    except Exception as e:
        print(f"Warning: Could not remove source log file: {e}")

    return model_name_prefix

def do_refinement(dat_files, model_folder, dataset_name=None, omit_background=False, param_info=None, 
                 use_digit_scaling=True, use_adaptive_scaling=False):

    if not dat_files or dat_files[0].lower() == "n":
        print("No .dat files for refinement. Skipping.")
        return
    if not model_folder:
        print("No model folder. Skipping refinement.")
        return

    chosen_folder_path = os.path.join('saved_models', model_folder)
    if not os.path.isdir(chosen_folder_path):
        print(f"Refinement aborted: folder '{model_folder}' not found.")
        return

    chosen_model_pth = os.path.join(chosen_folder_path, f"{model_folder}.pth")
    if not os.path.exists(chosen_model_pth):
        print(f"Refinement aborted: .pth file not found in '{chosen_model_pth}'.")
        return

    print(f"\nRefinement will use model => {chosen_model_pth}")
    refine_model = load_model(chosen_model_pth, param_info=param_info, omit_background=omit_background)
    
    output_dim = refine_model.output_dim
    print(f"Model loaded with output_dim={output_dim}, omit_background={omit_background}")

    dat_format = "Si.dat"  # Default format
    try:
        input_file = sys.argv[1] if len(sys.argv) > 1 else "ML_inputs.txt"
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip().startswith("# 12>"):
                    # The format should be on the next line
                    if i + 1 < len(lines):
                        dat_format = lines[i + 1].strip()
                        break
        print(f"Read DAT file format '{dat_format}' from {input_file}")
    except Exception as e:
        print(f"Warning: Could not read DAT format from {input_file}: {e}")
        print("Using default Si.dat format.")

    print(f"Using DAT file format structure: {dat_format}")

    # training_tth_min = 5.0
    # training_tth_max = 110.0
    # training_step = 0.3
    
    # if param_info and 'two_theta_range' in param_info:
    #     training_tth_min = param_info['two_theta_range']['min']
    #     training_tth_max = param_info['two_theta_range']['max']
    #     training_step = param_info['two_theta_range']['step']
    #     print(f"Using training data two-theta range: {training_tth_min} to {training_tth_max} with step {training_step}")
    # else:
    #     print(f"Using default two-theta range: {training_tth_min} to {training_tth_max} with step {training_step}")
    
    import re
    import glob

    training_tth_min = 5.0  # Default fallback
    training_tth_max = 110.0
    training_step = 0.3

    try:
        info_file = None
        
        if dataset_name:
            print(f"Looking for info file in dataset: {dataset_name}")
            possible_locations = [
                os.path.join('data', 'train_data', dataset_name, "*_2theta_param_info.dat"),
                os.path.join('data', 'train_data', dataset_name, "*simulated_data_2theta_param_info.dat"),
                os.path.join('saved_models', model_folder, 'training_result', dataset_name, "*_2theta_param_info.dat")
            ]
            
            for location in possible_locations:
                matching_files = glob.glob(location)
                if matching_files:
                    info_file = matching_files[0]
                    print(f"Found info file at: {info_file}")
                    break
        
        if not info_file:
            search_patterns = [
                "data/**/*_2theta_param_info.dat",
                "data/**/*simulated_data_2theta_param_info.dat",
                f"saved_models/{model_folder}/**/*_2theta_param_info.dat"
            ]
            
            for pattern in search_patterns:
                matching_files = glob.glob(pattern, recursive=True)
                if matching_files:
                    info_file = matching_files[0]
                    print(f"Found info file through recursive search: {info_file}")
                    break
        
        if info_file:
            with open(info_file, 'r') as f:
                first_line = f.readline().strip()
                
                # Parse the line (format: Initial=X.X, Final=Y.Y, Step=Z.Z)
                initial_match = re.search(r'Initial=(\d+\.?\d*)', first_line)
                final_match = re.search(r'Final=(\d+\.?\d*)', first_line)
                step_match = re.search(r'Step=(\d+\.?\d*)', first_line)
                
                if initial_match and final_match and step_match:
                    training_tth_min = float(initial_match.group(1))
                    training_tth_max = float(final_match.group(1))
                    training_step = float(step_match.group(1))
                    print(f"Found two-theta range in info file: {training_tth_min} to {training_tth_max} with step {training_step}")
                else:
                    print(f"Could not parse two-theta range from line: {first_line}")
        else:
            print("No two-theta info file found through any search method")
            
    except Exception as e:
        print(f"Error extracting two-theta range: {e}")
        import traceback
        traceback.print_exc()

    if training_tth_min == 5.0 and param_info and 'two_theta_range' in param_info:
        training_tth_min = param_info['two_theta_range']['min']
        training_tth_max = param_info['two_theta_range']['max']
        training_step = param_info['two_theta_range']['step']
        print(f"Using two-theta range from param_info: {training_tth_min} to {training_tth_max} with step {training_step}")

    print(f"Using two-theta range: {training_tth_min} to {training_tth_max} with step {training_step}")

    new_twotheta = np.arange(training_tth_min, training_tth_max + training_step, training_step)

    exp_folder = os.path.join('data', 'experimental_data')
    for fdat in dat_files:
        fdat = fdat.strip()
        if not fdat or fdat.lower() == "n":
            continue
        file_path = os.path.join(exp_folder, fdat)
        if not os.path.exists(file_path):
            print(f"Refine: file not found: {file_path}, skipping.")
            continue

        base_name = os.path.splitext(fdat)[0]
        
        # print(f"Parsing {fdat} using {dat_format} format structure")
        # ttheta_vals, intensity_vals = parse_dat_file(file_path, dat_format)

        # if not ttheta_vals:
        #     print(f"No valid data extracted from {fdat}, skipping.")
        #     continue

        # print(f"Found {len(ttheta_vals)} data points in DAT file")
        # print("First 5 data points from raw DAT file:")
        # for i in range(min(5, len(ttheta_vals))):
        #     print(f"  2-Theta: {ttheta_vals[i]:.3f}, Intensity: {intensity_vals[i]:.1f}")
        # print()

        # arr_tth = np.array(ttheta_vals)
        # arr_yobs = np.array(intensity_vals)
        # sort_idx = np.argsort(arr_tth)
        # arr_tth = arr_tth[sort_idx]
        # arr_yobs = arr_yobs[sort_idx]

        # print(f"Interpolating to {len(new_twotheta)} points using two-theta range: {training_tth_min} to {training_tth_max} with step {training_step}")
        # interp_y = np.interp(new_twotheta, arr_tth, arr_yobs)

        # print("First 5 data points after interpolation:")
        # for i in range(min(5, len(new_twotheta))):
        #     print(f"  2-Theta: {new_twotheta[i]:.3f}, Intensity: {interp_y[i]:.1f}")
        # print()

        # refine_res_dir = os.path.join(chosen_folder_path, 'refinement_result')
        # if dataset_name:
        #     refine_res_dir = os.path.join(refine_res_dir, dataset_name)
        # os.makedirs(refine_res_dir, exist_ok=True)

        # single_res_dir = os.path.join(refine_res_dir, base_name)
        # os.makedirs(single_res_dir, exist_ok=True)

        # info_path = os.path.join(single_res_dir, f"{base_name}_experimental_data_2theta_info.dat")
        # with open(info_path, 'w') as hf:
        #     hf.write(f"Initial={training_tth_min}, Final={training_tth_max}, Step={training_step}\n")

        print(f"Parsing {fdat} using {dat_format} format structure")
        ttheta_vals, intensity_vals = parse_dat_file(file_path, dat_format)

        if not ttheta_vals:
            print(f"No valid data extracted from {fdat}, skipping.")
            continue

        print(f"Found {len(ttheta_vals)} data points in DAT file")
        print("First 5 data points from raw DAT file:")
        for i in range(min(5, len(ttheta_vals))):
            print(f"  2-Theta: {ttheta_vals[i]:.3f}, Intensity: {intensity_vals[i]:.1f}")
        print()

        arr_tth = np.array(ttheta_vals)
        arr_yobs = np.array(intensity_vals)
        sort_idx = np.argsort(arr_tth)
        arr_tth = arr_tth[sort_idx]
        arr_yobs = arr_yobs[sort_idx]

        print(f"Interpolating to {len(new_twotheta)} points using two-theta range: {training_tth_min} to {training_tth_max} with step {training_step}")
        interp_y = np.interp(new_twotheta, arr_tth, arr_yobs)

        print("First 5 data points after interpolation:")
        for i in range(min(5, len(new_twotheta))):
            print(f"  2-Theta: {new_twotheta[i]:.3f}, Intensity: {interp_y[i]:.1f}")
        print()

        refine_res_dir = os.path.join(chosen_folder_path, 'refinement_result')
        if dataset_name:
            refine_res_dir = os.path.join(refine_res_dir, dataset_name)
        os.makedirs(refine_res_dir, exist_ok=True)

        single_res_dir = os.path.join(refine_res_dir, base_name)
        os.makedirs(single_res_dir, exist_ok=True)

        readable_dat_path = os.path.join(single_res_dir, f"{base_name}_readable_experimental_data.dat")
        with open(readable_dat_path, 'w') as rf:
            rf.write("# 2-Theta   Intensity\n")
            for tth, inten in zip(ttheta_vals, intensity_vals):
                rf.write(f"{tth:.4f}   {inten:.1f}\n")
        print(f"Raw experimental data saved to {readable_dat_path} for verification")

        info_path = os.path.join(single_res_dir, f"{base_name}_experimental_data_2theta_info.dat")
        with open(info_path, 'w') as hf:
            hf.write(f"Initial={training_tth_min}, Final={training_tth_max}, Step={training_step}\n")
      
        col_plot_path = os.path.join(single_res_dir, f"{base_name}_experimental_data_column_for_plot.dat")
        with open(col_plot_path, 'w') as hf:
            for tval, yval in zip(new_twotheta, interp_y):
                hf.write(f"{tval:.4f} {yval:.4f}\n")
        
        row_path = os.path.join(single_res_dir, f"{base_name}_experimental_data_row.dat")
        with open(row_path, 'w') as hf:
            row_line = " ".join(f"{val:.4f}" for val in interp_y)
            hf.write(row_line + "\n")

        intens = torch.FloatTensor(interp_y)
        max_val = torch.max(intens)
        intens /= max_val
        intens -= torch.mean(intens)
        intens = intens.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = refine_model(intens)

        param_names = []
        scaling_factors = None
        
        if param_info:
            if 'param_types' in param_info:
                param_types = param_info['param_types']
                
                for i, param_type in enumerate(param_types):
                    if param_type == 'zero':
                        param_names.append("Zero")
                    elif param_type == 'background':
                        if not omit_background:
                            param_names.append("Background")
                    elif 'lattice parameter' in param_type:
                        lattice_label = param_type.split()[-1] if len(param_type.split()) > 2 else ""
                        param_names.append(f"Lattice{' ' + lattice_label if lattice_label else ''}")
                    elif 'biso' in param_type:
                        atom_type = param_type.split('_')[1] if '_' in param_type else ""
                        param_names.append(f"Biso{' ' + atom_type if atom_type else ''}")
                    elif 'scale factor' in param_type:
                        param_names.append("Scale")
                    elif 'u parameter' in param_type:
                        param_names.append("U")
                    elif 'v parameter' in param_type:
                        param_names.append("V")
                    elif 'w parameter' in param_type:
                        param_names.append("W")
                    else:
                        param_names.append(f"Param{i+1}")
                
                if omit_background:
                    param_names = [name for name in param_names if name != "Background"]
                    
                print(f"Extracted parameter names from param_info: {param_names}")
            
            if 'scaling_factors' in param_info:
                scaling_factors = param_info['scaling_factors'].to(device)
                print(f"Using scaling factors from param_info: {scaling_factors}")
        
        if not param_names:
            print("Using default parameter names based on output dimension")
            if omit_background or output_dim == 7:
                param_names = ['Zero', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
            else:
                param_names = ['Zero', 'Background', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
        
        if scaling_factors is None:
            if use_digit_scaling:
                if omit_background or output_dim == 7:
                    scaling_factors = torch.FloatTensor([100, 1.0, 10.0, 100000, 1000, 1000, 1000]).to(device)
                else:
                    scaling_factors = torch.FloatTensor([100, 0.1, 1.0, 10.0, 100000, 1000, 1000, 1000]).to(device)
                print(f"Using default digit-based scaling factors: {scaling_factors}")

        if len(param_names) != output_dim:
            print(f"Adjusting parameter names list length from {len(param_names)} to match model output dimension ({output_dim})")
            if len(param_names) < output_dim:
                for i in range(len(param_names), output_dim):
                    param_names.append(f"Param{i+1}")
            else:
                param_names = param_names[:output_dim]
            print(f"Adjusted parameter names: {param_names}")
        
        if scaling_factors.size(0) != pred.size(1):
            print(f"Adjusting scaling factors count from {scaling_factors.size(0)} to match prediction dimension ({pred.size(1)})")
            if scaling_factors.size(0) < pred.size(1):
                padding = torch.ones(pred.size(1) - scaling_factors.size(0), device=device)
                scaling_factors = torch.cat([scaling_factors, padding])
            else:
                scaling_factors = scaling_factors[:pred.size(1)]
            print(f"Adjusted scaling factors: {scaling_factors}")

        pred = (pred / scaling_factors) / 10.0
        pred = pred.cpu().numpy()[0]

        refine_out = os.path.join(single_res_dir, f"{base_name}_refined_parameters.dat")

        print(f"\nRefinement of {fdat} with {model_folder}:")
        with open(refine_out, 'w') as rf:
            for name, val in zip(param_names, pred):
                line = f"{name:<12} = {val:12.6f}"
                print("  " + line)
                rf.write(line + "\n")
        print(f"Refinement results saved => {refine_out}\n")

def main():
    setup_directories()

    lines = []
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "ML_inputs.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Exiting.")
        return

    print("\nReading ML_inputs.txt configuration:")
    print("-" * 40)
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"):
                continue
            lines.append(line.strip())

    if len(lines) < 7:  # Updated to check for at least 7 lines
        print("Error: ML_inputs.txt must have at least 7 non-comment lines.")
        print(f"Found only {len(lines)} lines. Please check the file format.")
        return

    line1 = lines[0]  # e.g. "y;Si_trained_model" or "n"
    print("1> Train from scratch?:", line1)
    
    line2 = lines[1]  # e.g. "n" or "MyOldModel_20231212"
    print("2> Existing model to load:", line2)
    
    line3 = lines[2]  # e.g. "n;Si_20250210_135327" or "y;Si_10K\nPbSO4_8K..."
    print("3> Dataset configuration:", line3)
    
    line4 = lines[3]  # e.g. "Si.dat" or "n"
    print("4> Experimental .dat file(s):", line4)
    
    line5 = lines[4]  # e.g. "y" or "n"
    print("5> Do Rietveld Refinement?:", line5)
    
    line6 = lines[5]  # e.g. "y" or "n"
    print("6> Run feature importance test?:", line6)
    
    line7 = lines[6] if len(lines) >= 7 else "n"  # Default to "n" if not present
    print("7> Omit background parameter from CNN training and prediction?:", line7)
    
    line10 = lines[9] if len(lines) >= 10 else "y"  # Default to "y" if not present
    print("10> Use digit-based parameter scaling?:", line10)
    
    line11 = lines[10] if len(lines) >= 11 else "n"  # Default to "n" if not present
    print("11> Use reference-based adaptive scaling?:", line11)
    
    print("-" * 40, "\n")

    train_from_scratch = False
    new_model_name = ""
    
    use_digit_scaling = line10.lower().strip() == 'y'
    use_adaptive_scaling = line11.lower().strip() == 'y'
    
    if use_digit_scaling and use_adaptive_scaling:
        print("Warning: Both digit-based and adaptive scaling are enabled. Prioritizing digit-based scaling.")
        use_adaptive_scaling = False
        
    if ";" in line1:
        parts = line1.split(";", 1)
        if parts[0].lower().strip() == "y":
            train_from_scratch = True
            new_model_name = parts[1].strip()
    else:
        if line1.lower() == "y":
            train_from_scratch = True
            new_model_name = "MyNewModel"

    existing_model_folder = None
    if not train_from_scratch:
        if line2.lower() != "n":
            existing_model_folder = line2.strip()

    dataset_folders = []
    do_training = False
    if ";" in line3:
        parts = line3.split(";", 1)
        flag = parts[0].strip().lower()  # 'y' or 'n'
        if flag == 'y':
            do_training = True
            leftover = parts[1].strip()
            possible_ds = leftover.split(',')
            for ds in possible_ds:
                ds_clean = ds.strip()
                if ds_clean:
                    dataset_folders.append(ds_clean)
        else:
            leftover = parts[1].strip()
            if leftover:
                do_training = True
                splitted = leftover.split(',')
                for ds in splitted:
                    ds_clean = ds.strip()
                    if ds_clean:
                        dataset_folders.append(ds_clean)
    else:
        if line3.lower().startswith("y"):
            print("No dataset folder(s) listed after 'y;'. Skipping training.")
        elif line3.lower().startswith("n"):
            print("Skipping training because line3 is 'n' with no folder.")
        else:
            do_training = True
            dataset_folders.append(line3.strip())

    do_refine_files = []
    if line4.lower() != "n":
        do_refine_files = [x.strip() for x in line4.split(",") if x.strip()]

    do_refinement = line5.lower().strip() == 'y'

    do_feature_test = line6.lower().strip() == 'y'

    omit_background = line7.lower().strip() == 'y'

    if do_training and dataset_folders:
        model = None
        base_folder_used = ""
        current_param_info = None
        
        for folder_name in dataset_folders:
            dataset_path = os.path.join('data', 'train_data', folder_name)
            if not os.path.isdir(dataset_path):
                print(f"Dataset folder not found: {dataset_path}. Skipping.")
                continue

            row_file = None
            for candidate in os.listdir(dataset_path):
                if candidate.endswith('_row_param.dat'):
                    row_file = candidate
                    break

            if not row_file:
                print(f"No '*_row_param.dat' file found in {dataset_path}. Skipping.")
                continue

            row_path = os.path.join(dataset_path, row_file)
            
            base_name = os.path.basename(row_path).split('_row_param')[0]
            info_file = os.path.join(dataset_path, f"{base_name}_2theta_param_info.dat")
            
            param_info = None
            if os.path.exists(info_file):
                param_info = parse_info_file(info_file)
                if param_info:
                    print(f"Detected {param_info['total_params']} parameters from info file")
                    for param_type, count in param_info['param_counts'].items():
                        if count > 0:
                            print(f"  {param_type}: {count}")
                    
                    current_param_info = param_info
            
            if train_from_scratch and model is None:
                output_dim = None
                if current_param_info:
                    if omit_background and current_param_info['has_background']:
                        output_dim = current_param_info['total_params'] - 1
                        print(f"Adjusted output_dim to {output_dim} due to omit_background=True")
                    else:
                        output_dim = current_param_info['total_params']
                else:
                    output_dim = 7 if omit_background else 8
                
                print(f"\n*** Will train FROM SCRATCH => new model: {new_model_name} (output_dim={output_dim})")
                model = CNN(output_dim=output_dim, param_info=current_param_info).to(device)
            elif model is None:
                if existing_model_folder:
                    possible_pth = os.path.join('saved_models', existing_model_folder, f"{existing_model_folder}.pth")
                    if os.path.exists(possible_pth):
                        print(f"\n*** Will train using existing model => {existing_model_folder}")
                        model = load_model(possible_pth, param_info=current_param_info, omit_background=omit_background)
                        new_model_name = existing_model_folder
                        base_folder_used = existing_model_folder
                    else:
                        print(f"Could not find path: {possible_pth}. Creating from scratch.")
                        output_dim = None
                        if current_param_info:
                            if omit_background and current_param_info['has_background']:
                                output_dim = current_param_info['total_params'] - 1
                                print(f"Adjusted output_dim to {output_dim} due to omit_background=True")
                            else:
                                output_dim = current_param_info['total_params']
                        else:
                            output_dim = 7 if omit_background else 8
                        
                        model = CNN(output_dim=output_dim, param_info=current_param_info).to(device)
                        new_model_name = "updatedModel"
                        base_folder_used = ""
                else:
                    print("No existing model provided, starting from scratch.")
                    output_dim = None
                    if current_param_info:
                        if omit_background and current_param_info['has_background']:
                            output_dim = current_param_info['total_params'] - 1
                            print(f"Adjusted output_dim to {output_dim} due to omit_background=True")
                        else:
                            output_dim = current_param_info['total_params']
                    else:
                        output_dim = 7 if omit_background else 8
                    
                    model = CNN(output_dim=output_dim, param_info=current_param_info).to(device)
                    new_model_name = "updatedModel"
                    base_folder_used = ""

            source_log = os.path.join('data', 'train_data', 'cnn_training_results.dat')
            if os.path.exists(source_log):
                try:
                    os.remove(source_log)
                    print("Removed existing training log before starting new training.")
                except Exception as e:
                    print(f"Warning: Could not remove existing training log: {e}")

            print(f"\n=== Training dataset folder: {folder_name}")
            model, stime, etime, metrics_history, param_info = train_subfolder(
                row_path, model, base_folder_used or "Scratch", 
                omit_background, use_digit_scaling, use_adaptive_scaling)

            # Fix note: We do NOT append dataset name => reuse the same folder
            model_folder_prefix = new_model_name  

            if do_feature_test and param_info:
                print("\n=== Running Feature Importance Analysis ===")
                
                output_dir = os.path.join(
                    'saved_models', 
                    model_folder_prefix, 
                    'training_result',
                    folder_name,
                    'feature_importance'
                )
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    train_loader, val_loader, _ = get_loader(
                        row_path, 
                        omit_background=omit_background, 
                        info_file=info_file,
                        use_digit_scaling=use_digit_scaling,
                        use_adaptive_scaling=use_adaptive_scaling
                    )

                    param_types = param_info['param_types']
                    feature_param_names = []
                    
                    for i, param_type in enumerate(param_types):
                        if param_type == 'zero':
                            feature_param_names.append("Zero")
                        elif param_type == 'background':
                            if not omit_background:
                                feature_param_names.append("Background")
                        elif 'lattice parameter' in param_type:
                            lattice_label = param_type.split()[-1] if len(param_type.split()) > 2 else ""
                            feature_param_names.append(f"Lattice{' ' + lattice_label if lattice_label else ''}")
                        elif 'biso' in param_type:
                            atom_type = param_type.split('_')[1] if '_' in param_type else ""
                            feature_param_names.append(f"Biso{' ' + atom_type if atom_type else ''}")
                        elif 'scale factor' in param_type:
                            feature_param_names.append("Scale")
                        elif 'u parameter' in param_type:
                            feature_param_names.append("U")
                        elif 'v parameter' in param_type:
                            feature_param_names.append("V")
                        elif 'w parameter' in param_type:
                            feature_param_names.append("W")
                        else:
                            feature_param_names.append(f"Param{i+1}")
                    
                    if omit_background:
                        feature_param_names = [name for name in feature_param_names if name != "Background"]
                    
                    result_paths = run_feature_importance_analysis(
                        model=model,
                        val_loader=val_loader,
                        output_dir=output_dir,
                        data_dir=os.path.dirname(row_path),  # Pass the data directory
                        param_names=feature_param_names,
                        use_digit_scaling=use_digit_scaling,
                        use_adaptive_scaling=use_adaptive_scaling
                    )
                    
                    print("\nFeature importance analysis complete. Files saved:")
                    for key, path in result_paths.items():
                        print(f"  {key}: {path}")
                        
                except Exception as e:
                    print(f"\nWarning: Feature importance analysis encountered an error: {str(e)}")
                    print("Continuing with rest of pipeline...")

            updated_folder = post_training_logic(
                model,
                model_name_prefix=model_folder_prefix,
                used_row_path=row_path,
                do_refine_files=do_refine_files,
                base_folder_name=base_folder_used,
                start_time=stime,
                end_time=etime,
                metrics_history=metrics_history,
                omit_background=omit_background,
                param_info=param_info,
                use_digit_scaling=use_digit_scaling,
                use_adaptive_scaling=use_adaptive_scaling
            )

            if updated_folder:
                updated_pth = os.path.join('saved_models', updated_folder, f"{updated_folder}.pth")
                if os.path.exists(updated_pth):
                    model = load_model(updated_pth, param_info=param_info)
                    base_folder_used = updated_folder
                else:
                    print(f"Cannot find updated .pth in {updated_folder}, continuing with the same model in memory.")

            if os.path.exists(source_log):
                try:
                    os.remove(source_log)
                    print("Final cleanup: Removed leftover training log.")
                except Exception as e:
                    print(f"Warning: Could not remove leftover training log: {e}")

    else:
        if do_training:
            print("No valid dataset folders were provided. Training skipped.")
        else:
            print("Skipping training because line3 is 'n' (no training).")

    print("\nAll done. Each dataset folder has its own new model folder in 'saved_models'.")
    print("Exiting script now.")

if __name__ == "__main__":
    main()