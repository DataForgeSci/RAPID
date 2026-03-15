import os
import sys
import subprocess
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    print("\nSHAP package not found. Installing SHAP...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
        print("SHAP installation successful!")
        import shap
    except Exception as e:
        print(f"Error installing SHAP: {str(e)}")
        print("Will continue with permutation importance only.")

def compute_permutation_importance(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    param_names: List[str],
    n_repeats: int = 5
) -> Dict[str, float]:

    model.eval()
    baseline_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y, reduction='sum')
            baseline_loss += loss.item()
            n_samples += x.size(0)
    
    baseline_loss /= n_samples
    
    importance_scores = {}
    
    for param_idx, param_name in enumerate(param_names):
        total_increase = 0.0
        
        for _ in range(n_repeats):
            shuffled_loss = 0.0
            n_samples = 0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    perm_idx = torch.randperm(y.size(0))
                    y_shuffled = y.clone()
                    y_shuffled[:, param_idx] = y[perm_idx, param_idx]
                    
                    pred = model(x)
                    loss = torch.nn.functional.mse_loss(pred, y_shuffled, reduction='sum')
                    shuffled_loss += loss.item()
                    n_samples += x.size(0)
            
            shuffled_loss /= n_samples
            total_increase += (shuffled_loss - baseline_loss)
        
        importance_scores[param_name] = total_increase / n_repeats
    
    return importance_scores

def save_permutation_importance(
    scores: Dict[str, float],
    output_dir: str,
    timestamp: str
) -> Tuple[str, str]:

    params = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*params)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), values, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance Score (Increase in Loss)')
    plt.title('Parameter Importance (Permutation Method)')
    
    plot_path = os.path.join(output_dir, f'permutation_importance_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    data_path = os.path.join(output_dir, f'permutation_importance_{timestamp}.dat')
    with open(data_path, 'w') as f:
        f.write("# Permutation Importance Analysis Results\n")
        f.write("# Generated: {}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("# Parameter Name    Importance Score\n")
        for name, score in params:
            f.write(f"{name:<16} {score:.6f}\n")
    
    return plot_path, data_path

def save_shap_analysis(shap_values, feature_values, param_names, output_dir, timestamp, scaling_factors=None):

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import shap
    from datetime import datetime
    import os
    
    plot_paths = {}
    
    try:
        descaled_feature_values = feature_values.copy()
        
        if scaling_factors is None:
            has_background = 'Background' in param_names
            
            if has_background:
                scaling_factors = np.array([100, 0.1, 1.0, 10.0, 100000, 1000, 1000, 1000])
            else:
                scaling_factors = np.array([100, 1.0, 10.0, 100000, 1000, 1000, 1000])
                
            if len(scaling_factors) != len(param_names):
                if len(scaling_factors) > len(param_names):
                    scaling_factors = scaling_factors[:len(param_names)]
                else:
                    scaling_factors = np.append(
                        scaling_factors, 
                        np.ones(len(param_names) - len(scaling_factors))
                    )
        
        for i in range(len(param_names)):
            descaled_feature_values[:, i] = feature_values[:, i] / scaling_factors[i] / 10.0
        
        feature_matrix = pd.DataFrame(descaled_feature_values, columns=param_names)
        
        plt.figure(figsize=(10, 6))
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        indices = np.argsort(mean_abs_shap)
        sorted_params = [param_names[i] for i in indices[::-1]]  # Reverse to get descending order
        sorted_values = mean_abs_shap[indices[::-1]]
        
        plt.barh(sorted_params, sorted_values, color='#0099ff')
        plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
        plt.title('Parameter Importance Ranking (SHAP Analysis)')
        plt.tight_layout()
        
        shap_bar_path = os.path.join(output_dir, f'shap_importance_bars_{timestamp}.png')
        plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['shap_bar'] = shap_bar_path
        print(f"✓ Created SHAP bar plot: {shap_bar_path}")
        
        try:
            plt.figure(figsize=(10, 8))
            
            feature_order = np.argsort(mean_abs_shap)[::-1]  # Use same order as bar chart
            
            shap.summary_plot(
                shap_values, 
                feature_matrix,
                feature_names=param_names,
                plot_type="dot",  # This creates the beeswarm/dot plot
                plot_size=(10, 8),
                show=False,
                max_display=len(param_names),  # Show all parameters
                color_bar_label="Feature value"
            )
            
            plt.title('Rietveld Parameter Importance Visualization (SHAP Values)', fontsize=14)
            plt.tight_layout()
            beeswarm_path = os.path.join(output_dir, f'shap_beeswarm_{timestamp}.png')
            plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths['shap_beeswarm'] = beeswarm_path
            print(f"✓ Created SHAP beeswarm summary plot: {beeswarm_path}")
        except Exception as e:
            print(f"Warning: Could not create SHAP beeswarm plot: {e}")
            
            try:
                plt.figure(figsize=(10, 8))
                
                sorted_indices = np.argsort([-np.mean(np.abs(shap_values[:, i])) for i in range(shap_values.shape[1])])
                sorted_feature_names = [param_names[i] for i in sorted_indices]
                
                for i, idx in enumerate(sorted_indices):
                    param_shap = shap_values[:, idx]
                    param_values = feature_matrix.iloc[:, idx]
                    
                    norm_values = (param_values - param_values.min()) / (param_values.max() - param_values.min() + 1e-8)
                    
                    plt.scatter(
                        param_shap, 
                        np.ones(param_shap.shape[0]) * i, 
                        c=norm_values,
                        cmap='coolwarm',
                        alpha=0.7,
                        s=20
                    )
                
                plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
                plt.xlabel('SHAP value (impact on model output)')
                plt.title('Rietveld Parameter Importance - Manual SHAP Summary', fontsize=14)
                plt.colorbar(label='Feature value')
                plt.tight_layout()
                
                manual_beeswarm_path = os.path.join(output_dir, f'shap_manual_beeswarm_{timestamp}.png')
                plt.savefig(manual_beeswarm_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths['shap_manual_beeswarm'] = manual_beeswarm_path
                print(f"✓ Created manual SHAP beeswarm plot: {manual_beeswarm_path}")
            except Exception as e2:
                print(f"Warning: Could not create manual beeswarm plot either: {e2}")
        
        top_params = [param_names[i] for i in indices[::-1][:3]]

        for idx, param in enumerate(top_params):
            try:
                plt.figure(figsize=(10, 6))
                param_idx = param_names.index(param)
                
                remaining_indices = [i for i, p in enumerate(param_names) if p != param]
                remaining_importance = [mean_abs_shap[i] for i in remaining_indices]
                color_idx = remaining_indices[np.argmax(remaining_importance)]
                color_param = param_names[color_idx]
                
                scatter = plt.scatter(
                    feature_matrix[param], 
                    shap_values[:, param_idx],
                    c=feature_matrix[color_param],  # Color by next most important parameter
                    cmap='coolwarm', 
                    alpha=0.8,
                    s=50  # Slightly larger points
                )
                
                plt.colorbar(scatter, label=f'{color_param} value')
                plt.xlabel(f'{param}')
                plt.ylabel(f'SHAP value (impact on model output)')
                
                rank_text = ["#1", "#2", "#3"][idx]
                plt.title(f'SHAP Dependence Plot: {rank_text} Most Important Parameter ({param})\nColored by Most Influential Non-{param} Parameter ({color_param})', fontsize=14)
                
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                
                dep_plot_path = os.path.join(output_dir, f'shap_dependence_{param}_{timestamp}.png')
                plt.savefig(dep_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths[f'dependence_{param}'] = dep_plot_path
                print(f"✓ Created dependence plot for {param}: {dep_plot_path}")
                
            except Exception as e:
                print(f"Warning: Could not create dependence plot for {param}: {e}")

        try:
            top_param = top_params[0]
            top_idx = param_names.index(top_param)
            
            plt.figure(figsize=(10, 6))
            if hasattr(shap, 'dependence_plot'):
                interaction_indices = shap.approximate_interactions(top_idx, shap_values, feature_matrix)
                strongest_interaction_idx = interaction_indices[0] 
                interaction_param = param_names[strongest_interaction_idx]
                
                shap.dependence_plot(
                    top_idx, 
                    shap_values, 
                    feature_matrix,
                    feature_names=param_names,
                    interaction_index="auto", 
                    show=False
                )
                
                plt.title(f'Advanced SHAP Interaction Analysis\n\n{top_param} (Most Important Parameter) Colored by {interaction_param} (Strongest Interaction)', 
                          fontsize=14)
                
                adv_dep_path = os.path.join(output_dir, f'shap_adv_dependence_{top_param}_{timestamp}.png')
                plt.savefig(adv_dep_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths[f'adv_dependence_{top_param}'] = adv_dep_path
                print(f"✓ Created advanced dependence plot showing interaction between {top_param} and {interaction_param}: {adv_dep_path}")
        except Exception as e:
            print(f"Note: Could not create advanced SHAP dependence plot: {e}")
        
        plt.figure(figsize=(10, 8))
        
        corr_matrix = feature_matrix.corr()
        
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        cbar = plt.colorbar(im)
        cbar.set_label('Correlation Coefficient')
        
        plt.xticks(range(len(param_names)), param_names, rotation=45, ha='right')
        plt.yticks(range(len(param_names)), param_names)
        
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                         ha='center', va='center', 
                         color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        plt.title('Rietveld Parameter Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        
        corr_path = os.path.join(output_dir, f'parameter_correlation_{timestamp}.png')
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['correlation'] = corr_path
        print(f"✓ Created correlation heatmap: {corr_path}")
        
        data_paths = {}
        
        summary_data_path = os.path.join(output_dir, f'shap_summary_{timestamp}.dat')
        with open(summary_data_path, 'w') as f:
            f.write("# SHAP Analysis Results - Summary Statistics\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("# Parameter Name | Mean |SHAP| | Std SHAP | Max |SHAP| | Mean Value | Value Range\n")
            
            for param in sorted_params:
                i = param_names.index(param)
                values = shap_values[:, i]
                mean_abs = np.mean(np.abs(values))
                std = np.std(values)
                max_abs = np.max(np.abs(values))
                mean_value = np.mean(feature_matrix[param])
                min_value = np.min(feature_matrix[param])
                max_value = np.max(feature_matrix[param])
                f.write(f"{param:<15} {mean_abs:10.6f} {std:10.6f} {max_abs:10.6f} {mean_value:10.6f} [{min_value:10.6f}, {max_value:10.6f}]\n")
                
            f.write("\n\n# Parameter Correlation Matrix\n")
            f.write("# " + " ".join(f"{p:<12}" for p in param_names) + "\n")
            
            for i, param in enumerate(param_names):
                row = [param] + [f"{corr_matrix.iloc[i, j]:12.6f}" for j in range(len(param_names))]
                f.write(" ".join(row) + "\n")
        
        data_paths['summary'] = summary_data_path
        
        beeswarm_data_path = os.path.join(output_dir, f'shap_beeswarm_data_{timestamp}.dat')
        with open(beeswarm_data_path, 'w') as f:
            f.write("# SHAP Beeswarm Plot Raw Data\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("# This file contains the raw SHAP values and feature values for each sample and parameter\n\n")
            
            f.write("# Sample_ID")
            for param in param_names:
                f.write(f",{param}_SHAP,{param}_Value")
            f.write("\n")
            
            for sample_idx in range(shap_values.shape[0]):
                f.write(f"{sample_idx}")
                for param_idx, param in enumerate(param_names):
                    shap_val = shap_values[sample_idx, param_idx]
                    feature_val = feature_matrix[param].iloc[sample_idx]
                    f.write(f",{shap_val:.6f},{feature_val:.6f}")
                f.write("\n")
        
        data_paths['beeswarm'] = beeswarm_data_path
        
        for param in top_params:
            param_idx = param_names.index(param)
            
            remaining_indices = [i for i, p in enumerate(param_names) if p != param]
            remaining_importance = [mean_abs_shap[i] for i in remaining_indices]
            color_idx = remaining_indices[np.argmax(remaining_importance)]
            color_param = param_names[color_idx]
            
            dependence_data_path = os.path.join(output_dir, f'shap_dependence_{param}_data_{timestamp}.dat')
            with open(dependence_data_path, 'w') as f:
                f.write(f"# SHAP Dependence Plot Data for {param}\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Colored by: {color_param}\n\n")
                f.write(f"# {param}_Value,SHAP_Impact,{color_param}_Value\n")
                
                for sample_idx in range(shap_values.shape[0]):
                    param_val = feature_matrix[param].iloc[sample_idx]
                    shap_val = shap_values[sample_idx, param_idx] 
                    color_val = feature_matrix[color_param].iloc[sample_idx]
                    f.write(f"{param_val:.6f},{shap_val:.6f},{color_val:.6f}\n")
            
            data_paths[f'dependence_{param}'] = dependence_data_path
            
        bar_data_path = os.path.join(output_dir, f'shap_importance_bars_data_{timestamp}.dat')
        with open(bar_data_path, 'w') as f:
            f.write("# SHAP Bar Chart Importance Data\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("# Parameter,Mean_Absolute_SHAP_Value\n")
            
            for param, value in zip(sorted_params, sorted_values):
                f.write(f"{param},{value:.6f}\n")
        
        data_paths['bar_chart'] = bar_data_path
        
        corr_data_path = os.path.join(output_dir, f'parameter_correlation_data_{timestamp}.dat')
        with open(corr_data_path, 'w') as f:
            f.write("# Parameter Correlation Matrix\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Parameter," + ",".join(param_names) + "\n")
            
            for i, param in enumerate(param_names):
                row = [param] + [f"{corr_matrix.iloc[i, j]:.6f}" for j in range(len(param_names))]
                f.write(",".join(row) + "\n")
        
        data_paths['correlation'] = corr_data_path
        
        return plot_paths, data_paths
        
    except Exception as e:
        print(f"Error in SHAP visualization: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return {}, None

def compute_shap_values(model, val_loader, device, param_names, max_samples=50, data_dir=None):

    import time
    import torch
    import numpy as np
    import shap
    import traceback
    
    model.eval()
    
    print("\n=== SHAP Analysis Progress Tracking ===")
    start_time = time.time()
    
    print("\n1. Collecting background data for distribution reference...")
    background_data = []
    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            if i >= 10:  # Use just a few batches for background
                break
            background_data.append(x.to(device))
            print(f"   → Background batch {i+1}/10 collected")
    
    if not background_data:
        print("   ❌ ERROR: No background data could be collected")
        return None, None
        
    background = torch.cat(background_data)
    print(f"   ✓ Background data collected - Shape: {background.shape}")
    
    print("\n2. Loading classification report to identify CLOSE samples...")
    close_indices = []
    
    classification_file = None
    if data_dir:
        classification_file = os.path.join(data_dir, 'classification_report.dat')
    else:
        import glob
        possible_patterns = [
            os.path.join('data', 'train_data', '*', 'classification_report.dat'),
            os.path.join('data', '*', 'classification_report.dat'),
        ]
        
        for pattern in possible_patterns:
            matches = glob.glob(pattern)
            if matches:
                classification_file = matches[-1]  # Use the most recent one
                print(f"   ✓ Auto-detected classification report at: {classification_file}")
                break
    
    if classification_file and os.path.exists(classification_file):
        with open(classification_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.strip() and not line.startswith('#') and not line.startswith('Run:') and not line.startswith('DateTime:') and not line.startswith('Crystal'):
                parts = line.split()
                if len(parts) >= 2 and parts[0].startswith('sample'):
                    sample_id = parts[0]
                    classification = parts[-1]  # Last column is Class
                    if classification == 'CLOSE':
                        # Extract sample number (e.g., 'sample1' -> 0, 'sample2' -> 1)
                        sample_num = int(sample_id.replace('sample', '')) - 1
                        close_indices.append(sample_num)
        
        print(f"   ✓ Found {len(close_indices)} CLOSE samples in classification report")
    else:
        print("   ⚠ Warning: classification_report.dat not found. Using all samples.")
        close_indices = list(range(len(val_loader.dataset)))
    
    print("\n3. Collecting test samples for explanation (CLOSE samples only)...")
    
    all_x = []
    all_y = []
    with torch.no_grad():
        for x, y in val_loader:
            all_x.append(x)
            all_y.append(y)
    
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    val_size = len(all_x)  
    
    total_samples = 10000  
    train_ratio = 0.8  
    train_size = int(total_samples * train_ratio)
    
    val_close_indices = []
    for idx in close_indices:
        if idx >= train_size:  # This sample is in validation set
            val_idx = idx - train_size  # Convert to validation set index
            if val_idx < val_size:  
                val_close_indices.append(val_idx)
    
    print(f"   ✓ Found {len(val_close_indices)} CLOSE samples in validation set (from {len(close_indices)} total CLOSE samples)")
    
    if len(val_close_indices) > 0:
        close_x = all_x[val_close_indices]
        close_y = all_y[val_close_indices]
        
        if len(val_close_indices) >= max_samples:
            selected_indices = np.random.choice(len(val_close_indices), max_samples, replace=False)
            test_samples = close_x[selected_indices].to(device)
            test_targets = close_y[selected_indices].to(device)
            print(f"   ✓ Selected {max_samples} samples from {len(val_close_indices)} CLOSE validation samples")
        else:
            test_samples = close_x.to(device)
            test_targets = close_y.to(device)
            print(f"   ⚠ Only {len(val_close_indices)} CLOSE validation samples available (requested {max_samples})")
    else:
        print("   ⚠ No CLOSE samples found, using random selection from all samples")
        n_samples = min(max_samples, len(all_x))
        selected_indices = np.random.choice(len(all_x), n_samples, replace=False)
        test_samples = all_x[selected_indices].to(device)
        test_targets = all_y[selected_indices].to(device)
    
    print(f"   ✓ Test data collected - Shape: {test_samples.shape}")
    
    try:
        print("\n3. Creating GradientExplainer...")
        explainer_start = time.time()
        explainer = shap.GradientExplainer(model, background)
        explainer_end = time.time()
        print(f"   ✓ GradientExplainer created in {explainer_end - explainer_start:.2f} seconds")
        
        print("\n4. Computing SHAP values (this may take a while)...")
        shap_start = time.time()
        
        batch_size = 5  # Process 5 samples at a time for progress tracking
        total_batches = (test_samples.shape[0] + batch_size - 1) // batch_size
        all_shap_values = []
        
        for i in range(0, test_samples.shape[0], batch_size):
            batch_start = time.time()
            end_idx = min(i + batch_size, test_samples.shape[0])
            batch = test_samples[i:end_idx]
            
            print(f"   → Processing batch {i//batch_size + 1}/{total_batches} (samples {i+1}-{end_idx}/{test_samples.shape[0]})...")
            batch_shap_values = explainer.shap_values(batch)
            all_shap_values.append(batch_shap_values)
            
            batch_end = time.time()
            elapsed = batch_end - batch_start
            remaining = elapsed * (total_batches - (i//batch_size + 1))
            print(f"     • Batch completed in {elapsed:.2f} seconds")
            print(f"     • Estimated remaining time: {remaining:.2f} seconds")
        
        print("\n5. Processing SHAP values for visualization...")
        if isinstance(all_shap_values[0], list):
            shap_values = []
            for param_idx in range(len(all_shap_values[0])):
                combined = []
                for batch_result in all_shap_values:
                    combined.append(batch_result[param_idx])
                shap_values.append(np.concatenate(combined, axis=0))
                print(f"   → Combined results for parameter {param_idx+1}/{len(all_shap_values[0])}")
        else:
            shap_values = np.concatenate(all_shap_values, axis=0)
            print("   → Combined results for single output model")
        
        shap_end = time.time()
        print(f"   ✓ SHAP computation completed in {shap_end - shap_start:.2f} seconds")
        
        print("\n6. Formatting SHAP values for visualization...")
        if isinstance(shap_values, list):
            print(f"   → SHAP values shape (before processing): {[sv.shape for sv in shap_values]}")
            
            processed_shap = np.zeros((test_samples.shape[0], len(param_names)))
            for i in range(min(len(shap_values), len(param_names))):
                sv = shap_values[i]
                processed_shap[:, i] = np.mean(sv, axis=(1, 2))  
                print(f"   → Processed parameter {i+1}/{min(len(shap_values), len(param_names))}: {param_names[i]}")
                
            shap_values = processed_shap
        else:
            print(f"   → SHAP values shape: {shap_values.shape}")
            if len(shap_values.shape) > 2:
                shap_values = np.mean(shap_values, axis=(1, 2))  
                print("   → Averaged SHAP values across features")
        
        features = test_targets.cpu().numpy()
        
        total_time = time.time() - start_time
        print(f"\n✓ SHAP analysis completed successfully!")
        print(f"✓ Final processed SHAP values shape: {shap_values.shape}")
        print(f"✓ Features shape: {features.shape}")
        print(f"✓ Total time: {total_time:.2f} seconds")
        print("=====================================")
        
        return shap_values, features
        
    except Exception as e:
        print(f"\n❌ ERROR in SHAP computation with GradientExplainer: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return None, None

def run_feature_importance_analysis(
    model, 
    val_loader, 
    output_dir,
    data_dir=None,
    param_names=None,
    omit_background=False,
    use_digit_scaling=True,
    use_adaptive_scaling=False
):

    import os
    from datetime import datetime
    import torch
    
    output_dim = model.lin.out_features
    
    if param_names is None:
        if output_dim == 7:  # 7 parameters (likely background omitted)
            param_names = ['Zero', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
        else:  # Default 8 parameters
            param_names = ['Zero', 'Background', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
            
            if omit_background and 'Background' in param_names:
                param_names.remove('Background')
                print("Removed 'Background' from param_names due to omit_background=True")
    
    if len(param_names) != output_dim:
        print(f"Warning: Parameter names count ({len(param_names)}) doesn't match model output dimension ({output_dim})")
        if len(param_names) < output_dim:
            for i in range(len(param_names), output_dim):
                param_names.append(f"Param{i+1}")
            print(f"Added generic parameter names: {param_names}")
        else:
            param_names = param_names[:output_dim]
            print(f"Truncated parameter names: {param_names}")
    
    device = next(model.parameters()).device
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    scaling_factors = None
    for batch in val_loader:
        if hasattr(val_loader.dataset, 'scaling_factors'):
            scaling_factors = val_loader.dataset.scaling_factors
            print(f"Using scaling factors from dataset: {scaling_factors}")
            break
    
    if scaling_factors is None:
        has_background = 'Background' in param_names
        
        if has_background:
            scaling_factors = torch.FloatTensor([[100, 0.1, 1.0, 10.0, 100000, 1000, 1000, 1000]]).numpy()[0]
        else:
            scaling_factors = torch.FloatTensor([[100, 1.0, 10.0, 100000, 1000, 1000, 1000]]).numpy()[0]
        
        print(f"Using fallback scaling factors: {scaling_factors}")
    
    if len(scaling_factors) != len(param_names):
        print(f"Adjusting scaling factors to match parameter names ({len(param_names)})")
        if len(scaling_factors) > len(param_names):
            scaling_factors = scaling_factors[:len(param_names)]
        else:
            scaling_factors = np.append(
                scaling_factors, 
                np.ones(len(param_names) - len(scaling_factors))
            )
    
    print("\nComputing permutation importance...")
    perm_scores = compute_permutation_importance(model, val_loader, device, param_names)
    perm_plot, perm_data = save_permutation_importance(perm_scores, output_dir, timestamp)
    
    result_paths = {
        'permutation_plot': perm_plot,
        'permutation_data': perm_data,
    }
    
    try:
        print("\nAttempting SHAP analysis...")
        shap_values, feature_values = compute_shap_values(model, val_loader, device, param_names, data_dir=data_dir)
        
        if shap_values is not None and feature_values is not None:
            shap_plots, shap_data = save_shap_analysis(
                shap_values, feature_values, param_names, output_dir, timestamp, 
                scaling_factors=scaling_factors  # Pass the scaling factors
            )
            
            result_paths.update(shap_plots)
            result_paths['shap_data'] = shap_data
            
            print("\nSHAP analysis completed successfully. Generated files:")
            for key, path in shap_plots.items():
                print(f"  • {key}: {path}")
            print(f"  • Data: {shap_data}")
            
    except Exception as e:
        print(f"\nNote: SHAP analysis skipped due to error: {str(e)}")
        print("Using permutation importance results only.")
    
    return result_paths