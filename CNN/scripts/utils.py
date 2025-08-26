import numpy as np
import os
import torch
import torch.nn as nn
import decimal
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import time
from datetime import datetime, timedelta
import re

ctx = decimal.Context()

PARAM_TYPE_ZERO = 'zero'
PARAM_TYPE_BACKGROUND = 'background'
PARAM_TYPE_LATTICE = 'lattice parameter'
PARAM_TYPE_BISO = 'biso'
PARAM_TYPE_SCALE = 'scale factor'
PARAM_TYPE_U = 'u parameter'
PARAM_TYPE_V = 'v parameter'
PARAM_TYPE_W = 'w parameter'

class WeightedSmoothL1Loss(nn.Module):

    def __init__(self, weights=None, beta=1.0):
        super(WeightedSmoothL1Loss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)
        self.weights = weights if weights is not None else torch.FloatTensor([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.weight_sum = self.weights.sum()

    def forward(self, pred, target):
        per_param_loss = self.smooth_l1(pred, target)
        
        if self.weights.size(0) != per_param_loss.size(1):
            print(f"Loss function weights ({self.weights.size(0)}) don't match data dimensions ({per_param_loss.size(1)})")
            if self.weights.size(0) < per_param_loss.size(1):
                padding = torch.ones(per_param_loss.size(1) - self.weights.size(0), device=self.weights.device)
                adjusted_weights = torch.cat([self.weights, padding])
            else:
                adjusted_weights = self.weights[:per_param_loss.size(1)]
            
            self.weights = adjusted_weights
            self.weight_sum = self.weights.sum()
            print(f"Adjusted weights: {self.weights.tolist()}")
        
        weighted_loss = per_param_loss * self.weights.to(pred.device)
        
        return torch.sum(weighted_loss) / (self.weight_sum * pred.size(0))

def float_to_str(f):
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def parse_info_file(info_file_path):

    if not os.path.exists(info_file_path):
        return None
    
    try:
        param_list = []
        with open(info_file_path, 'r') as f:
            header_skipped = False
            for line in f:
                line = line.strip().lower()
                if not header_skipped and not line:
                    header_skipped = True
                    continue
                if header_skipped and line:
                    param_list.append(line)
        
        param_counts = {
            'zero': 0,
            'background': 0,
            'lattice parameter': 0,
            'biso': 0,
            'scale factor': 0,
            'u parameter': 0,
            'v parameter': 0,
            'w parameter': 0
        }
        
        param_types = []
        
        for param in param_list:
            if param == 'zero':
                param_counts['zero'] += 1
                param_types.append('zero')
            elif param == 'background':
                param_counts['background'] += 1
                param_types.append('background')
            elif 'lattice parameter' in param:
                param_counts['lattice parameter'] += 1
                param_types.append(param)  # Keep full description for specific lattice parameter
            elif 'biso' in param:
                param_counts['biso'] += 1
                param_types.append(param)  # Keep full description for specific atom
            elif 'scale factor' in param:
                param_counts['scale factor'] += 1
                param_types.append('scale factor')
            elif 'u parameter' in param:
                param_counts['u parameter'] += 1
                param_types.append('u parameter')
            elif 'v parameter' in param:
                param_counts['v parameter'] += 1
                param_types.append('v parameter')
            elif 'w parameter' in param:
                param_counts['w parameter'] += 1
                param_types.append('w parameter')
            else:
                param_types.append("unknown")
        
        has_background = param_counts['background'] > 0
        
        print(f"Parameter structure from info file:")
        print(f"Total parameters: {len(param_types)}")
        for i, param_type in enumerate(param_types):
            print(f"  {i+1}: {param_type}")
        
        print("Parameter counts (the number of each parameter type in the crystal structure):")
        
        return {
            'param_counts': param_counts,
            'param_types': param_types,
            'has_background': has_background,
            'total_params': len(param_list)
        }
    except Exception as e:
        print(f"Error parsing info file: {e}")
        return None

def process_data(data_tensor, label_tensor, scale=10, return_scales=False, omit_background=False, param_info=None):

    data_tensor = data_tensor / torch.max(data_tensor, dim=1)[0].unsqueeze(1)
    data_tensor -= torch.mean(data_tensor, dim=1).unsqueeze(1)
    data_tensor = data_tensor.unsqueeze(1)
    
    if param_info is not None:
        scaling_factors = []
        param_types = param_info['param_types']
        
        for param_type in param_types:
            if param_type == 'zero':
                scaling_factors.append(100) 
            elif param_type == 'background':
                scaling_factors.append(0.1)  
            elif 'lattice parameter' in param_type:
                scaling_factors.append(1.0)  
            elif 'biso' in param_type:
                scaling_factors.append(10.0)  
            elif 'scale factor' in param_type:
                scaling_factors.append(100000) 
            elif 'u parameter' in param_type:
                scaling_factors.append(1000)  
            elif 'v parameter' in param_type:
                scaling_factors.append(1000)  
            elif 'w parameter' in param_type:
                scaling_factors.append(1000)  
            else:
                scaling_factors.append(1.0)
        
        scaling_factors = torch.FloatTensor(scaling_factors)
        
        if omit_background and param_info['has_background']:
            bg_indices = [i for i, pt in enumerate(param_types) if pt == 'background']
            
            if bg_indices:
                if label_tensor.size(1) > bg_indices[0]:
                    indices_to_keep = [i for i in range(label_tensor.size(1)) if i not in bg_indices]
                    new_label = label_tensor[:, indices_to_keep]
                    label_tensor = new_label
                    
                    scaling_factors = scaling_factors[indices_to_keep]
                else:
                    print(f"Warning: Label tensor size ({label_tensor.size(1)}) doesn't match parameter count")
    else:
        if omit_background:
            scaling_factors = torch.FloatTensor([
                100,     
                1.0,     
                10.0,    
                100000,  
                1000,    
                1000,    
                1000     
            ])
            
            if label_tensor.size(1) == 8:  # Only if it has 8 parameters
                bg_index = 1  # Background is the second parameter (index 1)
                new_label = torch.cat([label_tensor[:, :bg_index], label_tensor[:, bg_index+1:]], dim=1)
                label_tensor = new_label
        else:
            scaling_factors = torch.FloatTensor([
                100,    
                0.1,    
                1.0,    
                10.0,   
                100000, 
                1000,   
                1000,   
                1000    
            ])
    
    label_tensor = label_tensor * scale
    
    if scaling_factors.size(0) != label_tensor.size(1):
        print(f"Warning: Scaling factors length ({scaling_factors.size(0)}) doesn't match label dimensions ({label_tensor.size(1)})")
        if scaling_factors.size(0) > label_tensor.size(1):
            scaling_factors = scaling_factors[:label_tensor.size(1)]
        else:
            extra_factors = torch.ones(label_tensor.size(1) - scaling_factors.size(0))
            scaling_factors = torch.cat([scaling_factors, extra_factors])
    
    label_tensor = label_tensor * scaling_factors
    
    if return_scales:
        return data_tensor, label_tensor, scaling_factors
    else:
        return data_tensor, label_tensor

def get_loader(filename, train_ratio=0.8, batch_size=100, label_cnts=8, omit_background=False, info_file=None, 
              use_digit_scaling=True, use_adaptive_scaling=False):

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")

    if info_file is None:
        base_name = os.path.basename(filename).split('_row_param')[0]
        possible_info = os.path.join(os.path.dirname(filename), f"{base_name}_2theta_param_info.dat")
        if os.path.exists(possible_info):
            info_file = possible_info
            print(f"Auto-detected info file: {info_file}")

    param_info = None
    if info_file and os.path.exists(info_file):
        param_info = parse_info_file(info_file)
        if param_info:
            label_cnts = param_info['total_params']
            print(f"Using {label_cnts} parameters based on info file: {info_file}")
            
            print("Parameter counts (number of each type in the crystal structure):")            
            for param_type, count in param_info['param_counts'].items():
                if count > 0:
                    print(f"  {param_type}: {count}")

    first_row_values = None
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            parts = first_line.split()
            if len(parts) >= label_cnts:
                first_row_values = [float(parts[-label_cnts + i]) for i in range(label_cnts)]
                print(f"Extracted first row values for adaptive scaling: {first_row_values}")

    data = []
    label = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                g = line.split()
                data.append([float(x) for x in g[:-label_cnts]])
                label.append([float(x) for x in g[-label_cnts:]])
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        raise
    
    data = torch.FloatTensor(data)
    label = torch.FloatTensor(label)
    
    modified_param_info = None
    if omit_background and param_info and param_info['has_background']:
        bg_indices = [i for i, pt in enumerate(param_info['param_types']) if pt == 'background']
        if bg_indices:
            bg_index = bg_indices[0]  # Use the first background parameter
            print(f"Background parameter found at index {bg_index}")
            
            modified_param_info = {key: value.copy() if isinstance(value, list) else value for key, value in param_info.items()}
            modified_param_info['param_types'] = [pt for i, pt in enumerate(param_info['param_types']) if i != bg_index]
            modified_param_info['has_background'] = False
            modified_param_info['total_params'] = len(modified_param_info['param_types'])
            modified_param_counts = dict(param_info['param_counts'])
            modified_param_counts['background'] = 0
            modified_param_info['param_counts'] = modified_param_counts
            
            print(f"Modified parameter info to reflect omit_background=True:")
            print(f"  New parameter count: {modified_param_info['total_params']}")
            
            if bg_index < label.size(1):
                print(f"Removing background parameter (index {bg_index}) from data")
                indices = list(range(label.size(1)))
                indices.pop(bg_index)
                label = label[:, indices]
                
                if first_row_values:
                    print(f"Original first row values: {first_row_values}")
                    first_row_values = [first_row_values[i] for i in range(len(first_row_values)) if i != bg_index]
                    print(f"Modified first row values after removing background: {first_row_values}")
                
                print(f"Modified label tensor shape: {label.shape}")
            else:
                print(f"Warning: bg_index {bg_index} is out of bounds for label tensor with {label.size(1)} columns")

    elif omit_background and not param_info and label.size(1) == 8:
        print(f"Removing background parameter (index 1) from data (no param_info)")
        indices = list(range(label.size(1)))
        indices.pop(1)  # Remove index 1 (Background)
        label = label[:, indices]
        
        if first_row_values and len(first_row_values) > 1:
            first_row_values = [first_row_values[i] for i in range(len(first_row_values)) if i != 1]
        
        print(f"Modified label tensor shape (default background removed): {label.shape}")
    
    if modified_param_info:
        param_info = modified_param_info
    
    data = data / torch.max(data, dim=1)[0].unsqueeze(1)
    data -= torch.mean(data, dim=1).unsqueeze(1)
    data = data.unsqueeze(1)
    
    param_types = []
    if param_info and 'param_types' in param_info:
        param_types = param_info['param_types']
    else:
        if label.size(1) == 7:  # No background
            param_types = ['zero', 'lattice parameter a', 'biso', 'scale factor', 'u parameter', 'v parameter', 'w parameter']
        else:  # With background
            param_types = ['zero', 'background', 'lattice parameter a', 'biso', 'scale factor', 'u parameter', 'v parameter', 'w parameter']
    
    if first_row_values:  # If we have first row values
        if len(first_row_values) != len(param_types):
            print(f"Warning: first_row_values length ({len(first_row_values)}) doesn't match parameter types length ({len(param_types)})")
            print("This can happen if background was removed from parameters but not from first_row_values")
            print("Current param_types:", param_types)
            print("Will attempt to truncate or extend first_row_values to match")
            
            if len(first_row_values) > len(param_types):
                first_row_values = first_row_values[:len(param_types)]
            else:
                first_row_values.extend([0.0] * (len(param_types) - len(first_row_values)))
            
            print(f"Adjusted first_row_values to length {len(first_row_values)}: {first_row_values}")
        
        if use_digit_scaling:
            print("Using digit-based scaling to normalize first significant digits")
            scaling_factors = determine_digit_based_scaling(first_row_values, param_types)
        elif use_adaptive_scaling:
            print("Using adaptive scaling based on first row values")
            scaling_factors = determine_adaptive_scaling(first_row_values, param_types)
        else:
            print("Using default parameter-specific scaling")
            scaling_factors = create_scaling_factors(param_info)
    else:  
        if use_digit_scaling or use_adaptive_scaling:
            print("Warning: Cannot use digit-based or adaptive scaling without first row values")
            print("Falling back to default parameter-specific scaling")
        
        print("Using default parameter-specific scaling")
        scaling_factors = create_scaling_factors(param_info)
    
    if scaling_factors.size(0) != label.size(1):
        print(f"Adjusting scaling factors to match label dimensions ({label.size(1)})")
        if scaling_factors.size(0) < label.size(1):
            padding = torch.ones(label.size(1) - scaling_factors.size(0))
            scaling_factors = torch.cat([scaling_factors, padding])
        else:
            scaling_factors = scaling_factors[:label.size(1)]
    
    if param_info:
        param_info['scaling_factors'] = scaling_factors
    
    label = label * 10  # Common scaling factor
    label = label * scaling_factors
    
    print(f"Final data shape: {data.shape}")
    print(f"Final label shape: {label.shape}")
    print(f"Scaling factors: {scaling_factors.tolist()}")
    
    train_cnt = int(len(data) * train_ratio)
    
    class ScaledTensorDataset(TensorDataset):
        def __init__(self, *tensors, scaling_factors=None):
            super(ScaledTensorDataset, self).__init__(*tensors)
            self.scaling_factors = scaling_factors
    
    train_dataset = ScaledTensorDataset(data[:train_cnt], label[:train_cnt], scaling_factors=scaling_factors)
    test_dataset = ScaledTensorDataset(data[train_cnt:], label[train_cnt:], scaling_factors=scaling_factors)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    
    return train_loader, test_loader, param_info

def create_scaling_factors(param_info=None):

    if param_info is None:
        return torch.FloatTensor([
            100,    
            0.1,    
            1.0,    
            10.0,  
            100000, 
            1000,   
            1000,   
            1000    
        ])
    
    scaling_factors = []
    for param_type in param_info['param_types']:
        if param_type == 'zero':
            scaling_factors.append(100) 
        elif param_type == 'background':
            scaling_factors.append(0.1)  
        elif 'lattice parameter' in param_type:
            scaling_factors.append(1.0)  
        elif 'biso' in param_type:
            scaling_factors.append(10.0)  
        elif 'scale factor' in param_type:
            scaling_factors.append(100000)  
        elif 'u parameter' in param_type:
            scaling_factors.append(1000)  
        elif 'v parameter' in param_type:
            scaling_factors.append(1000)  
        elif 'w parameter' in param_type:
            scaling_factors.append(1000)  
        else:
            scaling_factors.append(1.0)
    
    return torch.FloatTensor(scaling_factors)

class CNN(nn.Module):
    def __init__(self, act='relu', output_dim=8, param_info=None):
        super(CNN, self).__init__()
        self.nonlin = nn.ReLU() if act == 'relu' else nn.Tanh()
        self.output_dim = output_dim
        self.param_info = param_info
        
        self.lattice_count = 1  # Default for Si/cubic
        self.biso_count = 1     # Default for Si/cubic
        
        if param_info:
            self.lattice_count = param_info['param_counts'][PARAM_TYPE_LATTICE]
            self.biso_count = param_info['param_counts'][PARAM_TYPE_BISO]
            has_bg_in_info = param_info['has_background']
            
            if has_bg_in_info and output_dim < param_info['total_params']:
                self.has_background = False
                print(f"CNN model: Setting has_background=False because output_dim ({output_dim}) < total_params ({param_info['total_params']})")
            else:
                self.has_background = has_bg_in_info
        else:
            self.has_background = (output_dim == 8)  # For Si/cubic with 8 params, we assume background is included
            print(f"CNN model: Setting has_background={self.has_background} based on output_dim={output_dim}")
        
        class DummyLinear:
            def __init__(self, out_features):
                self.out_features = out_features
        self.lin = DummyLinear(output_dim)
        
        self.pool = nn.AvgPool1d(2)
        self.adppool = nn.AdaptiveAvgPool1d(4)
        
        self.conv1 = nn.Conv1d(1, 64, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 128, 5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.conv5 = nn.Conv1d(256, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        
        self.fc1 = nn.Linear(128*4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.zero_fc1 = nn.Linear(128, 64)
        self.zero_bn1 = nn.BatchNorm1d(64)
        self.zero_fc2 = nn.Linear(64, 32)
        self.zero_bn2 = nn.BatchNorm1d(32)
        self.zero_out = nn.Linear(32, 1)
        
        if self.has_background:
            self.bg_fc1 = nn.Linear(128, 32)
            self.bg_bn1 = nn.BatchNorm1d(32)
            self.bg_out = nn.Linear(32, 1)
        
        self.lattice_fc1 = nn.Linear(128, 64)
        self.lattice_bn1 = nn.BatchNorm1d(64)
        self.lattice_fc2 = nn.Linear(64, 32)
        self.lattice_bn2 = nn.BatchNorm1d(32)
        self.lattice_out = nn.Linear(32, self.lattice_count)  # Dynamic size
        
        self.biso_fc1 = nn.Linear(128, 64)
        self.biso_bn1 = nn.BatchNorm1d(64)
        self.biso_fc2 = nn.Linear(64, 32)
        self.biso_bn2 = nn.BatchNorm1d(32)
        self.biso_out = nn.Linear(32, self.biso_count)  # Dynamic size
        
        self.scale_fc1 = nn.Linear(128, 64)
        self.scale_bn1 = nn.BatchNorm1d(64)
        self.scale_fc2 = nn.Linear(64, 32)
        self.scale_bn2 = nn.BatchNorm1d(32)
        self.scale_out = nn.Linear(32, 1)
        
        self.uvw_fc1 = nn.Linear(128, 64)
        self.uvw_bn1 = nn.BatchNorm1d(64)
        self.uvw_out = nn.Linear(64, 3)  # U, V, W
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        with torch.no_grad():
            for m in [self.zero_out, self.lattice_out, self.biso_out, 
                      self.scale_out, self.uvw_out]:
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            if hasattr(self, 'bg_out'):
                nn.init.xavier_normal_(self.bg_out.weight, gain=0.01)
                if self.bg_out.bias is not None:
                    nn.init.zeros_(self.bg_out.bias)
    
    def forward(self, x):
        x = self.bn1(self.nonlin(self.conv1(x)))
        x = self.pool(x)
        
        x = self.bn2(self.nonlin(self.conv2(x)))
        x = self.pool(x)
        
        x = self.bn3(self.nonlin(self.conv3(x)))
        x = self.pool(x)
        
        x = self.bn4(self.nonlin(self.conv4(x)))
        x = self.pool(x)
        
        x = self.bn5(self.nonlin(self.conv5(x)))
        x = self.adppool(x).flatten(1)
        
        x = self.dropout1(self.nonlin(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(self.nonlin(self.bn_fc2(self.fc2(x))))
        
        zero = self.nonlin(self.zero_bn1(self.zero_fc1(x)))
        zero = self.nonlin(self.zero_bn2(self.zero_fc2(zero)))
        zero = self.zero_out(zero)
        
        lattice = self.nonlin(self.lattice_bn1(self.lattice_fc1(x)))
        lattice = self.nonlin(self.lattice_bn2(self.lattice_fc2(lattice)))
        lattice = self.lattice_out(lattice)  # Will output lattice_count values
        
        biso = self.nonlin(self.biso_bn1(self.biso_fc1(x)))
        biso = self.nonlin(self.biso_bn2(self.biso_fc2(biso)))
        biso = self.biso_out(biso)  # Will output biso_count values
        
        scale = self.nonlin(self.scale_bn1(self.scale_fc1(x)))
        scale = self.nonlin(self.scale_bn2(self.scale_fc2(scale)))
        scale = self.scale_out(scale)
        
        uvw = self.nonlin(self.uvw_bn1(self.uvw_fc1(x)))
        uvw = self.uvw_out(uvw)  # 3-channel output for U, V, W
        
        outputs = [zero]
        
        if self.has_background:
            bg = self.nonlin(self.bg_bn1(self.bg_fc1(x)))
            bg = self.bg_out(bg)
            outputs.append(bg)
        
        if self.lattice_count == 1:
            outputs.append(lattice)
        else:
            for i in range(self.lattice_count):
                outputs.append(lattice[:, i:i+1])
        
        if self.biso_count == 1:
            outputs.append(biso)
        else:
            for i in range(self.biso_count):
                outputs.append(biso[:, i:i+1])
        
        outputs.append(scale)
        outputs.append(uvw[:, 0:1])  # U
        outputs.append(uvw[:, 1:2])  # V
        outputs.append(uvw[:, 2:3])  # W
        
        return torch.cat(outputs, dim=1)

def calculate_mae(preds, targets):
    with torch.no_grad():
        mae_per_param = torch.mean(torch.abs(preds - targets), dim=0)
    return mae_per_param.cpu().numpy()

def calculate_rmse(preds, targets):
    with torch.no_grad():
        mse_per_param = torch.mean((preds - targets) ** 2, dim=0)
        rmse_per_param = torch.sqrt(mse_per_param)
    return rmse_per_param.cpu().numpy()

def train(model, train_loader, optimizer, criterion, m):
    tr_loss = 0
    output_dim = model.lin.out_features  # Get output dimension from model
    tr_mae = np.zeros(output_dim)  # Adjust for output_dim
    tr_rmse = np.zeros(output_dim)
    cnt = 0
    device = next(model.parameters()).device
    
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        model.zero_grad()
        preds = m(model(x))
        loss = criterion(preds, y)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            batch_mae = calculate_mae(preds, y)
            batch_rmse = calculate_rmse(preds, y)
            
            tr_mae += batch_mae * len(y)
            tr_rmse += batch_rmse * len(y)
            tr_loss += loss.item() * len(y)
            cnt += len(y)
    
    tr_loss /= cnt
    tr_mae /= cnt
    tr_rmse /= cnt
    return tr_mae, tr_rmse, tr_loss

def validate(model, val_loader, criterion, m):
    val_loss = 0
    output_dim = model.lin.out_features  # Get output dimension from model
    val_mae = np.zeros(output_dim)  # Adjust for output_dim
    val_rmse = np.zeros(output_dim)
    cnt = 0
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            preds = m(model(x))
            loss = criterion(preds, y)
            
            batch_mae = calculate_mae(preds, y)
            batch_rmse = calculate_rmse(preds, y)
            
            val_mae += batch_mae * len(y)
            val_rmse += batch_rmse * len(y)
            val_loss += loss.item() * len(y)
            cnt += len(y)
    
    val_loss /= cnt
    val_mae /= cnt
    val_rmse /= cnt
    return val_mae, val_rmse, val_loss

def create_loss_weights(param_info, output_dim=None):

    weights = []
    
    if param_info:
        for param_type in param_info['param_types']:
            if param_type == 'zero':
                weights.append(0.5)  # Zero parameter weight
            elif param_type == 'background':
                weights.append(1.0)  # Background weight
            elif 'lattice parameter' in param_type:
                weights.append(1.0)  # Lattice parameter weight
            elif 'biso' in param_type:
                weights.append(1.0)  # Biso parameter weight
            elif 'scale factor' in param_type:
                weights.append(1.0)  # Scale factor weight
            elif 'u parameter' in param_type:
                weights.append(1.0)  # U parameter weight
            elif 'v parameter' in param_type:
                weights.append(1.0)  # V parameter weight
            elif 'w parameter' in param_type:
                weights.append(1.0)  # W parameter weight
            else:
                weights.append(1.0)  # Default weight
    else:
        if output_dim == 7:
            weights = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 7 parameters without background
        else:
            weights = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 8 parameters with background
    
    if output_dim is not None and len(weights) != output_dim:
        print(f"Adjusting weights array from {len(weights)} to {output_dim} elements")
        if len(weights) < output_dim:
            weights.extend([1.0] * (output_dim - len(weights)))
        else:
            weights = weights[:output_dim]
    
    return torch.FloatTensor(weights)

def train_loop(model, train_loader, val_loader, optimizer, epochs=1,
               criterion=None, m=nn.Identity(), verbose=False,
               start_epoch=0, total_epochs=None, param_names=None, dataset_name=None,
               patience=7, min_improvement=1e-4, use_weighted_loss=True,
               zero_pct_error_threshold=3.0, 
               lattice_pct_error_threshold=0.01,  
               biso_pct_error_threshold=0.15,  
               scale_pct_error_threshold=0.1,  
               omit_background=False,
               param_info=None):
    
    if total_epochs is None:
        total_epochs = epochs
    
    for x, y in train_loader:
        output_dim = y.size(1)
        break
    
    print(f"Data from loader has {output_dim} parameters")
    
    if param_names is None:
        if param_info:
            param_types = param_info['param_types']
            param_names = []
            
            for i, param_type in enumerate(param_types):
                if param_type == 'zero':
                    param_names.append("Zero")
                elif param_type == 'background':
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
        else:
            if output_dim == 7:  
                param_names = ['Zero', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
            else:  
                param_names = ['Zero', 'Background', 'Lattice', 'Biso', 'Scale', 'U', 'V', 'W']
    
    if len(param_names) != output_dim:
        print(f"Adjusting parameter names to match data dimension ({output_dim})")
        if len(param_names) < output_dim:
            for i in range(len(param_names), output_dim):
                param_names.append(f"Param{i+1}")
        else:
            param_names = param_names[:output_dim]
        
        print(f"Updated parameter names: {param_names}")
    
    if criterion is None:
        if use_weighted_loss:
            weights = create_loss_weights(param_info, output_dim)
            
            print(f"Using loss weights: {weights.tolist()}")
            if len(param_names) == len(weights):
                print("Parameter weights:")
                for name, weight in zip(param_names, weights.tolist()):
                    print(f"  {name}: {weight}")
            
            criterion = WeightedSmoothL1Loss(weights=weights, beta=0.5)
        else:
            criterion = nn.SmoothL1Loss(beta=0.5)
    
    prev_val_loss = None
    prev_val_mae_avg = None
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    consecutive_increases = 0
    
    targets_mean = np.zeros(output_dim)
    target_count = 0
    
    for _, y in train_loader:
        targets_mean += np.sum(np.abs(y.cpu().numpy()), axis=0)
        target_count += len(y)
    for _, y in val_loader:
        targets_mean += np.sum(np.abs(y.cpu().numpy()), axis=0)
        target_count += len(y)
    targets_mean /= target_count
    
    start_time = time.time()
    epoch_times = []
    
    lr = optimizer.param_groups[0]['lr']
    real_batch_size = train_loader.batch_size
    
    if dataset_name is None:
        dataset_name = "unknown"
    
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_rmses, val_rmses = [], []
    train_pct_errs, val_pct_errs = [], []
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
        threshold=1e-4, threshold_mode='rel'
    )
    
    zero_param_idx = param_names.index('Zero') if 'Zero' in param_names else 0
    
    lattice_param_idx = None
    for i, name in enumerate(param_names):
        if 'Lattice' in name:
            lattice_param_idx = i
            break
    if lattice_param_idx is None:
        lattice_param_idx = 1 if 'Background' not in param_names else 2
    
    biso_param_idx = None
    for i, name in enumerate(param_names):
        if 'Biso' in name:
            biso_param_idx = i
            break
    if biso_param_idx is None:
        biso_param_idx = 2 if 'Background' not in param_names else 3
    
    scale_param_idx = None
    for i, name in enumerate(param_names):
        if name == 'Scale':
            scale_param_idx = i
            break
    if scale_param_idx is None:
        scale_param_idx = 3 if 'Background' not in param_names else 4
    
    print(f"Critical parameter indices: Zero={zero_param_idx}, Lattice={lattice_param_idx}, Biso={biso_param_idx}, Scale={scale_param_idx}")
    print(f"These indices track parameters with specific precision requirements for early stopping criteria")
    
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.time()
        
        tr_mae, tr_rmse, tr_loss = train(model, train_loader, optimizer, criterion, m)
        
        val_mae, val_rmse, val_loss = validate(model, val_loader, criterion, m)
        
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if verbose and prev_lr != current_lr:
            print(f"Learning rate changed: {prev_lr:.2e} -> {current_lr:.2e}")
        
        tr_mae_avg = np.mean(tr_mae)
        val_mae_avg = np.mean(val_mae)
        
        tr_pct_err = tr_mae / (targets_mean + 1e-8) * 100
        val_pct_err = val_mae / (targets_mean + 1e-8) * 100
        
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_maes.append(tr_mae_avg)
        val_maes.append(val_mae_avg)
        train_rmses.append(np.mean(tr_rmse))
        val_rmses.append(np.mean(val_rmse))
        train_pct_errs.append(np.mean(tr_pct_err))
        val_pct_errs.append(np.mean(val_pct_err))
        
        if prev_val_loss is not None and val_loss > prev_val_loss:
            consecutive_increases += 1
        else:
            consecutive_increases = 0
        
        zero_pct_error = val_pct_err[zero_param_idx] if zero_param_idx < len(val_pct_err) else 0.0
        lattice_pct_error = val_pct_err[lattice_param_idx] if lattice_param_idx < len(val_pct_err) else 0.0
        biso_pct_error = val_pct_err[biso_param_idx] if biso_param_idx < len(val_pct_err) else 0.0
        scale_pct_error = val_pct_err[scale_param_idx] if scale_param_idx < len(val_pct_err) else 0.0
        
        critical_params_ok = (
            zero_pct_error <= zero_pct_error_threshold and
            lattice_pct_error <= lattice_pct_error_threshold and
            biso_pct_error <= biso_pct_error_threshold and
            scale_pct_error <= scale_pct_error_threshold
        )
        
        if best_val_loss - val_loss > min_improvement:
            best_val_loss = val_loss
            patience_counter = 0
            consecutive_increases = 0
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
            stop_criteria_met = (
                (patience_counter >= patience and critical_params_ok) or
                (consecutive_increases >= 3 and critical_params_ok) or
                (val_loss > best_val_loss * 1.5 and critical_params_ok)
            )
            
            if patience_counter >= patience * 2 and not critical_params_ok:
                params_needing_work = []
                if zero_pct_error > zero_pct_error_threshold:
                    params_needing_work.append(f"Zero: {zero_pct_error:.2f}% > {zero_pct_error_threshold:.2f}%")
                if lattice_pct_error > lattice_pct_error_threshold:
                    params_needing_work.append(f"Lattice: {lattice_pct_error:.4f}% > {lattice_pct_error_threshold:.4f}%")
                if biso_pct_error > biso_pct_error_threshold:
                    params_needing_work.append(f"Biso: {biso_pct_error:.4f}% > {biso_pct_error_threshold:.4f}%")
                if scale_pct_error > scale_pct_error_threshold:
                    params_needing_work.append(f"Scale: {scale_pct_error:.4f}% > {scale_pct_error_threshold:.4f}%")
                
                optimizer.param_groups[0]['lr'] *= 0.5
                if verbose:
                    print(f"Parameter error thresholds not met. Reducing LR to {optimizer.param_groups[0]['lr']:.2e}")
                    for param_msg in params_needing_work:
                        print(f"  • {param_msg}")
                patience_counter = 0
            
            if stop_criteria_met:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    if patience_counter >= patience:
                        print(f"Validation loss did not improve by more than {min_improvement} for {patience} epochs")
                    elif consecutive_increases >= 3:
                        print(f"Validation loss increased for {consecutive_increases} consecutive epochs")
                    else:
                        print(f"Validation loss regressed significantly (>50% higher than best)")
                    print(f"Parameter error thresholds met:")
                    print(f"  • Zero: {zero_pct_error:.2f}% (target: {zero_pct_error_threshold:.2f}%)")
                    print(f"  • Lattice: {lattice_pct_error:.4f}% (target: {lattice_pct_error_threshold:.4f}%)")
                    print(f"  • Biso: {biso_pct_error:.4f}% (target: {biso_pct_error_threshold:.4f}%)")
                    print(f"  • Scale: {scale_pct_error:.4f}% (target: {scale_pct_error_threshold:.4f}%)")
                
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    if verbose:
                        print(f"Restored model from best epoch ({best_epoch+1})")
                break
        
        loss_change = ""
        mae_change = ""
        if prev_val_loss is not None:
            loss_diff = val_loss - prev_val_loss
            loss_change = f"{'+' if loss_diff > 0 else ''}{loss_diff:.4f} {'UP' if loss_diff > 0 else 'DOWN'}"
            
            mae_diff = val_mae_avg - prev_val_mae_avg
            mae_change = f"{'+' if mae_diff > 0 else ''}{mae_diff:.4f} {'UP' if mae_diff > 0 else 'DOWN'}"
        
        prev_val_loss = val_loss
        prev_val_mae_avg = val_mae_avg
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        
        if len(epoch_times) > 5:
            epoch_times = epoch_times[-5:]
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        elapsed = epoch_end - start_time
        remaining_epochs = total_epochs - epoch - 1
        remaining_time = avg_epoch_time * remaining_epochs
        
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
        eta = datetime.now() + timedelta(seconds=int(remaining_time))
        eta_str = eta.strftime("%H:%M:%S")
        
        progress = (epoch + 1) / total_epochs
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '>' + '.' * (bar_length - filled_length - 1)
        
        if verbose:
            print(f"+{'-' * 70}+")
            print(f"| EPOCH {epoch+1}/{total_epochs} [{bar}] {progress*100:.1f}% | Time: {elapsed_str}")
            print(f"+{'-' * 70}+")
            
            criterion_name = criterion.__class__.__name__
            print(f"| TRAINING:                                                           ")
            print(f"|   Loss:      {tr_loss:.6f} ({criterion_name})                       ")
            print(f"|   MAE:       {tr_mae_avg:.6f}                                       ")
            print(f"|   RMSE:      {np.mean(tr_rmse):.6f}                                 ")
            print(f"|                                                                    ")
            
            print(f"|   PARAMETERS      MAE           RMSE       % Error")
            for i, name in enumerate(param_names):
                if i < len(tr_mae):  # Avoid index error
                    print(f"|   {name:<14} {tr_mae[i]:.6f}    {tr_rmse[i]:.6f}    {tr_pct_err[i]:.2f}%     ")
            print(f"|                                                                    ")
            
            print(f"| VALIDATION:                                                         ")
            print(f"|   Loss:      {val_loss:.6f}    Change: {loss_change:<16}            ")
            print(f"|   MAE:       {val_mae_avg:.6f}    Change: {mae_change:<16}          ")
            print(f"|   RMSE:      {np.mean(val_rmse):.6f}                               ")
            print(f"|                                                                    ")
            
            print(f"|   PARAMETERS      MAE           RMSE       % Error     Target      ")
            for i, name in enumerate(param_names):
                if i < len(val_mae):  # Avoid index error
                    target_str = ""
                    if i == zero_param_idx:
                        target_str = f"< {zero_pct_error_threshold:.2f}%"
                    elif i == lattice_param_idx:
                        target_str = f"< {lattice_pct_error_threshold:.4f}%"
                    elif i == biso_param_idx:
                        target_str = f"< {biso_pct_error_threshold:.4f}%"
                    elif i == scale_param_idx:
                        target_str = f"< {scale_pct_error_threshold:.4f}%"
                        
                    print(f"|   {name:<14} {val_mae[i]:.6f}    {val_rmse[i]:.6f}    {val_pct_err[i]:.4f}%    {target_str:<10}")
            print(f"|                                                                    ")
            
            print(f"| Learning rate: {optimizer.param_groups[0]['lr']:.2e} | Batch size: {real_batch_size} | Dataset: {dataset_name} ")
            print(f"| Elapsed: {elapsed_str} | Remaining: {remaining_str} | ETA: {eta_str} ")
            print(f"| Early stopping patience: {patience_counter}/{patience} [Min val loss: {best_val_loss:.6f} at epoch {best_epoch+1}]")
            print(f"| Min improvement threshold: {min_improvement:.6f} | Consecutive increases: {consecutive_increases}/3")
            print(f"| Parameter thresholds: Zero {zero_pct_error:.2f}/{zero_pct_error_threshold:.2f}%, Lattice {lattice_pct_error:.4f}/{lattice_pct_error_threshold:.4f}%")
            print(f"| Biso {biso_pct_error:.4f}/{biso_pct_error_threshold:.4f}%, Scale {scale_pct_error:.4f}/{scale_pct_error_threshold:.4f}%")
            print(f"+{'-' * 70}+")
            print()
    
    best_epoch_idx = val_losses.index(min(val_losses)) if val_losses else len(val_losses) - 1
    
    metrics_history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_mae': train_maes,
        'val_mae': val_maes,
        'train_rmse': train_rmses,
        'val_rmse': val_rmses,
        'train_pct_err': train_pct_errs,
        'val_pct_err': val_pct_errs,
        'final_tr_mae': tr_mae,
        'final_val_mae': val_mae,
        'final_tr_rmse': tr_rmse,
        'final_val_rmse': val_rmse,
        'targets_mean': targets_mean,
        'best_epoch': best_epoch_idx + 1,
        'zero_final_pct_error': zero_pct_error,
        'lattice_final_pct_error': lattice_pct_error,
        'biso_final_pct_error': biso_pct_error,
        'scale_final_pct_error': scale_pct_error,
        'omit_background': omit_background,
        'param_info': param_info  # Include parameter info for reference
    }
    
    return metrics_history

def calculate_percent_error(pred, target, eps=1e-8):
    return np.abs((pred - target) / (np.abs(target) + eps)) * 100

def calculate_metrics_per_parameter(preds, targets):
    with torch.no_grad():
        mae = torch.mean(torch.abs(preds - targets), dim=0).cpu().numpy()
        
        mse = torch.mean((preds - targets) ** 2, dim=0)
        rmse = torch.sqrt(mse).cpu().numpy()
        
        mean_abs_targets = torch.mean(torch.abs(targets), dim=0).cpu().numpy()
        
        pct_err = np.zeros_like(mae)
        for i in range(len(mae)):
            if mean_abs_targets[i] > 1e-8:  # Avoid division by very small numbers
                pct_err[i] = (mae[i] / mean_abs_targets[i]) * 100
            else:
                pct_err[i] = 0.0  # Set to 0 if target is very close to 0
    
    return mae, rmse, pct_err

def determine_adaptive_scaling(first_row_values, param_types, reference_scale_values=None):

    if reference_scale_values is None:
        reference_scale_values = {
            'zero': 0.778,          
            'background': 1.918,    
            'lattice parameter': 5.43, 
            'biso': 7.03,           
            'scale factor': 66.79, 
            'u parameter': 5.0,     
            'v parameter': 6.07,    
            'w parameter': 4.58     
        }
    
    scaling_factors = []
    
    for i, param_type in enumerate(param_types):
        value = first_row_values[i]
        
        if abs(value) < 1e-10:
            if 'zero' in param_type:
                scaling_factors.append(100)
            elif 'background' in param_type:
                scaling_factors.append(0.1)
            elif 'lattice' in param_type:
                scaling_factors.append(1.0)
            elif 'biso' in param_type:
                scaling_factors.append(10.0)
            elif 'scale factor' in param_type:
                scaling_factors.append(100000)
            elif any(p in param_type for p in ['u parameter', 'v parameter', 'w parameter']):
                scaling_factors.append(1000)
            else:
                scaling_factors.append(1.0)
            continue
        
        if 'zero' in param_type:
            target = reference_scale_values['zero']
            factor = abs(target / value) if value != 0 else 100
            factor = round(factor / 10) * 10
            factor = max(10, min(1000, factor))  # Keep within reasonable bounds
            
        elif 'background' in param_type:
            target = reference_scale_values['background']
            factor = target / value if value != 0 else 0.1
            factor = round(factor * 1000) / 1000  # Round to 3 decimal places
            factor = max(0.001, min(1.0, factor))
            
        elif 'lattice parameter' in param_type:
            target = reference_scale_values['lattice parameter']
            factor = target / value if value != 0 else 1.0
            factor = round(factor * 100) / 100  # Round to 2 decimal places
            factor = max(0.1, min(10.0, factor))
            
        elif 'biso' in param_type:
            target = reference_scale_values['biso']
            factor = target / value if value != 0 else 10.0
            factor = round(factor * 10) / 10  # Round to 1 decimal place
            factor = max(0.1, min(100.0, factor))
            
        elif 'scale factor' in param_type:
            target = reference_scale_values['scale factor']
            factor = target / value if value != 0 else 100000
            # Round to nearest 5 for cleaner values
            factor = round(factor / 5) * 5
            if factor >= 10000:
                factor = round(factor / 1000) * 1000
            factor = max(10, min(1000000, factor))
            
        elif 'u parameter' in param_type:
            target = reference_scale_values['u parameter']
            factor = target / value if value != 0 else 1000
            factor = round(factor / 10) * 10
            factor = max(10, min(10000, factor))
            
        elif 'v parameter' in param_type:
            target = reference_scale_values['v parameter']
            factor = abs(target / value) if value != 0 else 1000  # Use abs for negative V values
            factor = round(factor / 10) * 10
            factor = max(10, min(10000, factor))
            
        elif 'w parameter' in param_type:
            target = reference_scale_values['w parameter']
            factor = target / value if value != 0 else 1000
            factor = round(factor / 10) * 10
            factor = max(10, min(10000, factor))
            
        else:
            factor = 1.0
            
        scaling_factors.append(factor)
    
    for i, factor in enumerate(scaling_factors):
        if not (0.0001 <= factor <= 1000000):
            print(f"Warning: Unreasonable scaling factor {factor} for {param_types[i]}, using default")
            if 'zero' in param_types[i]:
                scaling_factors[i] = 100
            elif 'background' in param_types[i]:
                scaling_factors[i] = 0.1
            elif 'lattice' in param_types[i]:
                scaling_factors[i] = 1.0
            elif 'biso' in param_types[i]:
                scaling_factors[i] = 10.0
            elif 'scale factor' in param_types[i]:
                scaling_factors[i] = 100000
            elif any(p in param_types[i] for p in ['u parameter', 'v parameter', 'w parameter']):
                scaling_factors[i] = 1000
            else:
                scaling_factors[i] = 1.0
    
    print("Adaptive scaling factors determined:")
    for param_type, factor in zip(param_types, scaling_factors):
        print(f"  {param_type}: {factor}")
    
    return torch.FloatTensor(scaling_factors)

def determine_digit_based_scaling(first_row_values, param_types):

    scaling_factors = []
    
    for i, value in enumerate(first_row_values):
        if abs(value) < 1e-10:
            param_type = param_types[i] if i < len(param_types) else ""
            if 'zero' in param_type:
                scaling_factors.append(100)
            elif 'background' in param_type:
                scaling_factors.append(0.1)
            elif 'lattice' in param_type:
                scaling_factors.append(1.0)
            elif 'biso' in param_type:
                scaling_factors.append(10.0)
            elif 'scale factor' in param_type:
                scaling_factors.append(100000)
            elif any(p in param_type for p in ['u parameter', 'v parameter', 'w parameter']):
                scaling_factors.append(1000)
            else:
                scaling_factors.append(1.0)
            continue
            
        magnitude = math.floor(math.log10(abs(value)))
        
        factor = 10 ** (-magnitude)
        
        scaled_value = value * factor
        
        if abs(scaled_value) < 1.0 or abs(scaled_value) >= 10.0:
            # Recalculate if necessary
            magnitude = math.floor(math.log10(abs(scaled_value)))
            factor *= 10 ** (-magnitude)
        
        factor = round(factor, 10)  # Remove floating point precision issues
        
        factor = max(1e-6, min(1e6, factor))
        
        scaling_factors.append(factor)
    
    print("Digit-based scaling factors determined:")
    for i, (param_type, factor) in enumerate(zip(param_types, scaling_factors)):
        scaled_val = first_row_values[i] * factor
        print(f"  {param_type}: {factor} (transforms {first_row_values[i]} to {scaled_val})")
    
    return torch.FloatTensor(scaling_factors)


def parse_si_dat_format(file_path):
    ttheta_vals, intensity_vals = [], []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
                ttheta_vals.append(x_val)
                intensity_vals.append(y_val)
            except ValueError:
                # Skip non-numeric lines
                continue
                
    return ttheta_vals, intensity_vals

def parse_ceo2_dat_format(file_path):
    ttheta_vals, intensity_vals = [], []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
                ttheta_vals.append(x_val)
                intensity_vals.append(y_val)
            except ValueError:
                # Skip non-numeric lines
                continue
                
    return ttheta_vals, intensity_vals

def parse_pbso4_dat_format(file_path):

    ttheta_vals, intensity_vals = [], []
    
    start_angle = 10.0  # Default start angle
    step_size = 0.05    # Default step size
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i in range(min(4, len(lines))):
        line = lines[i].strip()
        parts = line.split()
        if len(parts) == 1 and i == 2:  # Third line often has start angle
            try:
                start_angle = float(parts[0])
            except ValueError:
                pass
        elif len(parts) >= 3 and i == 1:  # Second line often has step size
            try:
                step_size = float(parts[2])
            except ValueError:
                pass
    
    print(f"Detected from header: start_angle={start_angle}, step_size={step_size}")
    
    current_angle = start_angle
    data_lines = []
    
    for line in lines[4:]:  # Skip 4 header lines
        line = line.strip()
        if not line:
            continue
        # Stop at termination markers
        if line.startswith('-1000') or line.startswith('-10000'):
            break
        data_lines.append(line)
    
    print(f"Found {len(data_lines)} data lines in the file")
    
    data_point_count = 0
    
    for line_num, line in enumerate(data_lines):
        parts = line.split()
        if not parts:
            continue
            
        current_prefix = parts[0]
        
        intensities = []
        
        if current_prefix == "10":
            for i in range(1, len(parts)):
                value_str = parts[i]
                
                if value_str.endswith("10") and len(value_str) > 2:
                    intensity_str = value_str[:-2]
                else:
                    intensity_str = value_str
                
                try:
                    intensity = int(intensity_str)
                    intensities.append(intensity)
                except ValueError:
                    print(f"Warning: Invalid intensity at line {line_num+5}, position {i}: {value_str}")
        else:
            for i in range(1, len(parts), 2):
                try:
                    intensity = int(parts[i])
                    intensities.append(intensity)
                except (ValueError, IndexError):
                    print(f"Warning: Invalid intensity at line {line_num+5}, position {i}")
        
        if len(intensities) != 10:
            print(f"Warning: Line {line_num+5} has {len(intensities)} intensity values (expected 10)")
        
        for intensity in intensities:
            ttheta_vals.append(current_angle)
            intensity_vals.append(intensity)
            current_angle += step_size
            data_point_count += 1
    
    print(f"Successfully extracted {data_point_count} data points from pbso4.dat format")
    
    print("First 10 data points:")
    for i in range(min(10, len(ttheta_vals))):
        print(f"  2-Theta: {ttheta_vals[i]:.3f}, Intensity: {intensity_vals[i]}")
    
    idx_60deg = next((i for i, angle in enumerate(ttheta_vals) if angle >= 60.0), 0)
    if idx_60deg > 0:
        print(f"\nData points around 2-theta = 60° (starting at index {idx_60deg}):")
        for i in range(idx_60deg, min(idx_60deg + 10, len(ttheta_vals))):
            print(f"  2-Theta: {ttheta_vals[i]:.3f}, Intensity: {intensity_vals[i]}")
    
    expected_count = len(data_lines) * 10  # Each line should have 10 intensity values
    actual_count = len(intensity_vals)
    
    if actual_count != expected_count:
        print(f"Warning: Expected {expected_count} data points but extracted {actual_count}")
        if actual_count < expected_count:
            print(f"Missing {expected_count - actual_count} data points")
    else:
        print(f"Data extraction validated: Expected {expected_count} points, extracted {actual_count}")
    
    return ttheta_vals, intensity_vals

def parse_tbbaco_dat_format(file_path):

    ttheta_vals, intensity_vals = [], []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return ttheta_vals, intensity_vals
    
    start_angle = 15.0   # Default
    step_size = 0.02     # Default
    
    header = lines[0].strip().split()
    if len(header) >= 2:
        try:
            start_angle = float(header[0])
            step_size = float(header[1])
        except ValueError:
            pass
    
    current_angle = start_angle
    
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        for part in parts:
            try:
                intensity = float(part)
                ttheta_vals.append(current_angle)
                intensity_vals.append(intensity)
                current_angle += step_size
            except ValueError:
                continue
    
    return ttheta_vals, intensity_vals

def verify_dat_format(file_path, expected_format):

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return False
            
        if expected_format == "Si.dat" or expected_format == "CeO2.dat":
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            float(parts[0])
                            float(parts[1])
                            return True
                        except ValueError:
                            pass
            return False
            
        elif expected_format == "pbso4.dat":
            for line in lines:
                if line.strip().startswith(('1 ', '2 ', '9 ', '10 ')):
                    return True
            return False
            
        elif expected_format == "tbbaco.dat":
            if len(lines) >= 2:
                header_parts = lines[0].strip().split()
                if len(header_parts) >= 2:
                    try:
                        float(header_parts[0])
                        float(header_parts[1])
                        
                        data_parts = lines[1].strip().split()
                        if data_parts and all(p.isdigit() for p in data_parts):
                            return True
                    except ValueError:
                        pass
            return False
            
        return False
    except Exception:
        return False

def parse_dat_file(file_path, format_type):

    import re
    
    if not verify_dat_format(file_path, format_type):
        print(f"Warning: {file_path} doesn't appear to match {format_type} format.")
        print("Falling back to default parsing method.")
        format_type = "Si.dat"  # Default format
    
    if format_type == "pbso4.dat":
        return parse_pbso4_dat_format(file_path)
    elif format_type == "tbbaco.dat":
        return parse_tbbaco_dat_format(file_path)
    elif format_type == "CeO2.dat":
        return parse_ceo2_dat_format(file_path)
    else: 
        return parse_si_dat_format(file_path)