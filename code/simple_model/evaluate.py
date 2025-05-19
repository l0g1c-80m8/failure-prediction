import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import wandb
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt
import sys
import argparse
import random

from concurrent.futures import ThreadPoolExecutor

from resnet_models import resnet18, resnet34, resnet50, resnet101, resnet152

def load_model(model_path, model_architecture, input_channels, dropout_rate=0.3):
    """Load a trained model from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the model
    if model_architecture == 'resnet18':
        model = resnet18(input_channels=input_channels, dropout_rate=dropout_rate)
    elif model_architecture == 'resnet34':
        model = resnet34(input_channels=input_channels, dropout_rate=dropout_rate)
    elif model_architecture == 'resnet50':
        model = resnet50(input_channels=input_channels, dropout_rate=dropout_rate)
    elif model_architecture == 'resnet101':
        model = resnet101(input_channels=input_channels, dropout_rate=dropout_rate)
    elif model_architecture == 'resnet152':
        model = resnet152(input_channels=input_channels, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle case where model was saved with DistributedDataParallel (has 'module.' prefix)
    if list(state_dict.keys())[0].startswith('module.'):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    return model

class RobotTrajectoryDataset(Dataset):
    def __init__(self, data_dir, window_size=10, stride=1, use_cache=True, cache_size=1000, batch_size=1024, sub_batch_size=128, dual_input=False):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.cache = {}
        self.cache_hits = 0
        self.dual_input = dual_input

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

        print(f"Loading data from {data_dir} with window size {window_size} and stride {stride}")

        os.makedirs(f"{data_dir}/sub_batches/", exist_ok=True)
        
        # Get list of all .npy files in directory
        self.episode_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        random.shuffle(self.episode_files)
        total_files = len(self.episode_files)
        self.total_samples = 0

        indices_cache_path = os.path.join(data_dir, f"indices_cache_w{window_size}_s{stride}.npy")
        
        # Check if we have already processed the data
        if os.path.exists(f"{data_dir}/sub_batches/") and len(os.listdir(f"{data_dir}/sub_batches/")) > 0:
            print(f"Sub-batches directory exists with data. Skipping pre-processing.")
            self.subbatch_files = [f for f in os.listdir(f"{data_dir}/sub_batches/") if f.endswith('.npy')]
            
            # Estimate total samples from existing sub-batches
            self.total_samples = len(self.subbatch_files) * sub_batch_size
        else:
            # Process data into sub-batches
            splitted_episodes = [self.episode_files[i:i + cache_size] for i in range(0, total_files, cache_size)]
            for split_id, split_list in enumerate(splitted_episodes):
                bulk_samples = []

                for episode_file in tqdm(split_list, desc=f"Creating sub-batches from list {split_id+1}/{len(splitted_episodes)}", total=len(split_list)):
                    if episode_file == f"indices_cache_w{window_size}_s{stride}.npy":
                        continue
                    episode_data = np.load(os.path.join(data_dir, episode_file), allow_pickle=True)
                    n_frames = len(episode_data)
                    
                    for start_idx in range(0, n_frames - window_size + 1, stride):
                        end_idx = start_idx + window_size
                        bulk_samples.append(self.process_sample(episode_data[start_idx:end_idx]))
                
                random.shuffle(bulk_samples)
                for idx, sub_batch in enumerate([bulk_samples[i:i + sub_batch_size] for i in range(0, len(bulk_samples), sub_batch_size)]):
                    sub_batch = np.array(sub_batch)
                    np.save(os.path.join(data_dir, f'sub_batches/{split_id}_sub_batch_{idx}.npy'), sub_batch)

                self.total_samples += len(bulk_samples)

            self.subbatch_files = [f for f in os.listdir(f"{data_dir}/sub_batches/") if f.endswith('.npy')]

        print(f"Total samples: {self.total_samples}, Total sub-batches: {len(self.subbatch_files)}")
        self.loaded_batch_samples = []

    def process_sample(self, window_data):
        # Get states sequence
        states = np.stack([frame['state'] for frame in window_data])
        
        if self.dual_input:
            states_cam1 = states[:, :6]
            states_cam2 = states[:, 6:12]
            ee_pos = states[:, 12:15]  # End effector position (3 values)

            # Concatenate the 3 EE_pos values to the first camera's 6 values (RT) (Total 9 values)
            states_cam1 = np.concatenate((states_cam1, ee_pos), axis=1)

            # Concatenate the 3 EE_pos values to the second camera's 6 values (RT) (Total 9 values)
            states_cam2 = np.concatenate((states_cam2, ee_pos), axis=1)
            
            # Get the risk for the last timestep
            risk = np.array(window_data[-1]['risk'], dtype=np.float32)
            
            return {
                'states_cam1': torch.FloatTensor(states_cam1).transpose(0, 1),
                'states_cam2': torch.FloatTensor(states_cam2).transpose(0, 1),
                'risk': torch.FloatTensor(risk)
            }
        else:
            # Get the risk for the last timestep
            risk = np.array(window_data[-1]['risk'], dtype=np.float32)
            
            return {
                'states': torch.FloatTensor(states).transpose(0, 1),
                'risk': torch.FloatTensor(risk)
            }

    def load_sub_batches(self):
        if len(self.subbatch_files) < self.batch_size//self.sub_batch_size:
            # If we're running out of sub-batches, reset the list
            self.subbatch_files = [f for f in os.listdir(f"{self.data_dir}/sub_batches/") if f.endswith('.npy')]
            
        self.loaded_batch_samples = []
        current_batch_files = random.sample(self.subbatch_files, min(self.batch_size//self.sub_batch_size, len(self.subbatch_files)))
        
        for sub_batch_file in current_batch_files:
            sub_batch_path = os.path.join(self.data_dir, "sub_batches", sub_batch_file)
            sub_batch_data = np.load(sub_batch_path, allow_pickle=True)
            self.loaded_batch_samples.extend(sub_batch_data)
            self.subbatch_files.remove(sub_batch_file)  # Remove loaded sub-batch file to avoid reloading

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if len(self.loaded_batch_samples) == 0 or idx % self.batch_size == 0:
            self.load_sub_batches()
        
        try:
            # Get a sample from the loaded batch
            data_sample = self.loaded_batch_samples[idx % len(self.loaded_batch_samples)]
        except Exception as e:
            print(f"Error: {e}", idx % self.batch_size, len(self.loaded_batch_samples))
            # Fallback mechanism if there's an issue
            self.load_sub_batches()
            data_sample = self.loaded_batch_samples[0]

        return data_sample

def create_data_loaders(test_dir, window_size=10, stride=1, batch_size=32, num_workers=4, dual_input=False):
    """Create train and validation data loaders with optimization options."""
    
    # Calculate appropriate sub_batch_size (must divide evenly into batch_size)
    sub_batch_size = min(128, batch_size // 8)  # Aim for 8 sub-batches per batch
    if batch_size % sub_batch_size != 0:
        sub_batch_size = batch_size // (batch_size // sub_batch_size)  # Adjust to ensure divisibility
       
    test_dataset = RobotTrajectoryDataset(
        data_dir=test_dir,
        window_size=window_size,
        stride=stride,
        use_cache=True,
        cache_size=1000,
        batch_size=batch_size,
        sub_batch_size=sub_batch_size,
        dual_input=dual_input
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    return test_loader

def evaluate_model(model, test_loader, model_name, training_mode="standard"):
    """Evaluate the model on test data with progress tracking and visualization."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_mse = 0
    
    # Create progress bar for evaluation
    eval_pbar = tqdm(test_loader, desc=f"Evaluating {model_name}")
    
    # Store predictions and ground truth for visualization
    all_predictions = []
    all_ground_truth = []
    all_mse_values = []
    sample_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_pbar):
            if training_mode == "dual_input_max_loss":
                # Mode 1: Single model with dual inputs
                states_cam1 = batch['states_cam1'].to(device)
                states_cam2 = batch['states_cam2'].to(device)
                risks = batch['risk'].to(device)
                
                # Forward pass for both camera inputs
                outputs_cam1 = model(states_cam1)
                outputs_cam2 = model(states_cam2)
                
                # Use maximum prediction
                outputs = torch.max(outputs_cam1, outputs_cam2)
            else:
                # Standard mode: Single input
                states = batch['states'].to(device)
                risks = batch['risk'].to(device)
                outputs = model(states)
            
            mse = nn.MSELoss()(outputs, risks).item()
            total_mse += mse
            
            # Store predictions and ground truth
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truth.append(risks.cpu().numpy())
            all_mse_values.append(mse)
            sample_indices.extend([batch_idx * test_loader.batch_size + i for i in range(len(outputs))])
            
            # Update progress bar
            eval_pbar.set_postfix({'mse': f'{mse:.4f}'})
    
    avg_mse = total_mse / len(test_loader)
    
    # Concatenate all data for visualization
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    # Create visualizations for wandb
    visualize_predictions(predictions, ground_truth)
    
    return avg_mse

def evaluate_dual_models(model_cam1, model_cam2, test_loader, model_name_prefix):
    """Evaluate the ensemble of two models."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cam1.eval()
    model_cam2.eval()
    total_mse = 0
    
    # Create progress bar for evaluation
    eval_pbar = tqdm(test_loader, desc=f"Evaluating Ensemble {model_name_prefix}")
    
    # Store predictions and ground truth for visualization
    all_predictions = []
    all_ground_truth = []
    all_mse_values = []
    
    with torch.no_grad():
        for batch in eval_pbar:
            states_cam1 = batch['states_cam1'].to(device)
            states_cam2 = batch['states_cam2'].to(device)
            risks = batch['risk'].to(device)
            
            # Forward pass for both models
            outputs_cam1 = model_cam1(states_cam1)
            outputs_cam2 = model_cam2(states_cam2)
            
            # Use maximum prediction from both models
            outputs = torch.max(outputs_cam1, outputs_cam2)
            
            mse = nn.MSELoss()(outputs, risks).item()
            total_mse += mse
            
            # Store predictions and ground truth
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truth.append(risks.cpu().numpy())
            all_mse_values.append(mse)
            
            # Update progress bar
            eval_pbar.set_postfix({'mse': f'{mse:.4f}'})
    
    avg_mse = total_mse / len(test_loader)
    
    # Concatenate all data for visualization
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    # Create visualizations for wandb
    visualize_predictions(predictions, ground_truth)
    
    return avg_mse

def visualize_predictions(predictions, ground_truth):
    """Create visualizations comparing predictions to ground truth."""
    # Print shapes for verification
    print("predictions", predictions.shape)
    print("ground_truth", ground_truth.shape)
    
    # Create indices for x-axis
    indices = np.arange(len(predictions))
    
    # Use all data points since dataset is small
    sample_indices = indices
    
    # Make sure predictions and ground_truth are properly formatted
    # Convert to float to ensure compatibility with wandb
    pred_values = predictions.flatten().astype(float)
    truth_values = ground_truth.flatten().astype(float)
    
    # Create wandb plot - make sure data is correctly formatted
    data = []
    for i in sample_indices:
        # Ensure values are scalar and not arrays
        pred_val = float(pred_values[i])
        truth_val = float(truth_values[i])
        data.append([int(i), pred_val, truth_val])
    
    # Create the table
    table = wandb.Table(data=data, columns=["index", "prediction", "ground_truth"])
    
    # Log the plot
    wandb.log({"predictions_vs_ground_truth": wandb.plot.line(
        table, 
        "index", 
        ["prediction", "ground_truth"],
        title="Model Predictions vs Ground Truth"
    )})

if __name__ == "__main__":
    os.environ['QT_X11_NO_MITSHM'] = '1'
    # Initialize wandb
    wandb.login()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate trained risk prediction models')
    parser.add_argument('--training-mode', type=str, default='standard',
                      choices=['standard', 'dual_input_max_loss', 'dual_models'],
                      help='Training mode to use')
    parser.add_argument('--local-rank', type=int, default=0, 
                      help='Local rank for distributed training')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained model file')
    parser.add_argument('--model-architecture', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                      help='Architecture of the model')
    parser.add_argument('--dual-model-path-cam2', type=str, default=None,
                      help='Path to the second model for dual model evaluation')
    
    args = parser.parse_args()
    training_mode = args.training_mode
    print(f"Using training mode: {training_mode}")
    
    # Determine if we need dual input based on training mode
    dual_input = training_mode in ["dual_input_max_loss", "dual_models"]
    
    # Create data loaders
    test_loader = create_data_loaders(
        test_dir='data/test',
        window_size=30,
        stride=1,
        batch_size=4096,
        num_workers=4,
        dual_input=dual_input
    )
    
    wandb_disabled = True
    
    if training_mode == "standard":
        input_channels = 15
        # Original single model mode
        model_name = 'ResNet18_Standard'
        model = resnet18(input_channels=input_channels, dropout_rate=0.3)
        
        # Evaluate each model
        print(f"\nTraining {model_name}")
        # Initialize wandb
        wandb.init(
            project="robot-trajectory-prediction",
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "architecture": model_name,
                "training_mode": training_mode,
                "batch_size": test_loader.batch_size,
                "window_size": test_loader.dataset.window_size,
                "stride": test_loader.dataset.stride,
                "state_dimensions": input_channels  # Original state dimension
            },
            mode="disabled" if wandb_disabled else None
        )
        # Load a single model
        model = load_model(
            args.model_path,
            args.model_architecture,
            input_channels
        )
        # Evaluate on test set
        test_mse = evaluate_model(model, test_loader, model_name, training_mode=training_mode)
        print(f"Test MSE for {model_name}: {test_mse:.4f}")
        wandb.log({
            "test_mse": test_mse
        })
        wandb.finish()
            
    elif training_mode == "dual_input_max_loss":
        # Set input channels based on training mode
        input_channels = 9  # 6 (RT) + 3 (EE position) for each camera
        # Mode 1: Single model with dual inputs, max loss
        model_name = 'ResNet18_DualInput'
        model = resnet18(input_channels=input_channels, dropout_rate=0.3)
        
        # Initialize wandb
        wandb.init(
            project="robot-trajectory-prediction",
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "architecture": model_name,
                "training_mode": training_mode,
                "batch_size": test_loader.batch_size,
                "window_size": test_loader.dataset.window_size,
                "stride": test_loader.dataset.stride,
                "state_dimensions": input_channels
            },
            mode="disabled" if wandb_disabled else None
        )
        # Load a single model
        model = load_model(
            args.model_path,
            args.model_architecture,
            input_channels
        )
        # Evaluate on test set
        test_mse = evaluate_model(model, test_loader, model_name, training_mode=training_mode)
        print(f"Test MSE for {model_name}: {test_mse:.4f}")
        wandb.log({
            "test_mse": test_mse
        })
        wandb.finish()
        
    elif training_mode == "dual_models":
        # Set input channels based on training mode
        input_channels = 9  # 6 (RT) + 3 (EE position) for each camera
        # Mode 2: Two separate models for dual inputs
        model_name = 'ResNet18_Ensemble'
        
        # Create separate models for each camera
        model_cam1 = resnet18(input_channels=input_channels, dropout_rate=0.3)
        model_cam2 = resnet18(input_channels=input_channels, dropout_rate=0.3)
        
        # Initialize wandb
        wandb.init(
            project="robot-trajectory-prediction",
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "architecture": model_name,
                "training_mode": training_mode,
                "batch_size": test_loader.batch_size,
                "window_size": test_loader.dataset.window_size,
                "stride": test_loader.dataset.stride,
                "state_dimensions": input_channels
            },
            mode="disabled" if wandb_disabled else None
        )
        
        # Load both models for dual model evaluation
        model_cam1 = load_model(
            args.model_path,
            args.model_architecture,
            args.input_channels
        )
        
        model_cam2 = load_model(
            args.dual_model_path_cam2,
            args.model_architecture,
            args.input_channels
        )
        
        # Evaluate the ensemble on testing set
        test_mse_ensemble = evaluate_dual_models(model_cam1, model_cam2, test_loader, model_name)
        print(f"Test MSE for Ensemble: {test_mse_ensemble:.4f}")
        
        wandb.log({
            "test_mse_ensemble": test_mse_ensemble
        })
        
        wandb.finish()