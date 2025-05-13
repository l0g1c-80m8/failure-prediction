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

from resnest_models_multimodal import multimodal_resnet18, multimodal_resnet34, multimodal_resnet50, multimodal_resnet101, multimodal_resnet152
# Import here to avoid dependency issues
from PIL import Image


class RobotTrajectoryDataset(Dataset):
    def __init__(self, data_dir, window_size=10, stride=1, preload=False, cache_indices=True, dual_input=False):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.preload = preload
        self.dual_input = dual_input  # Flag to determine if we need both camera inputs separately
        
        # Get list of all .npy files in directory
        self.episode_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        
        # Cache for loaded episodes
        self.episodes_cache = {}
        
        # Calculate total number of windows and their locations
        print(f"Calculating indices for {len(self.episode_files)} episode files...")
        
        # Either calculate and cache indices or load them if already cached
        indices_cache_path = os.path.join(data_dir, f"indices_cache_w{window_size}_s{stride}.npy")
        
        if cache_indices and os.path.exists(indices_cache_path):
            print(f"Loading cached indices from {indices_cache_path}")
            self.window_indices = np.load(indices_cache_path, allow_pickle=True).tolist()
            self.n_samples = len(self.window_indices)
        else:
            # For progress tracking during initialization
            print("Calculating window indices...")
            self.window_indices = []
            
            for file_idx, episode_file in enumerate(tqdm(self.episode_files, desc="Processing episodes")):
                episode_path = os.path.join(data_dir, episode_file)
                # Just load the length of the episode to calculate indices
                episode_data = np.load(episode_path, allow_pickle=True)
                n_frames = len(episode_data)
                
                for start_idx in range(0, n_frames - window_size + 1, stride):
                    end_idx = start_idx + window_size
                    self.window_indices.append((file_idx, start_idx, end_idx))
                
                # Clear memory
                del episode_data
            
            self.n_samples = len(self.window_indices)
            print(f"Total windows: {self.n_samples}")
            
            # Cache indices for future use
            if cache_indices:
                print(f"Caching indices to {indices_cache_path}")
                np.save(indices_cache_path, self.window_indices)
        
        # Optionally preload all data (only if requested)
        if preload:
            print("Preloading all episodes into memory...")
            for file_idx, episode_file in enumerate(tqdm(self.episode_files, desc="Preloading episodes")):
                episode_path = os.path.join(data_dir, episode_file)
                self.episodes_cache[file_idx] = np.load(episode_path, allow_pickle=True)
            print("Preloading complete")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        file_idx, start_idx, end_idx = self.window_indices[idx]
        
        # Load episode data if not already in cache
        if file_idx not in self.episodes_cache:
            episode_path = os.path.join(self.data_dir, self.episode_files[file_idx])
            episode_data = np.load(episode_path, allow_pickle=True)
        else:
            episode_data = self.episodes_cache[file_idx]
        
        # Extract window data
        window_data = episode_data[start_idx:end_idx]
        
        # Get states sequence
        states = np.stack([frame['state'] for frame in window_data])
        # print("states.shape", states.shape) # (30, 15)

        rgb_cam1 = np.stack([frame['full_top_frame_rgb'] for frame in window_data])
        # print("rgb_cam1.shape", rgb_cam1.shape) # rgb_cam1.shape (30, 640, 640, 3)
        rgb_cam2 = np.stack([frame['full_front_frame_rgb'] for frame in window_data])
        # print("rgb_cam2.shape", rgb_cam2.shape) # rgb_cam1.shape (30, 640, 640, 3)
        # Pre-process RGB images - resize and normalize
        # Convert from (time, H, W, C) to PyTorch format (time, C, H, W)
        # Also resize from 640x640 to a more manageable size like 64x64
        # We will handle image resizing in the actual implementation
        processed_rgb_cam1 = self.preprocess_images(rgb_cam1, width=64, height=64) 
        processed_rgb_cam2 = self.preprocess_images(rgb_cam2, width=64, height=64)
        
        states_cam1 = states[:, :6]
        states_cam2 = states[:, 6:12]
        ee_pos = states[:, 12:15]  # End effector position (3 values)

        # Concatenate the 3 EE_pos values to the first camera's 6 values (RT) (Total 9 values)
        states_cam1 = np.concatenate((states_cam1, ee_pos), axis=1)

        # Concatenate the 3 EE_pos values to the second camera's 6 values (RT) (Total 9 values)
        states_cam2 = np.concatenate((states_cam2, ee_pos), axis=1)

        # Get the risk for the last timestep
        risk = np.array(window_data[-1]['risk'], dtype=np.float32)
        
        # If we haven't preloaded everything, clean up this episode to save memory
        if not self.preload and file_idx not in self.episodes_cache:
            del episode_data
        
        if self.dual_input:
            return {
                'states_cam1': torch.FloatTensor(states_cam1).transpose(0, 1),
                'states_cam2': torch.FloatTensor(states_cam2).transpose(0, 1),
                'rgb_cam1': processed_rgb_cam1,
                'rgb_cam2': processed_rgb_cam2,
                'risk': torch.FloatTensor(risk)
            }
        else:
            return {
                'states': torch.FloatTensor(states).transpose(0, 1),
                'rgb_cam1': processed_rgb_cam1,
                'rgb_cam2': processed_rgb_cam2,
                'risk': torch.FloatTensor(risk)
            }
        
    def preprocess_images(self, images, width=64, height=64):
        """
        Preprocess RGB images:
        1. Resize from 640x640 to 64x64 to reduce computation
        2. Convert from uint8 to float32 and normalize to [0, 1]
        3. Convert from (time, H, W, C) to (time, C, H, W) for PyTorch
        """
        
        # Create a transformation pipeline
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((width, height)),  # Reduce size to 64x64
            transforms.ToTensor(),  # This also normalizes to [0, 1]
        ])
        
        # Process each frame in the sequence
        processed_images = []
        for t in range(images.shape[0]):
            # Process the image
            img = images[t]
            # Convert to PIL Image and apply transformations
            img_tensor = transform(img)
            processed_images.append(img_tensor)
        
        # Stack all frames into a single tensor of shape (time, C, H, W)
        return torch.stack(processed_images)

def create_data_loaders(train_dir, val_dir, window_size=10, stride=1, batch_size=32, num_workers=4, preload=False, dual_input=False):
    """Create train and validation data loaders with optimization options."""
    
    # Create datasets with optimizations
    train_dataset = RobotTrajectoryDataset(
        data_dir=train_dir,
        window_size=window_size,
        stride=stride,
        preload=preload,  # Set to True only if you have enough RAM
        cache_indices=True,  # Use cached indices if available
        dual_input=dual_input
    )
    
    val_dataset = RobotTrajectoryDataset(
        data_dir=val_dir,
        window_size=window_size,
        stride=stride,
        preload=preload,
        cache_indices=True,
        dual_input=dual_input
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def compute_metrics(outputs, targets):
    """Compute MSE and normalized MSE metrics."""
    mse = nn.MSELoss()(outputs, targets).item()
    
    return {
        'mse': mse
    }

def prepare_rgb_data(rgb_data):
    """
    Transform RGB data from [batch, time, C, H, W] to [batch, time, H, W, C]
    which is the format expected by RGBEncoder
    """
    # Permute from [batch, time, C, H, W] to [batch, time, H, W, C]
    return rgb_data.permute(0, 1, 3, 4, 2)

def train_model(model, train_loader, val_loader, model_name, num_epochs=50, 
              warmup_epochs=10, initial_lr=0.01, min_lr=1e-6, training_mode="standard"):
    """Train the model and validate periodically with progress tracking and logging."""
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0001)
    
    # Compute total steps for warmup and decay
    n_steps_per_epoch = len(train_loader)
    n_warmup_steps = warmup_epochs * n_steps_per_epoch
    n_total_steps = num_epochs * n_steps_per_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_lr(step):
        if step < n_warmup_steps:
            # Linear warmup
            return min_lr + (initial_lr - min_lr) * (step / n_warmup_steps)
        else:
            # Cosine decay
            decay_steps = n_total_steps - n_warmup_steps
            decay_step = step - n_warmup_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / decay_steps))
            return min_lr + (initial_lr - min_lr) * cosine_decay
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DistributedDataParallel!")
        # Set up process group
        torch.distributed.init_process_group(backend='nccl')
        # Create model on current device
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                        device_ids=[local_rank],
                                                        output_device=local_rank)
    else:
        model = model.to(device)
    
    best_val_mse = float('inf')
    global_step = 0
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:

        # Training phase
        model.train()
        train_metrics = {
            'loss': 0.0,
            'mse': 0.0
        }
        
        # Create progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch in train_pbar:
            # Update learning rate
            current_lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            
            # Get RGB data and prepare it for the model
            rgb_cam1 = prepare_rgb_data(batch['rgb_cam1'].to(device))
            rgb_cam2 = prepare_rgb_data(batch['rgb_cam2'].to(device))

            if training_mode == "dual_input_max_loss":
                # Mode 1: Single model with dual inputs, max loss
                states_cam1 = batch['states_cam1'].to(device)
                states_cam2 = batch['states_cam2'].to(device)
                risks = batch['risk'].to(device)
                
                # Forward pass for both camera inputs
                outputs_cam1 = model(states_cam1, rgb_cam1, rgb_cam2)
                outputs_cam2 = model(states_cam2, rgb_cam1, rgb_cam2)
                
                # Calculate loss for each output and take the maximum
                loss_cam1 = criterion(outputs_cam1, risks)
                loss_cam2 = criterion(outputs_cam2, risks)
                loss = torch.max(loss_cam1, loss_cam2)
                
                # For metrics tracking, use the maximum prediction
                outputs = torch.max(outputs_cam1, outputs_cam2)
            else:
                # Standard mode: Single input (combined or specific camera)
                states = batch['states'].to(device)
                risks = batch['risk'].to(device)
                
                # Forward pass with all modalities
                outputs = model(states, rgb_cam1, rgb_cam2)
                loss = criterion(outputs, risks)
            
            loss.backward()
            optimizer.step()
            
            print_gpu_memory_stats()
            
            # Compute batch metrics
            batch_metrics = compute_metrics(outputs, risks)
            
            # Update running metrics
            train_metrics['loss'] += loss.item()
            train_metrics['mse'] += batch_metrics['mse']
            
            global_step += 1
            
            # Update training progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}"
            })
        
        # Compute average training metrics
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_metrics = {
            'loss': 0.0,
            'mse': 0.0
        }

        # Create progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                if training_mode == "dual_input_max_loss":
                    # Mode 1: Single model with dual inputs, max loss
                    states_cam1 = batch['states_cam1'].to(device)
                    states_cam2 = batch['states_cam2'].to(device)
                    risks = batch['risk'].to(device)
                    
                    # Forward pass for both camera inputs
                    outputs_cam1 = model(states_cam1)
                    outputs_cam2 = model(states_cam2)
                    
                    # For validation, use the maximum prediction
                    outputs = torch.max(outputs_cam1, outputs_cam2)
                    loss = criterion(outputs, risks)
                else:
                    # Standard mode: Single input
                    states = batch['states'].to(device)
                    risks = batch['risk'].to(device)
                    outputs = model(states, rgb_cam1, rgb_cam2)
                    loss = criterion(outputs, risks)

                # Compute batch metrics
                batch_metrics = compute_metrics(outputs, risks)
                
                # Update running metrics
                val_metrics['loss'] += loss.item()
                val_metrics['mse'] += batch_metrics['mse']
                
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })

        # Compute average validation metrics
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'learning_rate': current_lr,
            'train_loss': train_metrics['loss'],
            'train_mse': train_metrics['mse'],
            'val_loss': val_metrics['loss'],
            'val_mse': val_metrics['mse']
        })
        
        # Save best model based on validation mse
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')
            wandb.save(f'best_model_{model_name}.pth')

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'lr': f"{current_lr:.6f}"
        })
    
    return best_val_mse  # Return best validation MSE for reference

def train_dual_models(model_cam1, model_cam2, train_loader, val_loader, model_name_prefix, 
                      num_epochs=50, warmup_epochs=10, initial_lr=0.01, min_lr=1e-6):
    """Train two separate models for the two camera inputs."""
    
    # First, train the model for camera 1
    print(f"\nTraining model for Camera 1")
    best_val_mse_cam1 = train_model(
        model_cam1, 
        train_loader, 
        val_loader, 
        f"{model_name_prefix}_cam1", 
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        initial_lr=initial_lr,
        min_lr=min_lr,
        training_mode="dual_models"  # Pass the training mode
    )
    
    # Then, train the model for camera 2
    print(f"\nTraining model for Camera 2")
    best_val_mse_cam2 = train_model(
        model_cam2, 
        train_loader, 
        val_loader, 
        f"{model_name_prefix}_cam2", 
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        initial_lr=initial_lr,
        min_lr=min_lr,
        training_mode="dual_models"  # Pass the training mode
    )
    
    return best_val_mse_cam1, best_val_mse_cam2

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
            # Get RGB data and prepare it for the model
            rgb_cam1 = prepare_rgb_data(batch['rgb_cam1'].to(device))
            rgb_cam2 = prepare_rgb_data(batch['rgb_cam2'].to(device))
            
            if training_mode == "dual_input_max_loss":
                # Mode 1: Single model with dual inputs
                states_cam1 = batch['states_cam1'].to(device)
                states_cam2 = batch['states_cam2'].to(device)
                risks = batch['risk'].to(device)
                
                # Forward pass for both camera inputs with RGB data
                outputs_cam1 = model(states_cam1, rgb_cam1, rgb_cam2)
                outputs_cam2 = model(states_cam2, rgb_cam1, rgb_cam2)
                
                # Use maximum prediction
                outputs = torch.max(outputs_cam1, outputs_cam2)
            else:
                # Standard mode: Single input
                states = batch['states'].to(device)
                risks = batch['risk'].to(device)
                outputs = model(states, rgb_cam1, rgb_cam2)
            
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
            # Get RGB data and prepare it for the model
            rgb_cam1 = prepare_rgb_data(batch['rgb_cam1'].to(device))
            rgb_cam2 = prepare_rgb_data(batch['rgb_cam2'].to(device))
            risks = batch['risk'].to(device)
            
            # Forward pass for both models with RGB data
            outputs_cam1 = model_cam1(states_cam1, rgb_cam1, rgb_cam2)
            outputs_cam2 = model_cam2(states_cam2, rgb_cam1, rgb_cam2)
            
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

# Add this function to monitor GPU memory
def print_gpu_memory_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # GB
            reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
            allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
            free = total_memory - allocated
            print(f"GPU {i}: Total: {total_memory:.2f}GB, Reserved: {reserved:.2f}GB, "
                  f"Allocated: {allocated:.2f}GB, Free: {free:.2f}GB")
            
if __name__ == "__main__":
    os.environ['QT_X11_NO_MITSHM'] = '1'
    # Initialize wandb
    wandb.login()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train risk prediction models')
    parser.add_argument('--training-mode', type=str, default='standard',
                      choices=['standard', 'dual_input_max_loss', 'dual_models'],
                      help='Training mode to use')
    parser.add_argument('--local-rank', type=int, default=0, 
                      help='Local rank for distributed training')
    
    args = parser.parse_args()
    training_mode = args.training_mode
    print(f"Using training mode: {training_mode}")
    
    # Determine if we need dual input based on training mode
    dual_input = training_mode in ["dual_input_max_loss", "dual_models"]
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dir='data/train',
        val_dir='data/val',
        window_size=30,
        stride=1,
        batch_size=128,
        num_workers=4,
        dual_input=dual_input
    )
    
    wandb_disabled = False
    
    # Common training parameters
    num_epochs = 100
    warmup_epochs = 10
    initial_lr = 0.01
    min_lr = 1e-6
    
    if training_mode == "standard":
        input_channels = 15
        # Original single model mode
        models = {
            'ResNet18': multimodal_resnet18(state_channels=input_channels, 
                                           fusion_type='concat', 
                                           rgb_feature_dim=128, 
                                           dropout_rate=0.3)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}")
            # Initialize wandb
            wandb.init(
                project="robot-trajectory-prediction",
                name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "architecture": name,
                    "training_mode": training_mode,
                    "epochs": num_epochs,
                    "warmup_epochs": warmup_epochs,
                    "initial_lr": initial_lr,
                    "min_lr": min_lr,
                    "batch_size": train_loader.batch_size,
                    "window_size": train_loader.dataset.window_size,
                    "stride": train_loader.dataset.stride,
                    "state_dimensions": input_channels,
                    "rgb_feature_dim": 128,
                    "fusion_type": "concat"
                },
                mode="disabled" if wandb_disabled else None
            )
            
            # Train the model
            train_model(model, train_loader, val_loader, name, 
                        num_epochs=num_epochs,
                        warmup_epochs=warmup_epochs,
                        initial_lr=initial_lr,
                        min_lr=min_lr,
                        training_mode=training_mode)
            
            # Evaluate on validation set
            val_mse = evaluate_model(model, val_loader, name, training_mode=training_mode)
            print(f"Validation MSE for {name}: {val_mse:.4f}")
            wandb.finish()
            
    elif training_mode == "dual_input_max_loss":
        # Set input channels based on training mode
        input_channels = 9  # 6 (RT) + 3 (EE position) for each camera
        # Mode 1: Single model with dual inputs, max loss
        model_name = 'ResNet18_DualInput'
        model = multimodal_resnet18(state_channels=input_channels, 
                                   fusion_type='concat', 
                                   rgb_feature_dim=128, 
                                   dropout_rate=0.3)
        
        # Initialize wandb
        wandb.init(
            project="robot-trajectory-prediction",
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "architecture": "ResNet18",
                "training_mode": training_mode,
                "epochs": num_epochs,
                "warmup_epochs": warmup_epochs,
                "initial_lr": initial_lr,
                "min_lr": min_lr,
                "batch_size": train_loader.batch_size,
                "window_size": train_loader.dataset.window_size,
                "stride": train_loader.dataset.stride,
                "state_dimensions": input_channels,
                "rgb_feature_dim": 128,
                "fusion_type": "concat"
            },
            mode="disabled" if wandb_disabled else None
        )
        
        # Train the model
        train_model(model, train_loader, val_loader, model_name, 
                    num_epochs=num_epochs,
                    warmup_epochs=warmup_epochs,
                    initial_lr=initial_lr,
                    min_lr=min_lr,
                    training_mode=training_mode)
        
        # Evaluate on validation set
        val_mse = evaluate_model(model, val_loader, model_name, training_mode=training_mode)
        print(f"Validation MSE for {model_name}: {val_mse:.4f}")
        wandb.finish()
        
    elif training_mode == "dual_models":
        # Set input channels based on training mode
        input_channels = 9  # 6 (RT) + 3 (EE position) for each camera
        # Mode 2: Two separate models for dual inputs
        model_name_prefix = 'ResNet18_Ensemble'
        
        # Create separate models for each camera, using multimodal models
        model_cam1 = multimodal_resnet18(state_channels=input_channels, 
                                        fusion_type='concat', 
                                        rgb_feature_dim=128, 
                                        dropout_rate=0.3)
        model_cam2 = multimodal_resnet18(state_channels=input_channels, 
                                        fusion_type='concat', 
                                        rgb_feature_dim=128, 
                                        dropout_rate=0.3)
        
        # Initialize wandb
        wandb.init(
            project="robot-trajectory-prediction",
            name=f"{model_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "architecture": "ResNet18",
                "training_mode": training_mode,
                "epochs": num_epochs,
                "warmup_epochs": warmup_epochs,
                "initial_lr": initial_lr,
                "min_lr": min_lr,
                "batch_size": train_loader.batch_size,
                "window_size": train_loader.dataset.window_size,
                "stride": train_loader.dataset.stride,
                "state_dimensions": input_channels,
                "rgb_feature_dim": 128,
                "fusion_type": "concat"
            },
            mode="disabled" if wandb_disabled else None
        )
        
        # Train both models
        best_val_mse_cam1, best_val_mse_cam2 = train_dual_models(
            model_cam1, model_cam2, 
            train_loader, val_loader, 
            model_name_prefix,
            num_epochs=num_epochs,
            warmup_epochs=warmup_epochs,
            initial_lr=initial_lr,
            min_lr=min_lr
        )
        
        # Evaluate the ensemble on validation set
        val_mse_ensemble = evaluate_dual_models(model_cam1, model_cam2, val_loader, model_name_prefix)
        print(f"Validation MSE for Camera 1 model: {best_val_mse_cam1:.4f}")
        print(f"Validation MSE for Camera 2 model: {best_val_mse_cam2:.4f}")
        print(f"Validation MSE for Ensemble: {val_mse_ensemble:.4f}")
        
        wandb.log({
            "val_mse_cam1": best_val_mse_cam1,
            "val_mse_cam2": best_val_mse_cam2,
            "val_mse_ensemble": val_mse_ensemble
        })
        
        wandb.finish()