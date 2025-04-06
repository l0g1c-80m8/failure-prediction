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

from resnet_models import resnet18, resnet34, resnet50, resnet101, resnet152


class RobotTrajectoryDataset(Dataset):
    def __init__(self, data_dir, window_size=10, stride=1):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        
        # Get list of all .npy files in directory
        self.episode_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        
        # Pre-calculate indices for all windows
        self.window_indices = []  # (file_idx, start_idx, end_idx)
        
        for file_idx, episode_file in enumerate(self.episode_files):
            episode_data = np.load(os.path.join(data_dir, episode_file), allow_pickle=True)
            n_frames = len(episode_data)
            
            for start_idx in range(0, n_frames - window_size + 1, stride):
                end_idx = start_idx + window_size
                self.window_indices.append((file_idx, start_idx, end_idx))
        
        # Pre-load all episodes into memory
        self.episodes = {}
        for file_idx, episode_file in enumerate(self.episode_files):
            episode_path = os.path.join(data_dir, episode_file)
            self.episodes[file_idx] = np.load(episode_path, allow_pickle=True)

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, start_idx, end_idx = self.window_indices[idx]
        episode_data = self.episodes[file_idx]  # Get from memory instead of disk
        
        # Extract window data
        window_data = episode_data[start_idx:end_idx]
        
        # Get states sequence (shape: window_size x 19)
        states = np.stack([frame['state'] for frame in window_data])        
        
        # Get the risk for the last timestep
        # Convert to numpy array first, then to tensor
        risk = np.array(window_data[-1]['risk'], dtype=np.float32)
        
        return {
            'states': torch.FloatTensor(states).transpose(0, 1),  # Transform to (19 x window_size) for 1D convolution
            'risk': torch.FloatTensor(risk)  # Convert numpy array to tensor
        }

def create_data_loaders(train_dir, val_dir, window_size=10, stride=1, batch_size=32, num_workers=4):
    """Create train and validation data loaders."""
    
    # Create datasets
    train_dataset = RobotTrajectoryDataset(
        data_dir=train_dir,
        window_size=window_size,
        stride=stride
    )
    
    val_dataset = RobotTrajectoryDataset(
        data_dir=val_dir,
        window_size=window_size,
        stride=stride
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

def train_model(model, train_loader, val_loader, model_name, num_epochs=50, 
              warmup_epochs=10, initial_lr=0.01, min_lr=1e-6):
    """Train the model and validate periodically with progress tracking and logging."""
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Compute total steps for warmup and decay
    n_steps_per_epoch = len(train_loader)
    n_warmup_steps = warmup_epochs * n_steps_per_epoch
    n_total_steps = num_epochs * n_steps_per_epoch
    
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            states = batch['states'].to(device)
            risks = batch['risk'].to(device)
            
            # Update learning rate
            current_lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, risks)
            
            loss.backward()
            optimizer.step()
            # scheduler.step()  # Step the scheduler after each optimization step
            
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
                states = batch['states'].to(device)
                risks = batch['risk'].to(device)
                
                outputs = model(states)
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
    

def evaluate_model(model, test_loader, model_name):
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
    try:
        os.environ['QT_X11_NO_MITSHM'] = '1'
        # Initialize wandb
        wandb.login()
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_dir='data/train',
            val_dir='data/val',
            window_size=1,
            stride=1,
            batch_size=1016,
            num_workers=8
        )
        
        # Example of using different ResNet architectures
        models = {
            'ResNet18': resnet18(input_channels=19)
            # 'ResNet34': resnet34(input_channels=19),
            # 'ResNet50': resnet50(input_channels=19),
            # 'ResNet101': resnet101(input_channels=19),
            # 'ResNet152': resnet152(input_channels=19)
        }

        wandb_disabled = False
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}")
            # Initialize wandb
            num_epochs=1000
            warmup_epochs=10
            initial_lr=0.01
            min_lr=1e-6
            wandb.init(
                project="robot-trajectory-prediction",
                name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "architecture": name,
                    "epochs": num_epochs,
                    "warmup_epochs": warmup_epochs,
                    "initial_lr": initial_lr,
                    "min_lr": min_lr,
                    "batch_size": train_loader.batch_size,
                    "window_size": train_loader.dataset.window_size,
                    "stride": train_loader.dataset.stride
                },
                mode="disabled" if wandb_disabled else None
            )
            # Train the model
            train_model(model, train_loader, val_loader, name, num_epochs=num_epochs,
                        warmup_epochs=warmup_epochs,
                        initial_lr=initial_lr,
                        min_lr=min_lr)
            
            # Evaluate on validation set
            val_mse = evaluate_model(model, val_loader, name)
            print(f"Validation MSE for {name}: {val_mse:.4f}")
            wandb.finish()
    except Exception as e:
        print(f"An error occurred: {e}")
