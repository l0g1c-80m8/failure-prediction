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

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, start_idx, end_idx = self.window_indices[idx]
        episode_path = os.path.join(self.data_dir, self.episode_files[file_idx])
        episode_data = np.load(episode_path, allow_pickle=True)
        
        # Extract window data
        window_data = episode_data[start_idx:end_idx]
        
        # Get states sequence (shape: window_size x 9)
        states = np.stack([frame['state'] for frame in window_data])
        
        # Get the action for the last timestep
        action = window_data[-1]['action']
        
        return {
            'states': torch.FloatTensor(states).transpose(0, 1),  # Transform to (9 x window_size) for 1D convolution
            'action': torch.FloatTensor([action])  # Single scalar target
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

def train_model(model, train_loader, val_loader, model_name, num_epochs=50, 
              warmup_epochs=10, initial_lr=0.01, min_lr=1e-6):
    """Train the model and validate periodically with progress tracking and logging."""
    # Initialize wandb
    wandb.init(
        project="robot-trajectory-prediction",
        name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "architecture": model_name,
            "epochs": num_epochs,
            "warmup_epochs": warmup_epochs,
            "initial_lr": initial_lr,
            "min_lr": min_lr,
            "batch_size": train_loader.batch_size,
            "window_size": train_loader.dataset.window_size,
            "stride": train_loader.dataset.stride
        }
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=100,  # Adjust this for different warm restart periods
    #     T_mult=2,  # Adjust this to change how the period grows
    #     eta_min=1e-6  # Minimum learning rate
    # )
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
    
    best_val_loss = float('inf')
    global_step = 0
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        
        # Create progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch in train_pbar:
            states = batch['states'].to(device)
            actions = batch['action'].to(device)
            
            # Update learning rate
            current_lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            loss.backward()
            optimizer.step()
            # scheduler.step()  # Step the scheduler after each optimization step
            
            train_loss += loss.item()
            global_step += 1
            
            # Update training progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        # Create progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                states = batch['states'].to(device)
                actions = batch['action'].to(device)
                
                outputs = model(states)
                loss = criterion(outputs, actions)
                val_loss += loss.item()
                
                # Update validation progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')
            wandb.save(f'best_model_{model_name}.pth')
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'lr': f'{current_lr:.6f}'
        })
    
    wandb.finish()

def evaluate_model(model, test_loader, model_name):
    """Evaluate the model on test data with progress tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_mse = 0
    
    # Create progress bar for evaluation
    eval_pbar = tqdm(test_loader, desc=f"Evaluating {model_name}")
    
    with torch.no_grad():
        for batch in eval_pbar:
            states = batch['states'].to(device)
            actions = batch['action'].to(device)
            
            outputs = model(states)
            mse = nn.MSELoss()(outputs, actions).item()
            total_mse += mse
            
            # Update progress bar
            eval_pbar.set_postfix({'mse': f'{mse:.4f}'})
    
    avg_mse = total_mse / len(test_loader)
    return avg_mse

if __name__ == "__main__":
    # Initialize wandb
    wandb.login()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dir='data/train',
        val_dir='data/val',
        window_size=2,
        stride=5,
        batch_size=32
    )
    
    # Example of using different ResNet architectures
    models = {
        'ResNet18': resnet18(input_channels=9)
        # 'ResNet34': resnet34(input_channels=9),
        # 'ResNet50': resnet50(input_channels=9),
        # 'ResNet101': resnet101(input_channels=9),
        # 'ResNet152': resnet152(input_channels=9)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}")
        # Train the model
        train_model(model, train_loader, val_loader, name, num_epochs=1000,
                    warmup_epochs=10,
                    initial_lr=0.01,
                    min_lr=1e-6)
        
        # Evaluate on validation set
        val_mse = evaluate_model(model, val_loader, name)
        print(f"Validation MSE for {name}: {val_mse:.4f}")
