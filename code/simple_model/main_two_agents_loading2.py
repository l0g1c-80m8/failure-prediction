import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

# os.environ.setdefault("MASTER_PORT", "29601")

from tqdm import tqdm
import wandb
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

import random 

from resnet_models import resnet18, resnet34, resnet50, resnet101, resnet152
import argparse


class RobotTrajectoryDataset(Dataset):
    def __init__(self, data_dir, window_size=10, stride=1, use_cache=True, cache_size=1000, batch_size=1024, sub_batch_size=128):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.cache = {}
        # self.episodes = {}
        self.cache_hits = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

        print(f"Loading data from {data_dir} with window size {window_size} and stride {stride}")

        os.makedirs(f"{data_dir}/sub_batches/", exist_ok=True)
        
        # Get list of all .npy files in directory
        self.episode_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        random.shuffle(self.episode_files)
        total_files = len(self.episode_files)
        self.total_samples = 0

        splitted_episodes = [self.episode_files[i:i + cache_size] for i in range(0, total_files, cache_size)]
        for split_id, split_list in enumerate(splitted_episodes):
            bulk_samples = []

            for episode_file in tqdm(split_list, desc=f"Createting sub-batches from list {split_id+1}/{len(splitted_episodes)}", total=len(split_list)):
                if episode_file == "indices_cache_w30_s1.npy":
                    continue
                episode_data = np.load(os.path.join(data_dir, episode_file), allow_pickle=True)
                # self.episodes[file_idx] = episode_data # Pre-load all episodes into memory
                n_frames = len(episode_data)
                
                for start_idx in range(0, n_frames - window_size + 1, stride):
                    end_idx = start_idx + window_size
                    # self.window_indices.append((episode_file, start_idx, end_idx))
                    bulk_samples.append(self.process_sample(episode_data[start_idx:end_idx]))
            
            random.shuffle(bulk_samples)
            for idx, sub_batch in enumerate([bulk_samples[i:i + sub_batch_size] for i in range(0, len(bulk_samples), sub_batch_size)]):
                sub_batch = np.array(sub_batch)
                np.save(os.path.join(data_dir, f'sub_batches/{split_id}_sub_batch_{idx}.npy'), sub_batch)

            self.total_samples += len(bulk_samples)

        self.subbatch_files = [f for f in os.listdir(f"{data_dir}/sub_batches/") if f.endswith('.npy')]

        self.loaded_batch_samples = []

    def load_episode(self, file_idx):
        
        if self.use_cache and file_idx in self.cache:
            # print(f"Loading episode {file_idx} from cache")
            self.cache_hits += 1
            return self.cache[file_idx]
        
        path = os.path.join(self.data_dir, self.episode_files[file_idx])
        # print(f"Loading episode {file_idx} from disk: {path}")
        episode_data = np.load(path, allow_pickle=True)
        data = [[frame['state'], frame['risk']] for frame in episode_data]

        if self.use_cache:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))  # remove oldest
            self.cache[file_idx] = data

        return data
    
    @staticmethod
    def process_sample(window_data):
        # Get states sequence (shape: window_size x 15)
        states = np.stack([frame['state'] for frame in window_data]) # (window_size x 15) 
        states_cam1 = states[:, :6]
        states_cam2 = states[:, 6:12]
        ee_pos = states[:, 12:15]  # End effector position (3 values)

        # Concatenate the 3 EE_pos values (15, 16, 17) to the first camera's 6 values (RT) (Total 9 values)
        states_cam1 = np.concatenate((states_cam1, ee_pos), axis=1)

        # Concatenate the 3 EE_pos values (15, 16, 17) to the second camera's 6 values (RT) (Total 9 values)
        states_cam2 = np.concatenate((states_cam2, ee_pos), axis=1)
        
        # Get the risk for the last timestep
        # Convert to numpy array first, then to tensor
        risk = np.array(window_data[-1]['risk'], dtype=np.float32)

        return {"states_cam1":torch.FloatTensor(states_cam1).transpose(0, 1), "states_cam2":torch.FloatTensor(states_cam2).transpose(0, 1), "risk":torch.FloatTensor(risk)}

    def load_sub_batches(self):
        self.loaded_batch_samples = []
        for sub_batch_file in random.sample(self.subbatch_files, self.batch_size//self.sub_batch_size):
            sub_batch_path = os.path.join(self.data_dir, "sub_batches", sub_batch_file)
            sub_batch_data = np.load(sub_batch_path, allow_pickle=True)
            self.loaded_batch_samples.extend(sub_batch_data)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if len(self.loaded_batch_samples) == 0:
            self.load_sub_batches()
        
        
        try:
            # Get a sample from the loaded batch
            data_sample = self.loaded_batch_samples[idx%len(self.loaded_batch_samples)]
        except Exception as e:
            print(f"Error:{e}", idx%self.batch_size, len(self.loaded_batch_samples))

        return data_sample


def create_datasets(data_dir, window_size=10, stride=1, split="train"):
    datasplit_dir=f'{data_dir}/{split}'

    print(f"Creating data loaders with window size {window_size} and stride {stride}")
    """Create train and validation data loaders."""
    
    # Create datasets
    dataset = RobotTrajectoryDataset(
        data_dir=datasplit_dir,
        window_size=window_size,
        stride=stride
    )
    # Save dataset
    torch.save(dataset, f'{data_dir}/{split}_dataset.pt')
    print(f"Number of {split.upper()} samples: {len(dataset)}")

    print(f"Dataset saved as {split}_dataset.pt and val_dataset.pt in {data_dir}")

def load_datasets_to_dataloaders(trainset_path, testset_path, batch_size=32, num_workers=4):

    """Load datasets for training and testing."""
    train_dataset = torch.load(trainset_path, weights_only=False)
    val_dataset = torch.load(testset_path, weights_only=False)
    print(f"Loaded train dataset with {len(train_dataset)} samples")
    print(f"Loaded test dataset with {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4
    )

    return train_loader, val_loader

def compute_metrics(outputs, targets):
    """Compute MSE and normalized MSE metrics."""
    mse = nn.MSELoss()(outputs, targets).item()
    
    return {
        'mse': mse
    }

def train_model(model, train_loader, val_loader, model_name, camera_name, run_name, num_epochs=50, 
              warmup_epochs=10, initial_lr=0.01, min_lr=1e-6, patience=10):
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
    
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            
            states = batch[f'states_{camera_name}'].to(device)
            risks = batch['risk'].to(device)

            # print(f"Input tensor device: {batch['states'].device}")
            # print(f"Model's first parameter device: {next(model.parameters()).device}")

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
                states = batch[f'states_{camera_name}'].to(device)
                risks = batch['risk'].to(device)
                
                outputs = model(states)
                # Compute loss
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
            torch.save(model.state_dict(), f'runs/{run_name}/best_model_{model_name}.pth')
            wandb.save(f'{run_name}/best_model_{model_name}.pth')
            patient_epoch = 0
        else:
            patient_epoch += 1
            if patient_epoch >= patience:
                print(f"Early stopping at epoch {epoch + 1} ... best val mse: {best_val_mse:.4f}")
                break

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'lr': f"{current_lr:.6f}"
        })

def evaluate_model(model_cam1, model_cam2, test_loader, model_name):
    """Evaluate the model on test data with progress tracking and visualization."""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cam1.eval()
    model_cam2.eval()
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
            states_cam1 = batch['states_cam1'].to(device)
            states_cam2 = batch['states_cam2'].to(device)
            risks = batch['risk'].to(device)
            
            outputs_cam1 = model_cam1(states_cam1)
            outputs_cam2 = model_cam2(states_cam2)
            outputs = torch.maximum(outputs_cam1, outputs_cam2) # takes the maximum elements from both the outputs

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
    # print("predictions", predictions.shape)
    # print("ground_truth", ground_truth.shape)

    assert predictions.shape == ground_truth.shape, "Shape mismatch"
    
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a ResNet model for robot trajectory prediction.")
    parser.add_argument('--data_dir', type=str, default='/data/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/data', help='Directory for training data')
    parser.add_argument('--mode', type=str, help='Directory for validation data')
    parser.add_argument('--camera_name', type=str, default=None, help='Camera name for training')
    parser.add_argument('--window_size', type=int, default=1, help='Window size for data')
    parser.add_argument('--stride', type=int, default=1, help='Stride for data')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')

    parser.add_argument("--ckpt_1", '--model_checkpoint_cam1', type=str, default=None, help='Path to model checkpoint (top cam (1)) for evaluation')
    parser.add_argument("--ckpt_2",'--model_checkpoint_cam2', type=str, default=None, help='Path to model checkpoint (top cam (2)) for evaluation')

    parser.add_argument('--custom_tag', type=str, default="", help='Specific tag you want to add to the run name')

    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training") #ignore this for now
    
    args = parser.parse_args()

    os.environ['QT_X11_NO_MITSHM'] = '1'
    # Initialize wandb
    wandb.login()
    
    # # Create data loaders
    # # if not os.path.exists(f'{args.data_dir}/train/sub_batches/'):
    # os.makedirs(f'{args.data_dir}/train/sub_batches/', exist_ok=True)
    # print("Creating Train batch files...")
    # create_datasets(
    #     data_dir=args.data_dir,
    #     window_size= args.window_size,
    #     stride=args.stride,
    #     split="train"
    # )
    # # if not os.path.exists(f'{args.data_dir}/val/sub_batches/'):
    # os.makedirs(f'{args.data_dir}/val/sub_batches/', exist_ok=True)
    # print("Creating Val batch files...")
    # create_datasets(
    #     data_dir=args.data_dir,
    #     window_size= args.window_size,
    #     stride=args.stride,
    #     split="val"
    # )
    
    print("Datasets already exist. Loading from disk...")
    train_loader, val_loader = load_datasets_to_dataloaders(
        trainset_path=f'{args.data_dir}/train_dataset.pt',
        testset_path=f'{args.data_dir}/val_dataset.pt',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Example of using different ResNet architectures
    models = {
        'ResNet18': resnet18(input_channels=9)
        # 'ResNet34': resnet34(input_channels=9),
        # 'ResNet50': resnet50(input_channels=9),
        # 'ResNet101': resnet101(input_channels=9),
        # 'ResNet152': resnet152(input_channels=9)
    }

    wandb_disabled = False
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}")
        # Initialize wandb
        num_epochs=args.num_epochs
        warmup_epochs=args.warmup_epochs
        initial_lr=args.initial_lr
        min_lr=args.min_lr
        patience=args.patience

        assert args.mode in ["train", "eval"], "mode must be either 'train' or 'eval'"

        if args.mode == "train":
            assert args.camera_name in ["cam1", "cam2"], "camera_name must be either 'cam1' or 'cam2'"
            tag = args.camera_name
        elif args.mode == "eval":
            assert args.ckpt_1 is not None, "model checkpoint for cam1 must be provided for evaluation"
            assert args.ckpt_2 is not None, "model checkpoint for cam2 must be provided for evaluation"
            tag = "eval"
        # assert camera_name in ["cam1", "cam2"], "camera_name must be either 'cam1' or 'cam2'"
        run_name = f"{model_name}_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.custom_tag}"

        os.makedirs(f"runs/{run_name}", exist_ok=True)

        # Get local rank for distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize wandb
        wandb.init(
            project="robot-trajectory-prediction",
            name=f"{run_name}_GPU{local_rank}",
            config={
                "architecture": model_name,
                "epochs": num_epochs,
                "warmup_epochs": warmup_epochs,
                "initial_lr": initial_lr,
                "min_lr": min_lr,
                "batch_size": train_loader.batch_size,
                "window_size": train_loader.dataset.window_size,
                "stride": train_loader.dataset.stride,
                "state_dimensions": 6+3  # Update this to reflect the new state dimension
            },
            mode="disabled" if wandb_disabled else None
        )
        if args.mode == "train":
            # Train the model
            train_model(model, train_loader, val_loader, model_name, args.camera_name, run_name, num_epochs=num_epochs,
                        warmup_epochs=warmup_epochs,
                        initial_lr=initial_lr,
                        min_lr=min_lr, patience=patience)
        
        elif args.mode == "eval":
            # Evaluate on validation set
            def remove_module_prefix(state_dict):
                return {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_cam1 = resnet18(input_channels=9).to(device)
            model_cam2 = resnet18(input_channels=9).to(device)
            model_cam1.load_state_dict(remove_module_prefix(torch.load(f'runs/{args.model_checkpoint_cam1}', map_location=device)))
            model_cam2.load_state_dict(remove_module_prefix(torch.load(f'runs/{args.model_checkpoint_cam2}', map_location=device)))
            val_mse = evaluate_model(model_cam1, model_cam2, val_loader, model_name)
            print(f"Validation MSE for {model_name}: {val_mse:.4f}")

        wandb.finish()
