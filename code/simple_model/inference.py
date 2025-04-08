import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# Import the ResNet models
from resnet_models import resnet18, resnet34, resnet50, resnet101, resnet152

def load_model(model_path, model_type='ResNet18', input_channels=19, device=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model weights
        model_type (str): Type of ResNet model to use
        input_channels (int): Number of input channels
        device (torch.device): Device to load the model on
    
    Returns:
        model: Loaded PyTorch model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Initialize the model based on type
    if model_type == 'ResNet18':
        model = resnet18(input_channels=input_channels)
    elif model_type == 'ResNet34':
        model = resnet34(input_channels=input_channels)
    elif model_type == 'ResNet50':
        model = resnet50(input_channels=input_channels)
    elif model_type == 'ResNet101':
        model = resnet101(input_channels=input_channels)
    elif model_type == 'ResNet152':
        model = resnet152(input_channels=input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_trajectory(trajectory_data, window_size=1):
    """
    Preprocess a trajectory for inference.
    
    Args:
        trajectory_data (list or np.ndarray): List of trajectory frames
        window_size (int): Size of the window to use
    
    Returns:
        torch.Tensor: Preprocessed data ready for model input
    """
    if window_size > len(trajectory_data):
        raise ValueError(f"Window size ({window_size}) is larger than trajectory length ({len(trajectory_data)})")
    
    # Extract the relevant window
    window_data = trajectory_data[-window_size:]
    
    # Get states sequence (shape: window_size x 19)
    states = np.stack([frame['state'] for frame in window_data])

    # Take only the first 8 state values
    states = states[:, :8]
    
    # Convert to tensor and transpose for 1D convolution (to 19 x window_size)
    states_tensor = torch.FloatTensor(states).transpose(0, 1)
    
    # Add batch dimension
    states_tensor = states_tensor.unsqueeze(0)
    
    return states_tensor

def predict_risk(model, trajectory_data, window_size=1, device=None):
    """
    Predict risk from a trajectory.
    
    Args:
        model: Trained PyTorch model
        trajectory_data (list or np.ndarray): List of trajectory frames
        window_size (int): Size of the window to use
        device (torch.device): Device to run inference on
    
    Returns:
        float: Predicted risk value
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess the trajectory
    inputs = preprocess_trajectory(trajectory_data, window_size)
    inputs = inputs.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(inputs)
    
    # Return the predicted risk
    return outputs.cpu().numpy().flatten()[0]

def predict_from_states(model, states, device=None, channel=19):
    """
    Predict risk directly from state data (single window).
    
    Args:
        model: Trained PyTorch model
        states (np.ndarray or torch.Tensor): State data with shape (channel, window_size) or (window_size, channel)
        device (torch.device): Device to run inference on
        
    Returns:
        float: Predicted risk value
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensor if numpy array
    if isinstance(states, np.ndarray):
        # Check if we need to transpose
        if states.shape[0] != channel and states.shape[1] == channel:
            states = states.T  # Transpose to (channel, window_size)
        states_tensor = torch.FloatTensor(states)
    else:
        states_tensor = states
        
    # Make sure shape is correct (channel, window_size)
    if states_tensor.shape[0] != channel:
        raise ValueError(f"Expected first dimension to be channel, got {states_tensor.shape[0]}")
        
    # Add batch dimension if not present
    if len(states_tensor.shape) == 2:
        states_tensor = states_tensor.unsqueeze(0)
        
    # Move to device
    states_tensor = states_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(states_tensor)
    
    # Return the predicted risk
    return outputs.cpu().numpy().flatten()[0]

def batch_inference(model, data_dir, window_size=1, device=None):
    """
    Run inference on multiple trajectories in a directory.
    
    Args:
        model: Trained PyTorch model
        data_dir (str): Directory containing trajectory files
        window_size (int): Size of the window to use
        device (torch.device): Device to run inference on
    
    Returns:
        dict: Dictionary mapping file names to predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get list of all .npy files in directory
    episode_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    
    results = {}
    ground_truth = {}
    
    # Create progress bar
    pbar = tqdm(episode_files, desc="Running inference")
    
    for episode_file in pbar:
        # Load trajectory data
        trajectory_data = np.load(os.path.join(data_dir, episode_file), allow_pickle=True)
        
        # Predict risk
        risk = predict_risk(model, trajectory_data, window_size, device)
        
        # Store the result
        results[episode_file] = risk
        
        # Store ground truth if available
        if 'risk' in trajectory_data[-1]:
            ground_truth[episode_file] = trajectory_data[-1]['risk']
        
        pbar.set_postfix({'file': episode_file})
    
    return results, ground_truth

def visualize_results(predictions, ground_truth=None, save_path=None):
    """
    Visualize prediction results.
    
    Args:
        predictions (dict): Dictionary of predictions
        ground_truth (dict, optional): Dictionary of ground truth values
        save_path (str, optional): Path to save the visualization
    """
    # Sort keys to ensure consistent ordering
    sorted_keys = sorted(predictions.keys())
    pred_values = [predictions[k] for k in sorted_keys]
    
    plt.figure(figsize=(12, 6))
    
    # Plot predictions
    plt.bar(range(len(sorted_keys)), pred_values, alpha=0.6, label='Predictions')
    
    # Plot ground truth if available
    if ground_truth is not None:
        truth_values = [ground_truth.get(k, float('nan')) for k in sorted_keys]
        plt.scatter(range(len(sorted_keys)), truth_values, color='red', label='Ground Truth')
    
    plt.xlabel('Trajectory Index')
    plt.ylabel('Risk Value')
    plt.title('Risk Predictions')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def calculate_metrics(predictions, ground_truth):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (dict): Dictionary of predictions
        ground_truth (dict): Dictionary of ground truth values
    
    Returns:
        dict: Dictionary of metrics
    """
    # Find common keys
    common_keys = set(predictions.keys()).intersection(set(ground_truth.keys()))
    
    if not common_keys:
        return {"error": "No common keys found between predictions and ground truth"}
    
    # Extract values
    preds = np.array([predictions[k] for k in common_keys])
    truths = np.array([ground_truth[k] for k in common_keys])
    
    # Calculate MSE
    mse = ((preds - truths) ** 2).mean()
    
    # Calculate MAE
    mae = np.abs(preds - truths).mean()
    
    # Calculate R²
    if len(common_keys) > 1:
        # Calculate correlation coefficient
        corr = np.corrcoef(preds, truths)[0, 1]
        r_squared = corr ** 2
    else:
        r_squared = float('nan')
    
    return {
        "mse": mse,
        "mae": mae,
        "r_squared": r_squared,
        "num_samples": len(common_keys)
    }

def single_window_inference_example(model_path, model_type='ResNet18', input_channels=19):
    """
    Example function showing how to perform inference on a single window of data.
    
    Args:
        model_path (str): Path to the saved model weights
        model_type (str): Type of ResNet model
        input_channels (int): Number of input channels for the model
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = load_model(model_path, model_type, input_channels, device)
    
    # Create synthetic data for demonstration (19 features, window_size=1)
    # In a real scenario, this would come from your robot's sensors or a data source
    window_size = 1
    synthetic_states = np.random.rand(input_channels, window_size)
    
    # Run inference
    print("Running inference on single window...")
    risk = predict_from_states(model, synthetic_states, device, input_channels)
    
    print(f"Predicted risk: {risk}")
    
    # For comparison, you could also use it with trajectory-like data structure
    # Create a synthetic trajectory with the same structure as your training data
    synthetic_trajectory = [{'state': np.random.rand(input_channels)} for _ in range(window_size)]
    
    # Run inference using the trajectory method
    risk_from_trajectory = predict_risk(model, synthetic_trajectory, window_size, device)
    
    print(f"Predicted risk from trajectory: {risk_from_trajectory}")
    
    return risk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Trajectory Risk Inference")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--model_type', type=str, default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'], help='Type of ResNet model')
    parser.add_argument('--data_dir', type=str, help='Directory containing trajectory files for batch inference')
    parser.add_argument('--window_size', type=int, default=1, help='Size of the trajectory window')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    parser.add_argument('--single_window', action='store_true', help='Run single window inference example instead of batch')
    parser.add_argument('--input_file', type=str, help='Path to a single trajectory file for inference')
    parser.add_argument('--input_channels', type=int, default=19, help='Number of input channels for the model')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.model_type, input_channels=args.input_channels, device=device)
    
    if args.single_window:
        # Run the single window inference example
        single_window_inference_example(args.model_path, args.model_type, input_channels=args.input_channels)
    elif args.input_file:
        # Run inference on a single file
        print(f"Running inference on single file: {args.input_file}")
        trajectory_data = np.load(args.input_file, allow_pickle=True)
        risk = predict_risk(model, trajectory_data, args.window_size, device)
        print(f"Predicted risk: {risk}")
        
        # Get ground truth if available
        if 'risk' in trajectory_data[-1]:
            ground_truth = trajectory_data[-1]['risk']
            print(f"Ground truth risk: {ground_truth}")
            print(f"Error: {abs(risk - ground_truth)}")
    elif args.data_dir:
        # Run batch inference
        print(f"Running inference on trajectories in {args.data_dir}")
        predictions, ground_truth = batch_inference(model, args.data_dir, args.window_size, device)
        
        if args.output_dir:
            # Save predictions
            predictions_file = os.path.join(args.output_dir, 'predictions.npy')
            np.save(predictions_file, predictions)
            print(f"Saved predictions to {predictions_file}")
        
        # Calculate metrics if ground truth is available
        if ground_truth:
            metrics = calculate_metrics(predictions, ground_truth)
            print("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            
            if args.output_dir:
                # Save metrics
                metrics_file = os.path.join(args.output_dir, 'metrics.txt')
                with open(metrics_file, 'w') as f:
                    for metric, value in metrics.items():
                        f.write(f"{metric}: {value}\n")
                print(f"Saved metrics to {metrics_file}")
        
        # Visualize results if requested
        if args.visualize and args.output_dir:
            viz_file = os.path.join(args.output_dir, 'visualization.png')
            visualize_results(predictions, ground_truth, viz_file)
            print(f"Saved visualization to {viz_file}")
    else:
        print("Error: Either --data_dir, --input_file, or --single_window must be specified.")
        parser.print_help()
    
    print("Inference completed successfully!")