import torch
import argparse
import os
import onnx
from resnet_models import resnet18, resnet34, resnet50, resnet101, resnet152

def convert_to_onnx(model_path, model_type, input_channels, window_size, onnx_path, opset_version=13):
    """
    Convert a trained PyTorch model to ONNX format with more explicit name handling.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Create a dummy input tensor with the correct shape
    # Using a fixed window_size for more stability
    dummy_input = torch.randn(1, input_channels, window_size, device=device)
    
    # Export the model to ONNX with fixed input/output names
    print("Exporting model to ONNX...")
    
    # Use simpler settings first to ensure compatibility
    torch.onnx.export(
        model,                         # PyTorch model
        dummy_input,                   # Input tensor
        onnx_path,                     # Output file path
        export_params=True,            # Store the trained parameter weights
        opset_version=opset_version,   # ONNX version
        do_constant_folding=True,      # Fold constants
        input_names=['input'],         # Fixed input name
        output_names=['output'],       # Fixed output name
        # No dynamic axes for better compatibility
    )
    
    print(f"Model exported to ONNX format at: {onnx_path}")
    
    # Verify and fix the model if needed
    try:
        # Load the model
        print("Verifying ONNX model...")
        onnx_model = onnx.load(onnx_path)
        
        # Check the model
        onnx.checker.check_model(onnx_model)
        
        # Extra check: verify that input/output names are as expected
        print("Checking model inputs/outputs...")
        inputs = [input.name for input in onnx_model.graph.input]
        outputs = [output.name for output in onnx_model.graph.output]
        
        print(f"Model inputs: {inputs}")
        print(f"Model outputs: {outputs}")
        
        # Initialize the modified flag
        modified = False
        
        # If needed, we could modify the names here
        if inputs and inputs[0] != 'input':
            print(f"Fixing input name from '{inputs[0]}' to 'input'")
            onnx_model.graph.input[0].name = 'input'
            modified = True
        
        if outputs and outputs[0] != 'output':
            print(f"Fixing output name from '{outputs[0]}' to 'output'")
            onnx_model.graph.output[0].name = 'output'
            modified = True
        
        # Save the modified model if needed
        if modified:
            print("Saving modified model...")
            onnx.save(onnx_model, onnx_path)
        
        print("ONNX model verification passed!")
        
    except ImportError:
        print("ONNX package not found. Skipping model verification.")
    except Exception as e:
        print(f"ONNX model verification failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved PyTorch model weights')
    parser.add_argument('--model_type', type=str, default='ResNet18', 
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'], 
                        help='Type of ResNet model')
    parser.add_argument('--input_channels', type=int, default=19, help='Number of input channels')
    parser.add_argument('--window_size', type=int, default=1, help='Size of the input window')
    parser.add_argument('--onnx_path', type=str, default='model.onnx', help='Output path for the ONNX model')
    parser.add_argument('--opset_version', type=int, default=13, help='ONNX opset version')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.onnx_path)), exist_ok=True)
    
    # Convert the model
    convert_to_onnx(
        args.model_path, 
        args.model_type, 
        args.input_channels, 
        args.window_size,
        args.onnx_path,
        args.opset_version
    )