import os
import torch
import numpy as np
import cv2
import random
import pyrealsense2 as rs
from pathlib import Path
import argparse
from datetime import datetime
import colorsys
import time
import sys

# Add project root to path to ensure imports work correctly. Get the absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from sam2.build_sam import build_sam2_camera_predictor
from simulation.demo.common_functions import (process_consecutive_frames, extract_points_from_mask, extract_transform_features)

parser = argparse.ArgumentParser()

# Input source options
parser.add_argument("--input_type", type=str, default="video", choices=["video", "realsense"],
                    help="Input source: 'video' for MP4 files or 'realsense' for RealSense camera")
parser.add_argument("-v", "--video_path", type=str, default=None,
                    help="Path to video file (required if input_type is 'video')")

# RealSense settings
parser.add_argument("--rs_width", type=int, default=640, help="RealSense width resolution")
parser.add_argument("--rs_height", type=int, default=480, help="RealSense height resolution")
parser.add_argument("--rs_fps", type=int, default=15, help="RealSense framerate")
parser.add_argument("--frame_timeout", type=int, default=5000, help="Timeout (ms) for frame acquisition")

# Camera selection parameters
parser.add_argument("--list_cameras", action="store_true", help="List all connected RealSense cameras and exit")
parser.add_argument("--camera_serial", type=str, default=None, 
                    help="Serial number of the RealSense camera to use")
parser.add_argument("--camera_index", type=int, default=0, 
                    help="Index of the RealSense camera to use (0, 1, 2, etc.)")

# Output settings
parser.add_argument("--out_dir", type=str, default="../videos/")
parser.add_argument("--model", "--model_checkpoint_path", type=str, default="../checkpoints/sam2.1_hiera_tiny.pt")
parser.add_argument("--cfg", "--model_config_path", type=str, default="configs/sam2.1/sam2.1_hiera_t_512")

args = parser.parse_args()

# Check for list_cameras first
if args.list_cameras:
    # If we're just listing cameras, skip other validation
    pass
# Otherwise validate arguments
elif args.input_type == "video" and args.video_path is None:
    parser.error("--video_path is required when input_type is 'video'")

# CUDA setup
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

# Use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # Turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def list_realsense_cameras():
    """List all connected RealSense cameras and their details."""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if devices.size() == 0:
        print("No RealSense cameras detected")
        return []
    
    camera_list = []
    print(f"\nFound {devices.size()} RealSense camera(s):")
    for i in range(devices.size()):
        device = devices[i]
        camera_info = {
            "index": i,
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version)
        }
        camera_list.append(camera_info)
        
        print(f"  Camera {i}:")
        print(f"    Name: {camera_info['name']}")
        print(f"    Serial: {camera_info['serial']}")
        print(f"    Firmware: {camera_info['firmware']}")
    
    return camera_list

def generate_fluorescent_color(num=10):
    """Generates random bright fluorescent colors as RGB tuples."""
    colors = []
    for _ in range(num):
        h, s, l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
        r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h, l, s)]
        colors.append((b, g, r))
    return colors

def setup_realsense(camera_serial=None, camera_index=0):
    """Set up RealSense camera pipeline with specific camera selection.
    
    Args:
        camera_serial: Serial number of the camera to use
        camera_index: Index of the camera to use if serial is not provided
    
    Returns:
        pipeline, align objects
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Find requested camera
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if devices.size() == 0:
        raise RuntimeError("No RealSense cameras detected")
    
    # Select camera by serial number or index
    selected_device = None
    device_serial = None
    
    if camera_serial is not None:
        for i in range(devices.size()):
            device = devices[i]
            if device.get_info(rs.camera_info.serial_number) == camera_serial:
                selected_device = device
                device_serial = camera_serial
                print(f"Selected camera by serial: {camera_serial}")
                config.enable_device(camera_serial)
                break
        if selected_device is None:
            print(f"Warning: Camera with serial {camera_serial} not found")
            # Fall back to index selection
    
    if selected_device is None:
        # Use index selection
        if camera_index >= devices.size():
            print(f"Warning: Camera index {camera_index} out of range. Using camera 0")
            camera_index = 0
        
        selected_device = devices[camera_index]
        device_serial = selected_device.get_info(rs.camera_info.serial_number)
        print(f"Selected camera by index: {camera_index}, Serial: {device_serial}")
        config.enable_device(device_serial)
    
    # Print camera advanced information
    print(f"Camera info:")
    for info_key in [rs.camera_info.name, rs.camera_info.serial_number, 
                     rs.camera_info.firmware_version, rs.camera_info.physical_port,
                     rs.camera_info.product_id, rs.camera_info.product_line]:
        try:
            value = selected_device.get_info(info_key)
            print(f"  {info_key}: {value}")
        except:
            pass
    
    # Enable streams with lower resolution and frame rate to reduce bandwidth requirements
    try:
        # Enable streams
        config.enable_stream(rs.stream.depth, args.rs_width, args.rs_height, rs.format.z16, args.rs_fps)
        config.enable_stream(rs.stream.color, args.rs_width, args.rs_height, rs.format.bgr8, args.rs_fps)
        
        # Advanced options to help with multiple cameras
        advanced_mode = rs.rs400_advanced_mode(selected_device)
        if advanced_mode.is_enabled():
            print("Advanced mode is enabled. Optimizing for multiple cameras.")
            # Could add advanced settings here if needed
    
        # Start streaming with timeout handling
        try:
            print(f"Starting pipeline for camera {device_serial}...")
            profile = pipeline.start(config)
            print("Pipeline started successfully")
            
            # Get the device sensor and check streaming status
            sensor = profile.get_device().first_depth_sensor()
            print(f"Depth sensor found: {sensor.get_info(rs.camera_info.name)}")
            
        except Exception as e:
            print(f"Error starting camera pipeline: {e}")
            raise
            
    except Exception as e:
        print(f"Error configuring camera streams: {e}")
        raise
    
    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # Wait a moment for the camera to stabilize
    print("Waiting for camera to stabilize...")
    time.sleep(2)
    print("Camera setup complete")
    
    return pipeline, align

def get_frame_from_realsense(pipeline, align):
    """Get a frame from the RealSense camera."""
    try:
        # Increase timeout to 15000ms (15 seconds) if needed
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            print("Warning: Invalid frames received from camera")
            return None, None
        
        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image
    except RuntimeError as e:
        print(f"Camera frame acquisition error: {e}")
        # Return None to indicate failure
        return None, None

def main():
    # List cameras if requested
    if args.list_cameras:
        list_realsense_cameras()
        return
    
    # Load SAM2 model
    sam2_checkpoint = args.model
    model_cfg = args.cfg
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
    print(f"Predictor device: {predictor.device}")
    
    # Initialize input source
    if args.input_type == "video":
        # Video file input
        video_path = args.video_path
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rs_pipeline, rs_align = None, None
    else:
        # RealSense camera input
        rs_pipeline, rs_align = setup_realsense(args.camera_serial, args.camera_index)
        frame_width = args.rs_width
        frame_height = args.rs_height
        cap = None
    
    # Setup output directory and video writer
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    
    if args.input_type == "video":
        output_path = f"{args.out_dir}/output_{Path(args.video_path).name}"
        if output_path[-3:] != 'mp4':
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d%H%M%S")
            output_path = output_path + f"_{timestamp}_.mp4"
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        camera_id = args.camera_serial if args.camera_serial else f"cam{args.camera_index}"
        output_path = f"{args.out_dir}/realsense_{camera_id}_{timestamp}.mp4"
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))
    
    # Tracking variables
    colors = generate_fluorescent_color(10)
    if_init = False
    fcount = 0
    top_camera_object_contours = []
    top_camera_panel_contours = []
    window = 30
    
    while True:
        # Get frame from appropriate source
        if args.input_type == "video":
            ret, frame = cap.read()
            if not ret:
                break
            depth_frame = None  # No depth frame for video files
        else:
            frame, depth_frame = get_frame_from_realsense(rs_pipeline, rs_align)
            if frame is None:
                print("Failed to get frame. Retrying...")
                time.sleep(0.5)  # Short pause before retrying
                continue
            ret = True
        
        fcount += 1
        
        # Convert to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if not if_init:
            # First frame - initialize tracking
            predictor.load_first_frame(frame_rgb)
            if_init = True
            
            ann_frame_idx = 0  # Frame index for interaction
            ann_obj_id = [1]  # Object IDs
            points, labels = {1: []}, {1: []}
            print("Object ID:", ann_obj_id)
            
            # Mouse callback for point selection
            def select_points(event, x, y, flags, params):
                obj_id = ann_obj_id[-1]
                if event == cv2.EVENT_LBUTTONDOWN:  # Positive point
                    points[obj_id].append((x, y))
                    labels[obj_id].append(1)
                    cv2.circle(frame, (x, y), 5, colors[obj_id], -1)
                    cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    print("Object ID:", obj_id, "\tPositive point:", x, y)
                elif event == cv2.EVENT_MBUTTONDOWN:  # Negative point
                    points[obj_id].append((x, y))
                    labels[obj_id].append(0)
                    cv2.circle(frame, (x, y), 5, colors[obj_id], 2)
                    cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    print("Object ID:", obj_id, "\tNegative point:", x, y)
            
            # Setup point selection UI
            cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.setMouseCallback("Select Key Points", select_points, ann_obj_id)
            
            # Wait for user input
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                # Space key - change object ID
                if key == 32 and len(points[ann_obj_id[-1]]) > 0:
                    obj_id = ann_obj_id[-1]
                    ann_obj_id.append(obj_id+1)
                    points[ann_obj_id[-1]], labels[ann_obj_id[-1]] = [], []
                    print("Object ID changed to:", ann_obj_id[-1])
                
                # Enter key - exit selection mode
                elif key == 13:
                    break
            
            cv2.destroyAllWindows()
            
            assert len(ann_obj_id) <= 3, "Object limit is set to 3 as the visualization code supports only 3 colors."
            
            # Add prompts for each selected object
            for i in ann_obj_id:
                if len(points[i]) > 0:
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, obj_id=i, points=points[i], labels=labels[i]
                    )
        
        else:
            # Tracking mode
            out_obj_ids, out_mask_logits = predictor.track(frame_rgb)
            print("out_obj_ids", out_obj_ids)
            
            # Process masks for visualization and feature extraction
            if len(out_obj_ids) >= 2:  # Make sure we have at least two objects
                top_camera_object_out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                top_camera_panel_out_mask = (out_mask_logits[1] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                
                top_camera_object_contours.append(top_camera_object_out_mask)
                top_camera_panel_contours.append(top_camera_panel_out_mask)
                
                # Calculate motion features if we have enough frames
                if fcount > window:
                    top_camera_object_pre_points = extract_points_from_mask(top_camera_object_contours[-window])
                    top_camera_object_current_points = extract_points_from_mask(top_camera_object_out_mask)
                    top_camera_panel_pre_points = extract_points_from_mask(top_camera_panel_contours[-window])
                    top_camera_panel_current_points = extract_points_from_mask(top_camera_panel_out_mask)
                    
                    if (len(top_camera_object_pre_points) > 0 and 
                        len(top_camera_object_current_points) > 0 and 
                        len(top_camera_panel_pre_points) > 0 and 
                        len(top_camera_panel_current_points) > 0):
                        
                        top_camera_object_pre_contours = [top_camera_object_pre_points.reshape(-1, 1, 2).astype(np.int32)]
                        top_camera_object_current_contours = [top_camera_object_current_points.reshape(-1, 1, 2).astype(np.int32)]
                        top_camera_panel_pre_contours = [top_camera_panel_pre_points.reshape(-1, 1, 2).astype(np.int32)]
                        top_camera_panel_current_contours = [top_camera_panel_current_points.reshape(-1, 1, 2).astype(np.int32)]
                        
                        # Extract transform features
                        top_camera_object_transforms = process_consecutive_frames(
                            top_camera_object_pre_contours, top_camera_object_current_contours)
                        top_camera_object_features = extract_transform_features(top_camera_object_transforms)
                        
                        top_camera_panel_transforms = process_consecutive_frames(
                            top_camera_panel_pre_contours, top_camera_panel_current_contours)
                        top_camera_panel_features = extract_transform_features(top_camera_panel_transforms)
                        
                        print("object_features", top_camera_object_features)
                        print("panel_features", top_camera_panel_features)
                    else:
                        print("Not enough points to calculate transforms")
                
                # Visualize masks on frame
                top_camera_object_out_mask = cv2.cvtColor(top_camera_object_out_mask, cv2.COLOR_GRAY2RGB)
                top_camera_object_out_mask[:, :, 0] = np.clip(top_camera_object_out_mask[:, :, 0] * 255, 0, 255).astype(np.uint8)
                frame_rgb = cv2.addWeighted(frame_rgb, 1, top_camera_object_out_mask, 0.5, 0)
                
                top_camera_panel_out_mask = cv2.cvtColor(top_camera_panel_out_mask, cv2.COLOR_GRAY2RGB)
                top_camera_panel_out_mask[:, :, 1] = np.clip(top_camera_panel_out_mask[:, :, 1] * 255, 0, 255).astype(np.uint8)
                frame_rgb = cv2.addWeighted(frame_rgb, 1, top_camera_panel_out_mask, 0.5, 0)
        
        # Convert back to BGR for display and saving
        frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Display frame
        cv2.imshow("frame", frame_display)
        
        # Write frame to output video
        out.write(frame_display)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Cleanup
    print(f"Video saved at {output_path}")
    if args.input_type == "video" and cap is not None:
        cap.release()
    elif rs_pipeline is not None:
        rs_pipeline.stop()
    
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()