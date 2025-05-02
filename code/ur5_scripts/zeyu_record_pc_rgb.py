import pyrealsense2 as rs
import numpy as np
import cv2
import os

import argparse

# Function to capture an image from the depth camera
def capture_video_from_depth_camera(save_folder, fps=30, save_all=True):
    """
    Capture video from the depth camera and save it to the specified folder.
    Press 'Q' to stop recording and quit.
    """
    
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams with aligned resolution
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)

    # Start streaming
    profile = pipeline.start(config)

    # Get the depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    try:
        # Initialize video writers and paths
        color_video_path = os.path.join(save_folder, 'color_video.avi')
        depth_video_path = os.path.join(save_folder, 'depth_video.avi')
        combined_video_path = os.path.join(save_folder, 'combined_video.avi')
        raw_depth_path = os.path.join(save_folder, 'raw_depth_frames.npy')
        
        # Initialize raw depth data list with pre-allocated memory
        raw_depth_frames = []
        
        if save_all:
            color_writer = cv2.VideoWriter(color_video_path, 
                                         cv2.VideoWriter_fourcc(*'XVID'), 
                                         fps, (1280, 720))
            depth_writer = cv2.VideoWriter(depth_video_path, 
                                         cv2.VideoWriter_fourcc(*'XVID'), 
                                         fps, (1280, 720))
            combined_writer = cv2.VideoWriter(combined_video_path, 
                                            cv2.VideoWriter_fourcc(*'XVID'), 
                                            fps, (2560, 720))
        else:
            combined_writer = cv2.VideoWriter(combined_video_path, 
                                            cv2.VideoWriter_fourcc(*'XVID'), 
                                            fps, (2560, 720))

        # Get depth camera intrinsics
        depth_frame = pipeline.wait_for_frames().get_depth_frame()
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        frame_count = 0
        print("Recording started. Press 'Q' to stop and save.")
        
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Store raw depth frame (make a copy to ensure data persistence)
            if save_all:
                raw_depth_frames.append(depth_image.copy())
            
            # Create depth colormap with adjusted scale
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Create combined image
            combined_image = np.hstack((color_image, depth_colormap))
            
            # Write frames
            if save_all:
                color_writer.write(color_image)
                depth_writer.write(depth_colormap)
                combined_writer.write(combined_image)
            else:
                combined_writer.write(combined_image)
            
            # Display preview (resized for better viewing)
            display_image = cv2.resize(combined_image, (1920, 540))
            cv2.imshow('Recording Preview (Press Q to stop)', display_image)
            
            # Check for 'Q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            print(f"Recording: {frame_count} frames", end='\r')
        
        print(f"\nRecording complete! Saved {frame_count} frames.")
        
        # Save raw depth frames as numpy array
        if save_all and raw_depth_frames:
            print("Saving raw depth data...")
            raw_depth_array = np.array(raw_depth_frames)
            np.save(raw_depth_path, raw_depth_array)
            print(f"Raw depth data saved with shape: {raw_depth_array.shape}")
            
            # Close video writers
            color_writer.release()
            depth_writer.release()
        
        combined_writer.release()
        cv2.destroyAllWindows()
        
    finally:
        # Stop streaming
        pipeline.stop()
    
    if save_all:
        return (color_video_path, depth_video_path, combined_video_path, 
                raw_depth_path, depth_intrinsics)
    else:
        return combined_video_path
    
def view_real_time_from_depth_camera():


    print("reset start")
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()

    print("reset end")


    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)   # USB 2.0
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)    # USB 3.0

    # Start streaming
    pipeline.start(config)





    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)




            
    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture an image from the depth camera.')
    parser.add_argument('--save_all', action='store_true', help='Save all streams (color, depth, combined, raw)')
    parser.add_argument('--save_folder', type=str, default='./', help='Folder to save the image.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    args = parser.parse_args()

    if not os.path.exists(args.save_folder) and args.save_all:
        os.makedirs(args.save_folder)
    
    
    if args.save_all:
        capture_video_from_depth_camera(
        args.save_folder,
        fps=args.fps,
        save_all=args.save_all
    )
    else:
        view_real_time_from_depth_camera()