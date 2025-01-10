import pyrealsense2 as rs
import cv2
import numpy as np

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the RGB stream
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Create a VideoWriter object for saving to MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 720))

try:
    print("Recording... Press Ctrl+C to stop.")
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Write frame to file
        out.write(color_image)

        # Display the video stream
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Stopped recording.")

finally:
    # Release resources
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
