import cv2
import numpy as np

fx = 892.2883911132812
fy = 892.2883911132812
cx = 643.8308715820312
cy = 376.3149414062


# Known 2D - 3D coordinate pairs
# video used "../videos/zeyu_record_0218/record_blue_cube/color_video.avi"

obj_img_pairs = [
                    [[916, 322],[587, 320, 40]], #frame 400 -> blue cube -> TL corner
                    [[958, 345],[638, 341, 40]], #frame 400 -> blue cube -> BR corner

                    [[319, 306],[0, 320, 463]], #frame 450 -> tray -> Left-mid-edge in lifted position
                    [[595, 333],[251, 349, 463]], #frame 450 -> tray -> Right-mid-edge in lifted position
                    [[378, 326],[0, 320, 43]],  #frame 450 -> tray -> Left-mid-edge in low position
                    [[602, 348],[251, 349, 43]], #frame 450 -> tray -> Right-mid-edge in low position
                ]

# Known 3D points in the world (e.g., corners of a chessboard or object in real-world coordinates)
object_points = np.array([pair[1] for pair in obj_img_pairs], dtype=np.float32)

# Corresponding 2D pixel coordinates in the video frame
image_points = np.array([pair[0] for pair in obj_img_pairs], dtype=np.float32)

# Camera intrinsic matrix (from calibration, or approximate for your camera)
K = np.array([
    [fx,  0, cx],  # Focal length & principal point
    [0,  fy, cy],  
    [0,   0,  1]
], dtype=np.float32)

# Distortion coefficients (if known, else use np.zeros(5))
dist_coeffs = np.zeros(5)

# Solve for rotation (R) and translation (t)
success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs)

# Convert rotation vector to rotation matrix
R, _ = cv2.Rodrigues(rvec)

# Construct the 4x4 extrinsic matrix [R|t]
extrinsic_matrix = np.eye(4)
extrinsic_matrix[:3, :3] = R
extrinsic_matrix[:3, 3] = tvec.flatten()

print("Extrinsic Matrix:\n", extrinsic_matrix)

np.savetxt("../videos/camera_extrinsic_pose.txt", extrinsic_matrix)