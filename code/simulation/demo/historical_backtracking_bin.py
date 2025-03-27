import os
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
import sys

from mujoco_base import MuJoCoBase
import imageio
import random

import tqdm
# from scipy.interpolate import CubicSpline
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
from scipy.spatial.distance import cdist


N_TRAIN_EPISODES = 20
N_VAL_EPISODES = 25
EPISODE_LENGTH = 350  # Number of points in trajectory

RANDOM_EPISODE_TMP = random.randint(0, EPISODE_LENGTH) # 67 86 #  109 282 731 344

# Define different interpolation methods
def linear_interpolation(first_failure_time_step, failure_time_step_trim):
    return np.linspace(0, 1, first_failure_time_step - failure_time_step_trim + 1)

def sin_interpolation(first_failure_time_step, failure_time_step_trim):
    x = np.linspace(0, 1, first_failure_time_step - failure_time_step_trim + 1)
    return np.sin(x)

def find_closest_points(src_points, dst_points):
    """
    Find closest point pairs between source and destination point sets.
    
    Args:
        src_points: (N, 2) array of source points
        dst_points: (M, 2) array of destination points
        
    Returns:
        Tuple of matched points arrays, both of shape (K, 2)
    """
    # Calculate pairwise distances between all points
    distances = cdist(src_points, dst_points)
    
    # Find closest destination point for each source point
    closest_indices = np.argmin(distances, axis=1)
    
    # Create matched pairs
    src_matched = src_points
    dst_matched = dst_points[closest_indices]
    
    return src_matched, dst_matched

def calculate_transformation(src_points, dst_points):
    """
    Calculate rigid transformation (R, t) between matched point sets.
    
    Args:
        src_points: (N, 2) array of source points
        dst_points: (N, 2) array of destination points
        
    Returns:
        R: 2x2 rotation matrix
        t: 2D translation vector
    """
    # Calculate centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)
    
    # Center the point sets
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    # Calculate covariance matrix
    H = src_centered.T @ dst_centered
    
    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Calculate rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Calculate translation
    t = dst_centroid - (R @ src_centroid)
    
    return R, t

def create_homogeneous_matrix(R, t):
    """
    Create a 3x3 homogeneous transformation matrix from rotation R and translation t.
    
    Args:
        R: 2x2 rotation matrix
        t: 2D translation vector
        
    Returns:
        3x3 homogeneous transformation matrix
    """
    H = np.eye(3)
    H[:2, :2] = R
    H[:2, 2] = t
    return H

def icp_2d(src_contour, dst_contour, max_iterations=16, tolerance=1e-6):
    """
    Perform 2D ICP algorithm between two contours.
    
    Args:
        src_contour: Source contour points from OpenCV findContours
        dst_contour: Destination contour points from OpenCV findContours
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for mean squared error
        
    Returns:
        H: 3x3 homogeneous transformation matrix
        error: Final mean squared error
    """
    # Convert contours to point arrays
    src_points = src_contour.reshape(-1, 2).astype(np.float32)
    dst_points = dst_contour.reshape(-1, 2).astype(np.float32)
    
    # Initialize transformation
    R_total = np.eye(2)
    t_total = np.zeros(2)
    
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # Find closest point pairs
        src_matched, dst_matched = find_closest_points(src_points, dst_points)
        
        # Calculate transformation
        R, t = calculate_transformation(src_matched, dst_matched)
        
        # Update total transformation
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Apply transformation to source points
        src_points = (R @ src_points.T).T + t
        
        # Calculate error
        current_error = np.mean(np.sum((src_matched - dst_matched) ** 2, axis=1))
        
        # Check convergence
        if abs(prev_error - current_error) < tolerance:
            break
        
        prev_error = current_error
    
    # Apply log1p scaling to R_total and t_total
    scale_factor = 1e5
    R_total_scaled = np.sign(R_total) * np.log1p(np.abs(R_total) * scale_factor)
    t_total_scaled = np.sign(t_total) * np.log1p(np.abs(t_total) * scale_factor)
    
    # print("Original R_total:", R_total)
    # print("Scaled R_total:", R_total_scaled)
    # print("Original t_total:", t_total)
    # print("Scaled t_total:", t_total_scaled)
    
    # Create homogeneous transformation matrix with scaled values
    H = create_homogeneous_matrix(R_total_scaled, t_total_scaled)

    # print("H with scaled values:", H)
    
    return H, current_error

def process_consecutive_frames(contours1, contours2):
    """
    Process consecutive frames and calculate transformations for each contour pair.
    
    Args:
        contours1: List of contours from first frame
        contours2: List of contours from second frame
        
    Returns:
        List of (R, t, error) tuples for each matched contour pair
    """
    results = []
    
    # Match contours based on area similarity
    areas1 = [cv2.contourArea(cnt) for cnt in contours1]
    areas2 = [cv2.contourArea(cnt) for cnt in contours2]
    
    for i, cnt1 in enumerate(contours1):
        # Find best matching contour in second frame
        best_match = None
        best_area_diff = float('inf')
        
        for j, cnt2 in enumerate(contours2):
            area_diff = abs(areas1[i] - areas2[j])
            if area_diff < best_area_diff:
                best_area_diff = area_diff
                best_match = cnt2
        
        if best_match is not None:
            # Calculate ICP between matched contours
            H, error = icp_2d(cnt1, best_match)
            results.append((H, error))
    
    return results

def extract_transform_features(transforms):
    if not transforms:
        # Return zeros if no transforms
        return np.zeros(6)  # 4 for rotation matrix elements + 2 for translation
    
    # Take first transform if multiple are present
    H, error = transforms[0]
    
    # Extract rotation matrix and translation vector
    R = H[:2, :2]  # 2x2 rotation matrix
    t = H[:2, 2]   # 2D translation vector
    
    # Convert to numpy arrays if needed
    R = np.array(R)
    t = np.array(t)
    
    # Ensure correct shapes before concatenating
    R_flat = R.flatten()  # Make 2x2 matrix into 1D array of length 4
    t = t.reshape(-1)    # Ensure translation is 1D array
    
    # print("R_flat", R_flat)
    # print("t", t)
    # Combine into single feature vector
    features = np.concatenate([R_flat, t])
    features = [features[0], features[1], features[4], features[5]]
    # print("features", features)
    
    return features

def process_camera_frame(frame):
    """
    Process camera frame to get mask and contours for non-zero pixel values.
    
    Args:
        frame: RGB image array (height, width, 3)
    
    Returns:
        tuple: (mask, contours, filtered_image)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Create mask for non-zero pixels
    # Use a small threshold to filter out near-black pixels
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours (noise)
    min_contour_area = 100  # Adjust this threshold as needed
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Create visualization of the filtered image
    filtered_image = frame.copy()
    cv2.drawContours(filtered_image, contours, -1, (0, 255, 0), 2)
    
    # Draw bounding panel around detected objects
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(filtered_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    #     # Calculate and display centroid
    #     M = cv2.moments(contour)
    #     if M["m00"] != 0:
    #         cx = int(M["m10"] / M["m00"])
    #         cy = int(M["m01"] / M["m00"])
    #         cv2.circle(filtered_image, (cx, cy), 5, (0, 0, 255), -1)
    
    return mask, contours, filtered_image

class Projectile(MuJoCoBase):
    def __init__(self, xml_path, traj_file, initial_delay=3.0, display_camera=False, enable_cameras=True):
        super().__init__(xml_path)
        
        # Simulation parameters
        self.initial_delay = initial_delay  # Delay before starting movement
        self.display_camera = display_camera
        self.episode = [] # To store the episode data
        # self.high_threshold_step = 0  # Step when displacement > DISPLACEMENT_THRESHOLD_HIGH
        # self.is_high_threshold_step_set = False

        # Trajectory loading
        self.traj_file = traj_file
        self.trajectory = self.load_trajectory()

        # Simulation state
        self.current_step = 0
        self.current_target = None
        self.next_target = None
        self.transition_start_time = None
        self.start_time = None  # To track the start time

        # Camera settings
        self.ncam = self.model.ncam  # Get number of cameras in the model
        self.cam_position_init = [None] * self.ncam  # Initialize with None for each camera
        self.cam_position_read = [False] * self.ncam  # Initialize all as unread

        # New camera control flag
        self.enable_cameras = enable_cameras

        # self.speed_scale = random.uniform(2.5, 3.5)  # New parameter to control joint speed
        # self.joint_pause = random.uniform(0.2, 0.8)  # Duration of pause between movements

    def load_trajectory(self, randomize=True):
        """
        Load trajectory from a text file containing lists of joint positions.
        
        Args:
            randomize (bool): Whether to add random variations to the joint positions
            
        Returns:
            list: List of joint position arrays with optional randomization
        """
        with open(self.traj_file, "r") as file:
            trajectory = []
            for line in file:
                try:
                    joint_positions = np.array(ast.literal_eval(line.strip()))
                    
                    # Apply randomization if enabled
                    if randomize:
                        # Define randomization ranges for each joint (in radians)
                        # These values can be adjusted based on how much variation you want
                        variation_ranges = [
                            (-0.05, 0.05),  # Joint 1: ±0.05 radians
                            (-0.03, 0.03),  # Joint 2: ±0.03 radians
                            (-0.04, 0.04),  # Joint 3: ±0.04 radians
                            (-0.03, 0.03),  # Joint 4: ±0.03 radians
                            (-0.05, 0.05),  # Joint 5: ±0.05 radians
                            (-0.02, 0.02),  # Joint 6: ±0.02 radians
                        ]
                        
                        # Apply random variation to each joint
                        for i in range(min(len(joint_positions), len(variation_ranges))):
                            min_var, max_var = variation_ranges[i]
                            variation = random.uniform(min_var, max_var)
                            joint_positions[i] += variation
                    
                    trajectory.append(joint_positions)
                except Exception as e:
                    print(f"Error parsing line: {line.strip()}. Error: {e}")
                    continue
            
            print(f"Loaded {len(trajectory)} waypoints{' with randomization' if randomize else ''}")
            return trajectory

    def reset(self, seed):
        """Reset the simulation environment with the given seed."""
        random.seed(seed)

        # Set camera configuration
        self.cam.azimuth = 300 # -216 random.uniform(-225, -315)
        self.cam.distance = 2.5 # random.uniform(2, 3)
        self.cam.elevation = -40 # random.uniform(-16, -30)
        # print("self.cam", self.cam.azimuth, self.cam.distance, self.cam.elevation)

        # Randomize camera positions
        self.randomize_camera_position('top_camera')
        self.randomize_camera_position('front_camera')

        # Set random rotation parameters
        self.rotation_speed = random.uniform(-0.01, 0.01)  # Slower random speed between 0.2 and 0.5 rad/s
        self.target_angle = random.uniform(-0.03, 0.03)   # Small random angle between -0.3 and 0.3 rad (about ±17 degrees)
        self.reverse_on_target = random.choice([True, False])  # Randomly decide to stop or reverse
        self.reached_target = False  # Track if we've reached target
        # print(f"New rotation parameters - Speed: {self.rotation_speed:.2f} rad/s, Target angle: {self.target_angle:.2f} rad, Reverse: {self.reverse_on_target}")

        # Reset simulation state
        # self.high_threshold_step = 0  # Step when displacement > DISPLACEMENT_THRESHOLD_HIGH
        # self.is_high_threshold_step_set = False
        self.start_time = None
        self.current_step = 0
        self.current_target = None
        self.next_target = None
        self.transition_start_time = None
        self.speed_scale = random.uniform(2.5, 3.5)  # New parameter to control joint speed
        self.joint_pause = random.uniform(0.2, 0.8)  # Duration of pause between movements

        # Emergency stop settings
        self.emergency_stop = False  # Flag to trigger the emergency stop
        self.emergency_stop_pos_first_set = False  # Flag to first set the emergency stop pose
        self.current_qpos = None

        # Randomize environment
        self.randomize_floor()
        self.randomize_object_colors("object_collision")
        self.randomize_object('object_body', 'object_collision')

        mj.set_mjcb_control(self.controller)

    def randomize_floor(self):
        """Randomize the floor appearance by changing materials and colors."""
        # Get floor geom ID
        floor_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        
        # Randomly select floor material
        materials = ["groundplane1", "groundplane2", "groundplane3", "groundplane4", "groundplane5", "groundplane6", "groundplane7", "groundplane8", "groundplane9"]
        selected_material = random.choice(materials)
        material_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_MATERIAL, selected_material)
        
        # Apply selected material to floor
        self.model.geom_matid[floor_id] = material_id
        
        # Randomly adjust floor parameters
        # Random scale for texture repeat
        texture_scale = random.uniform(3, 7)
        self.model.mat_texrepeat[material_id] = [texture_scale, texture_scale]
        
        # Random reflectance
        self.model.mat_reflectance[material_id] = random.uniform(0.1, 0.3)
        
        # Optional: Add some random noise to the floor color
        rgb_noise = np.random.uniform(-0.1, 0.1, 3)
        self.model.mat_rgba[material_id][:3] += rgb_noise
        self.model.mat_rgba[material_id][:3] = np.clip(self.model.mat_rgba[material_id][:3], 0, 1)

    def randomize_object(self, object_body_id, object_geom_id):
        """Randomize the object's size, mass, friction, and other physical properties."""
        # Get object body and geom IDs
        object_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, object_body_id)
        object_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, object_geom_id)

        fixed_panel_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_panel')
        fixed_panel_pos = self.data.xpos[fixed_panel_body_id]
        # print("fixed_panel_pos", fixed_panel_pos)
        
        # Randomize object size (within reasonable bounds)
        base_size = 0.015  # Original size
        size_variation = random.uniform(1.0, 3)  # n% variation
        new_size = base_size * size_variation
        self.model.geom_size[object_geom_id] = [new_size, new_size, new_size]
        
        # Randomize mass (scaled with size)
        base_mass = 0.2  # Original mass
        mass_variation = random.uniform(0.8, 1.2)  # ±20% variation
        new_mass = base_mass * mass_variation  # base_mass * size_variation * mass_variation  # Scale mass with size
        self.model.body_mass[object_body_id] = new_mass
        
        # Adjust inertia based on new mass and size
        new_inertia = (new_mass * new_size**2) / 6  # Simple panel inertia approximation
        self.model.body_inertia[object_body_id] = [new_inertia, new_inertia, new_inertia]
        
        # Randomize friction properties
        friction_variation = random.uniform(0.25, 0.35)  # Base friction is 0.2
        # Find the contact pair involving the sliding object
        for i in range(self.model.npair):
            if (self.model.pair_geom1[i] == object_geom_id or 
                self.model.pair_geom2[i] == object_geom_id):
                self.model.pair_friction[i, 0] = friction_variation  # Sliding friction
                self.model.pair_friction[i, 1] = friction_variation * 2.5  # Rolling friction
                self.model.pair_friction[i, 2] = friction_variation * 0.005  # Torsional friction
        
        # Randomize initial position (within reasonable bounds)
        x_pos = random.uniform(fixed_panel_pos[0]-0.04, fixed_panel_pos[0]+0.04) # random.uniform(0.35, 0.42)
        y_pos = random.uniform(fixed_panel_pos[1]-0.04, fixed_panel_pos[1]+0.04) # random.uniform(-0.5, -0.35)
        z_pos = random.uniform(fixed_panel_pos[2]+0.05, fixed_panel_pos[2]+0.06)
        self.data.qpos[self.model.body_jntadr[object_body_id]:self.model.body_jntadr[object_body_id]+3] = [x_pos, y_pos, z_pos]
        
        # Randomize initial orientation (uncomment when needed)
        # quat = [random.uniform(-1, 1) for _ in range(4)]
        # quat = quat / np.linalg.norm(quat)  # Normalize quaternion
        # self.data.qpos[self.model.body_jntadr[object_body_id]+3:self.model.body_jntadr[object_body_id]+7] = quat

    def randomize_object_colors(self, object_geom_name):
        """Randomize colors for fixed panel and free object"""
        # Zeyu: need to be revised, since combined object and the panel, should be decoupled
        # Get geom IDs
        fixed_panel_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "panel")
        free_object_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, object_geom_name)
        
        # Generate new colors
        panel_color = [random.uniform(0.01, 1.0), random.uniform(0.01, 1.0), random.uniform(0.01, 1.0), 1.0]
        object_color = np.array([1.0, 1.0, 1.0, 2.0]) - panel_color  # [random.uniform(0.01, 1.0), random.uniform(0.01, 1.0), random.uniform(0.01, 1.0), 1.0]
        
        # Set new colors
        self.model.geom_rgba[fixed_panel_geom_id] = panel_color
        self.model.geom_rgba[free_object_geom_id] = object_color

    def randomize_camera_position(self, camera_name='top_camera'):
        """
        Improved camera position randomization with proper orientation handling
        and direct camera parameter updates.
        """
        # Get camera id
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)

        # print("self.cam_position_read[cam_id]", cam_id, self.cam_position_read[cam_id])

        if not self.cam_position_read[cam_id]:
            self.cam_position_init[cam_id] = self.model.cam_pos[cam_id].copy()
            # print(self.cam_position_init[cam_id])
            self.cam_position_read[cam_id] = True
        
        # print("self.cam_position_init[cam_id]", cam_id, self.cam_position_init[cam_id])
        
        # Define ranges for camera randomization
        # Wider ranges for more noticeable variation
        pos_ranges = {
            'x': (self.cam_position_init[cam_id][0]-0.1, self.cam_position_init[cam_id][0]+0.1),    # Wider range for x offset
            'y': (self.cam_position_init[cam_id][1]-0.1, self.cam_position_init[cam_id][1]+0.1),    # Wider range for y offset
            'z': (self.cam_position_init[cam_id][2]-0.1, self.cam_position_init[cam_id][2]+0.1),     # Height variation
        }
        
        # Randomly sample camera parameters
        pos_x = random.uniform(*pos_ranges['x'])
        pos_y = random.uniform(*pos_ranges['y'])
        pos_z = random.uniform(*pos_ranges['z'])
        
        # Update camera self.cam_position
        self.model.cam_pos[cam_id] = np.array([pos_x, pos_y, pos_z])
        
        # Force update scene
        mj.mj_forward(self.model, self.data)

    def activate_emergency_stop(self):
        """
        Active the e-stop, robot pose hold
        """
        self.emergency_stop = True
        # self.emergency_stop_first_set = True

        self.current_qpos = []
        for joint_idx in range(self.model.nu):
            self.current_qpos.append(self.data.qpos[self.model.jnt_qposadr[joint_idx]])
        self.emergency_stop_pos_first_set = True
        print("Emergency stop triggered. Robot will hold position.")

    def controller(self, model, data):
        if self.emergency_stop:
            data.ctrl = self.current_qpos
            # print("Emergency Stop Activated! Robot is holding position.")
            return

        if self.start_time is None:
            self.start_time = data.time
            self.current_step = 0
            self.transition_start_time = data.time
            self.current_target = self.trajectory[0]
            self.next_target = self.trajectory[1] if len(self.trajectory) > 1 else None
            self.is_pausing = True  # Start with a pause at the first pose
            self.pause_start_time = data.time
            return

        # Handle UR5 control first (original 6 joints)
        if self.current_target is not None:
            for joint_idx in range(min(6, len(self.current_target))):
                data.ctrl[joint_idx] = self.current_target[joint_idx]

        elapsed_time = data.time - self.start_time
        if elapsed_time < self.initial_delay:
            return

        current_angle = data.qpos[model.jnt_qposadr[model.nu-1]]
        angle_diff = self.target_angle - current_angle
        
        # Check if we're close to target
        ANGLE_THRESHOLD = 0.005  # About 0.6 degrees threshold
        
        if abs(angle_diff) < ANGLE_THRESHOLD:
            if not self.reached_target:
                self.reached_target = True
                if self.reverse_on_target:
                    # Reverse direction by setting new target to opposite angle
                    self.target_angle = -self.target_angle
                else:
                    # Stop at target
                    data.ctrl[6] = 0
                    return
        
        # Apply rotation with smoothed speed near target
        smoothing_factor = min(1.0, abs(angle_diff) / 0.05)  # Smooth speed when within 0.1 rad of target
        panel_rotation = self.rotation_speed * np.sign(angle_diff) * smoothing_factor
        # Set the panel rotation control
        data.ctrl[6] = panel_rotation

        # Handle pausing at key poses
        if self.is_pausing:
            pause_elapsed = data.time - self.pause_start_time
            if pause_elapsed < self.joint_pause:
                # Hold the current position during pause
                if self.current_target is not None:
                    for joint_idx, position in enumerate(self.current_target):
                        data.ctrl[joint_idx] = position
                return
            else:
                # End pause and prepare for next transition
                self.is_pausing = False
                self.transition_start_time = data.time

        # Calculate transition duration based on speed scale
        transition_duration = 1.0 / self.speed_scale  # Adjust this base duration as needed
        
        # Check if we need to move to the next target
        current_transition_time = data.time - self.transition_start_time
        if current_transition_time >= transition_duration and self.next_target is not None:
            self.current_step += 1
            self.current_target = self.next_target
            self.next_target = (self.trajectory[self.current_step + 1] 
                              if self.current_step + 1 < len(self.trajectory) 
                              else None)
            self.transition_start_time = data.time
            current_transition_time = 0
            # Start pause at new key pose
            self.is_pausing = True
            self.pause_start_time = data.time
            return

        # If we have both current and next targets, interpolate between them
        if self.current_target is not None and self.next_target is not None:
            # Calculate interpolation factor
            t = min(current_transition_time / transition_duration, 1.0)
            
            # Linear interpolation between current and next target
            for joint_idx in range(len(self.current_target)):
                current_pos = self.current_target[joint_idx]
                next_pos = self.next_target[joint_idx]
                interpolated_pos = current_pos + t * (next_pos - current_pos)
                data.ctrl[joint_idx] = interpolated_pos
        # If we only have current target (last position), hold it
        elif self.current_target is not None:
            for joint_idx, position in enumerate(self.current_target):
                data.ctrl[joint_idx] = position
        
    def data_collection(self, object_body_id, fixed_panel_body_id):
        """
        Collect transformation data between adjacent timesteps for both objects.
        Returns positions, rotations, and relative transforms between timesteps.
        """
        # Get current positions and orientations
        object_pos = self.data.xpos[object_body_id].copy()  # 3D position
        object_quat = self.data.xquat[object_body_id].copy()  # Quaternion orientation
        fixed_panel_pos = self.data.xpos[fixed_panel_body_id].copy()
        fixed_panel_quat = self.data.xquat[fixed_panel_body_id].copy()

        # print("object_pos", object_pos)
        # print("fixed_panel_pos", fixed_panel_pos)
        
        # Get end effector position (wrist_3_link)
        wrist_3_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'wrist_3_link')
        end_effector_pos = self.data.xpos[wrist_3_body_id].copy()
        # print("end_effector_pos", end_effector_pos)

        # Store velocities
        object_lin_vel = self.data.qvel[self.model.body_jntadr[object_body_id]:self.model.body_jntadr[object_body_id]+3].copy()
        object_ang_vel = self.data.qvel[self.model.body_jntadr[object_body_id]+3:self.model.body_jntadr[object_body_id]+6].copy()
        # fixed_panel_lin_vel = self.data.cvel[fixed_panel_body_id].reshape((6,))[:3].copy()
        
        # Convert quaternions to rotation matrices (3x3)
        def quat_to_mat(quat):
            w, x, y, z = quat
            return np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
            ])
        
        object_rot = quat_to_mat(object_quat)
        fixed_panel_rot = quat_to_mat(fixed_panel_quat)
        
        # If this is not the first frame, calculate transforms between frames
        if hasattr(self, 'prev_object_pos'):
            # Calculate translation vectors (movement since last frame)
            object_translation = object_pos - self.prev_object_pos
            fixed_panel_translation = fixed_panel_pos - self.prev_fixed_panel_pos
            
            # Calculate rotation matrices between frames
            # R2 = dR * R1 -> dR = R2 * R1^T
            object_rot_delta = object_rot @ self.prev_object_rot.T
            fixed_panel_rot_delta = fixed_panel_rot @ self.prev_fixed_panel_rot.T
            
            # Calculate relative transform between object and fixed panel
            relative_pos = fixed_panel_pos - object_pos
            
            transform_data_3D = {
                'object_translation': object_translation,
                'object_rotation_delta': object_rot_delta,
                'fixed_panel_translation': fixed_panel_translation,
                'fixed_panel_rotation_delta': fixed_panel_rot_delta,
                'relative_position': relative_pos
            }
        else:
            # For first frame, set deltas to identity/zero
            transform_data_3D = {
                'object_translation': np.zeros(3),
                'object_rotation_delta': np.eye(3),
                'fixed_panel_translation': np.zeros(3),
                'fixed_panel_rotation_delta': np.eye(3),
                'relative_position': fixed_panel_pos - object_pos
            }
        
        # Store current transforms for next frame
        self.prev_object_pos = object_pos
        self.prev_object_rot = object_rot
        self.prev_fixed_panel_pos = fixed_panel_pos
        self.prev_fixed_panel_rot = fixed_panel_rot
        
        return end_effector_pos, transform_data_3D

    # Function to calculate action value based on displacement, linear speed, and angular speed
    def calculate_action(self, displacement, object_pos, panel_pos):
        """
        Calculate action value based on object position relative to panel.
        
        Args:
            displacement: Current displacement vector between object and panel
            object_pos: 3D position of the object (bunny)
            panel_pos: 3D position of the panel
            
        Returns:
            float: Action value (0.0 = safe on panel, 1.0 = fallen off panel)
        """
        # print("displacement", displacement)
        # Check if object is below the panel surface (has fallen)
        if object_pos[2] < panel_pos[2] - 0.02:  # Small threshold to account for contact
            return 1.0
        
        # Check horizontal displacement (if object center is too far from panel center)
        # panel_radius = 0.15  # Approximate radius of panel - adjust based on your model
        # horizontal_dist = np.sqrt(displacement[0]**2 + displacement[1]**2)
        
        # if horizontal_dist > panel_radius:
        #     return 1.0  # Object center is outside panel radius
            
        # Object is safely on the panel
        return 0.0

    def get_camera_image(self, camera_name):
        """
        Get two images from the specified camera - one for fixed panel and one for object.
        
        Args:
            camera_name (str): Name of the camera to capture from
            
        Returns:
            tuple: Two RGB image arrays (fixed_panel_img, object_img) each of shape (height, width, 3)
        """
        # Skip if cameras are disabled
        if not self.enable_cameras:
            return None, None
    
        # Get camera id
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")
        
        # Get image dimensions
        width = 640
        height = 640
        
        # Initialize image arrays for both fixed panel and object
        fixed_panel_img = np.zeros((height, width, 3), dtype=np.uint8)
        object_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create camera instance
        cam = mj.MjvCamera()
        # Copy camera configuration from model to camera instance
        cam.type = mj.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
        
        # Save current geom groups visibility
        original_geomgroup = self.opt.geomgroup.copy()
        
        if camera_name in ['top_camera', 'front_camera']:
            # Set up viewport
            viewport = mj.MjrRect(0, 0, width, height)
            
            # First render - fixed panel only
            self.opt.geomgroup[:] = 0  # Hide all groups
            self.opt.geomgroup[1] = 1  # Show only fixed panel
            
            # Update and render scene for fixed panel
            mj.mjv_updateScene(self.model, self.data, self.opt, None, cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            
            # Read pixels for fixed panel image
            mj.mjr_readPixels(fixed_panel_img, None, viewport, self.context)
            
            # Second render - object only
            self.opt.geomgroup[:] = 0  # Hide all groups
            self.opt.geomgroup[2] = 1  # Show only object
            
            # Update and render scene for object
            mj.mjv_updateScene(self.model, self.data, self.opt, None, cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            
            # Read pixels for object image
            mj.mjr_readPixels(object_img, None, viewport, self.context)
            
            # Restore original visibility settings
            self.opt.geomgroup[:] = original_geomgroup
            
            return fixed_panel_img, object_img
        
        return None, None  # Return None for both images if camera name not recognized

    def simulate(self, dataset="train"):
        video_dir = './demo'
        os.makedirs(video_dir, exist_ok=True)
        if self.enable_cameras:
            top_camera_video_filename = os.path.join(video_dir, 'top_camera_video.mp4')
            front_camera_video_filename = os.path.join(video_dir, 'front_camera_video.mp4')

            # writer = imageio.get_writer(video_filename, fps=60)
            top_panel_writer = imageio.get_writer(top_camera_video_filename, fps=60, macro_block_size=16)
            top_object_writer = imageio.get_writer(top_camera_video_filename, fps=60, macro_block_size=16)
            front_panel_writer = imageio.get_writer(front_camera_video_filename, fps=60, macro_block_size=16)
            front_object_writer = imageio.get_writer(top_camera_video_filename, fps=60, macro_block_size=16)

        # Create data directories
        os.makedirs('demo/data/train', exist_ok=True)
        os.makedirs('demo/data/val', exist_ok=True)

        # Get object IDs
        object_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'object_body')
        fixed_panel_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_panel')

        # while not glfw.window_should_close(self.window):
        if dataset == "train":
            N_EPISODES = N_TRAIN_EPISODES
        elif dataset == "val":
            N_EPISODES = N_VAL_EPISODES
        else:
            print(f"Unknown dataset type: {dataset}")
            return


        for episode_num in range(N_EPISODES):
            
            random_seed_tmp = random.randint(0, EPISODE_LENGTH)
            print(f"Episode {episode_num+1}/{N_EPISODES}, random seed: {random_seed_tmp}")

            # Initialize episode variables
            failure_time_step = -1
            first_failure_time_step = -1
            object_drop_time = 16  # Steps to ignore while object is initially falling
            window = 30
            episode_filled_tag = False

            # Initialize lists to store metrics
            action_values = []
            top_camera_object_contours = []
            top_camera_panel_contours = []
            front_camera_object_contours = []
            front_camera_panel_contours = []

            self.episode = []  # Reset episode data for each new episode
        
            # Initialize dummy contours for use when cameras are disabled
            dummy_contour = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)

            # Main simulation loop for backtracking
            for overal_step_num in range(EPISODE_LENGTH):
                episode_failed_tag = False
                backtracking_steps = 5

                # Load the 'home' keyframe for initial position
                keyframe_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, 'home')
                if keyframe_id >= 0:
                    mj.mj_resetDataKeyframe(self.model, self.data, keyframe_id)

                # Reset to start pose again at the beginning of each episode
                start_pose = self.trajectory[0]  # 'start' joint pose from traj.txt
                for joint_idx, position in enumerate(start_pose):
                    self.data.qpos[joint_idx] = position

                mj.mj_forward(self.model, self.data)
                
                self.reset(RANDOM_EPISODE_TMP if episode_num == 0 else random_seed_tmp)  # Reset simulation

                # Inner simulation loop
                for step_num in range(EPISODE_LENGTH):
                    simstart = self.data.time
                    while (self.data.time - simstart < 1.0/60.0):
                        # Step simulation environment
                        mj.mj_step(self.model, self.data)

                        # Record the object's position
                        object_pos = self.data.qpos[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'stanford_bunny')]

                    # get framebuffer viewport
                    viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
                    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

                    # Update scene
                    mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                    mj.mjtCatBit.mjCAT_ALL.value, self.scene)

                    # Render scene
                    mj.mjr_render(viewport, self.scene, self.context)
                    

                    # Initialize contour variables with default values
                    top_panel_contour = dummy_contour
                    top_object_contour = dummy_contour
                    front_panel_contour = dummy_contour
                    front_object_contour = dummy_contour

                    # Camera operations only if enabled
                    if self.enable_cameras:
                        # Get top camera frame
                        top_panel_frame, top_object_frame = self.get_camera_image('top_camera')
                        # top_camera_frame = top_camera_frame[::-1, :, :]
                        if top_panel_frame is not None and top_object_frame is not None:
                            # Process panel frame
                            top_panel_writer.append_data(top_panel_frame)
                            top_panel_view = cv2.cvtColor(top_panel_frame, cv2.COLOR_RGB2BGR)
                            top_panel_mask, top_panel_contour, top_panel_filtered = process_camera_frame(top_panel_view)
                            
                            # Process object frame
                            top_object_writer.append_data(top_object_frame)
                            top_object_view = cv2.cvtColor(top_object_frame, cv2.COLOR_RGB2BGR)
                            top_object_mask, top_object_contour, top_object_filtered = process_camera_frame(top_object_view)

                        # Get front camera frames
                        front_panel_frame, front_object_frame = self.get_camera_image('front_camera')
                        if front_panel_frame is not None and front_object_frame is not None:
                            # Rotate and process panel frame
                            front_panel_frame = cv2.rotate(front_panel_frame, cv2.ROTATE_90_CLOCKWISE)
                            front_panel_writer.append_data(front_panel_frame)
                            front_panel_view = cv2.cvtColor(front_panel_frame, cv2.COLOR_RGB2BGR)
                            front_panel_mask, front_panel_contour, front_panel_filtered = process_camera_frame(front_panel_view)
                            
                            # Rotate and process object frame
                            front_object_frame = cv2.rotate(front_object_frame, cv2.ROTATE_90_CLOCKWISE)
                            front_object_writer.append_data(front_object_frame)
                            front_object_view = cv2.cvtColor(front_object_frame, cv2.COLOR_RGB2BGR)
                        front_object_mask, front_object_contour, front_object_filtered = process_camera_frame(front_object_view)

                    # Process both frames

                    # Display images
                    if self.display_camera:
                        if self.display_camera:
                            cv2.imshow('Top Camera Panel View', top_panel_filtered)
                            cv2.imshow('Top Camera Object View', top_object_filtered)
                            cv2.imshow('Front Camera Panel View', front_panel_filtered)
                            cv2.imshow('Front Camera Object View', front_object_filtered)

                        # Check for 'q' key press to quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # swap OpenGL buffers (blocking call due to v-sync)
                    glfw.swap_buffers(self.window)

                    # process pending GUI events, call GLFW callbacks
                    glfw.poll_events()

                    # skip the first object_drop_time steps because the object is falling from the sky
                    if step_num >= object_drop_time:
                        # print("step_num", step_num)
                        end_effector_pos, transform_data_3D = self.data_collection(object_body_id, fixed_panel_body_id)
                        displacement = transform_data_3D['relative_position']

                        # Get current positions
                        object_pos = self.data.xpos[object_body_id].copy()
                        panel_pos = self.data.xpos[fixed_panel_body_id].copy()

                        action_value = self.calculate_action(displacement, object_pos, panel_pos)

                        # Historical backtracking for backtracking_steps time steps
                        # print("episode_filled_tag!!!!!!", episode_filled_tag)
                        if episode_filled_tag:
                            if step_num == (failure_time_step - backtracking_steps):
                                self.activate_emergency_stop()  # manually active the e-stop, robot pose hold
                                continue
                            if step_num > (failure_time_step - backtracking_steps) and action_value == 1.0: # after e-stop, still failed enven e-stop in advance
                                episode_failed_tag = True
                            continue
                        else:
                            # Once if action_value == 1 for the first time, it will always == 1.
                            # If failed, apply e-stop
                            # print("failure_time_step!!!!!!", failure_time_step)
                            if action_value == 1.0 and not self.emergency_stop and not self.emergency_stop_pos_first_set:
                                # print("!!!!!!!!self.emergency_stop", self.emergency_stop)
                                self.activate_emergency_stop()  # automatically active the e-stop, robot pose hold
                                episode_failed_tag = True
                                failure_time_step = step_num
                                # object_drop_time is for avoid counting in the initial falling of the object
                                first_failure_time_step = step_num - object_drop_time - window # + episode_num * (EPISODE_LENGTH - object_drop_time)

                            top_camera_object_contours.append(top_object_contour)
                            top_camera_panel_contours.append(top_panel_contour)
                            front_camera_object_contours.append(front_object_contour)
                            front_camera_panel_contours.append(front_panel_contour)

                            if len(top_camera_object_contours) > window:
                                top_object_transforms = process_consecutive_frames(top_camera_object_contours[-window], top_object_contour)
                                # print("top_object_transforms", top_object_transforms)
                                top_panel_transforms = process_consecutive_frames(top_camera_panel_contours[-window], top_panel_contour)
                                # print("top_panel_transforms", top_panel_transforms)
                                front_object_transforms = process_consecutive_frames(front_camera_object_contours[-window], front_object_contour)
                                # print("front_object_transforms", front_object_transforms)
                                front_panel_transforms = process_consecutive_frames(front_camera_panel_contours[-window], front_panel_contour)
                                # print("front_panel_transforms", front_panel_transforms)

                                try:
                                    # Check for empty transforms
                                    if len(top_object_transforms) == 0 and len(top_panel_transforms) == 0 and \
                                       len(front_object_transforms) == 0 and len(front_panel_transforms) == 0:
                                        raise Exception("All transforms are empty")
                                except Exception as e:
                                    print(f"Error in contour processing: {e}")
                                    continue

                                # Extract features from each transform set
                                top_object_features = extract_transform_features(top_object_transforms)
                                top_panel_features = extract_transform_features(top_panel_transforms)
                                front_object_features = extract_transform_features(front_object_transforms)
                                front_panel_features = extract_transform_features(front_panel_transforms)
                                # print("top_object_transforms",top_object_transforms)
                                # print("top_object_features",top_object_features)
                                
                                # Combine all features
                                combined_features = np.concatenate([
                                    top_object_features,
                                    top_panel_features,
                                    front_object_features,
                                    front_panel_features,
                                    end_effector_pos
                                ])

                                # print("top_object_features", np.asarray(top_object_features).shape) # (4,)
                                # print("top_panel_features", np.asarray(top_panel_features).shape) # (4,)
                                # print("front_object_features", np.asarray(front_object_features).shape) # (4,)
                                # print("front_panel_features", np.asarray(front_panel_features).shape) # (4,)
                                # print("end_effector_pos", np.asarray(end_effector_pos).shape) # (3,)
                                # print("combined_features", combined_features.shape) # (19,)
                                if combined_features.shape[0]!=19:
                                    raise ValueError(f"Error: combined_features shape {combined_features.shape} != 19")

                                self.episode.append({
                                # 'image': top_panel_frame,
                                # 'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
                                'state': np.asarray(combined_features, dtype=np.float32),  # Save the padded state
                                'action': np.asarray([action_value], dtype=np.float32),  # Ensure action is a tensor of shape (1,)
                                # 'language_instruction': 'dummy instruction',
                                    })
                                # For plot     
                                print("action_value!!!!!!!!!!!!step_num - object_drop_time", step_num - object_drop_time - window, action_value) # + episode_num * (EPISODE_LENGTH - object_drop_time), action_value)
                                # action_values.append(action_value) # Action value

                # After a round of simulation: 
                episode_filled_tag = True
                print("len(self.episode)", len(self.episode), "episode_filled_tag", episode_filled_tag, "episode_failed_tag", episode_failed_tag, "failure_time_step", failure_time_step)

                failure_time_step -= backtracking_steps
                # object_drop_time is for avoid counting in the initial falling of the object
                failure_time_step_trim = failure_time_step - object_drop_time - window #  + episode_num * (EPISODE_LENGTH - object_drop_time)
                # print("episode_num", episode_num)
                # print("N_EPISODES", N_EPISODES)
                print("first_failure_time_step", first_failure_time_step)
                print("failure_time_step_trim", failure_time_step_trim)

                if episode_failed_tag:
                    # if failure_time_step_trim < 0:
                    #     raise ValueError(f"failure_time_step_trim '{failure_time_step_trim}' < 0")
                    # print("self.episode[failure_time_step_trim-1]['action']", self.episode[failure_time_step_trim-1]['action'])
                    # print("self.episode[failure_time_step_trim]['action']", self.episode[failure_time_step_trim]['action'])
                    
                    # Set action to 1 at failure point
                    # self.episode[failure_time_step_trim]['action'] = np.asarray([1.0], dtype=np.float32)
                    # action_values[failure_time_step_trim] = 1

                    # Set actions to 1 for failure range
                    # for idx in range(failure_time_step_trim, first_failure_time_step+1):
                    #     # print("idx", idx)
                    #     self.episode[idx]['action'] = np.asarray([1.0], dtype=np.float32)
                        # action_values[idx] = 1
                    pass
                # If backtracking was successful (no failure after applying e-stop)
                elif first_failure_time_step != -1: # if not failed after backtracking, the time step to apply e-stop is safe, backtracking over
                    # self.episode[failure_time_step_trim]['action'] = np.asarray([0.0], dtype=np.float32)
                    # action_values[failure_time_step_trim] = 0

                    # Interpolate values from 0 to 1 for latest failure_time_step to first_failure_time_step
                    interpolated_values = linear_interpolation(first_failure_time_step, failure_time_step_trim)
                    # Update the "action" key for the dictionaries between i and k
                    for idx, value in enumerate(interpolated_values, start=failure_time_step_trim):
                        # print("value", value)
                        self.episode[idx]["action"] = np.asarray([value], dtype=np.float32)
                    #     action_values[idx] = value
                    break
                # If no failure occurred at all during the episode
                elif first_failure_time_step == -1:
                    break
        
            # Resample the data
            episode_resampled = self.resample_data(self.episode)
            
            # Plot after simulation
            self.plot_metrics(self.episode, episode_resampled, episode_num)

            if dataset == "train":
                print("Generating train examples...")
                np.save(f'demo/data/train/episode_{episode_num}.npy', episode_resampled)
            elif dataset == "val":
                print("Generating val examples...")
                np.save(f'demo/data/val/episode_{episode_num}.npy', episode_resampled)

        # writer.close()
        if self.display_camera:
            top_panel_writer.close()
            top_object_writer.close()
            front_panel_writer.close()
            front_object_writer.close()
        glfw.terminate()

    def resample_data(self, episode):
        episode_resampled = []
        scale=15
        for item_idx in range(len(episode)):
            print(episode[item_idx]['action'][0])
            if episode[item_idx]['action'][0] == 0.0 and item_idx%scale==0:
                episode_resampled.append(episode[item_idx])
            elif episode[item_idx]['action'][0] > 0.0 and episode[item_idx]['action'][0] < 1.0:
                episode_resampled.append(episode[item_idx])
            elif episode[item_idx]['action'][0] == 1.0 and item_idx%scale==0:
                episode_resampled.append(episode[item_idx])
        return episode_resampled
                     

    def plot_metrics(self, original_episode, resampled_episode, episode_num):
        """
        Visualize original and resampled action curves in a single plot.
        
        Parameters:
        -----------
        original_episode : list
            Original episode data
        resampled_episode : list
            Resampled episode data
        episode_num : int
            Episode number for the filename
        """
        # Extract original data
        original_time_steps = range(len(original_episode))
        original_risk_values = [item['action'][0] for item in original_episode]
        
        # Map resampled points to their original indices
        resampled_indices = []
        for r_item in resampled_episode:
            # Find matching item in original episode
            for i, o_item in enumerate(original_episode):
                if np.array_equal(r_item['state'], o_item['state']) and r_item['action'][0] == o_item['action'][0]:
                    resampled_indices.append(i)
                    break
        
        # Extract resampled values
        resampled_risk_values = [item['action'][0] for item in resampled_episode]
        
        # Create figure and plot
        plt.figure(figsize=(12, 6))
        
        # Plot original data as a continuous line
        plt.plot(original_time_steps, original_risk_values, 'b-', 
                linewidth=2, alpha=0.6, label='Original data')
        
        # Plot resampled data as a dotted line with markers
        plt.plot(resampled_indices, resampled_risk_values, 'r--', 
                linewidth=1.5, alpha=0.8, label='Resampled curve')
        
        # Add markers for resampled points
        plt.scatter(resampled_indices, resampled_risk_values, 
                    color='red', s=40, label='Resampled points', zorder=5)
        
        # Enhance the plot
        plt.title('Comparison of Original and Resampled Action Values')
        plt.xlabel('Time Step')
        plt.ylabel('Risk Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text annotation showing reduction
        reduction = 100 * (1 - len(resampled_episode)/len(original_episode))
        plt.annotate(f"Data reduction: {reduction:.1f}%\nOriginal: {len(original_episode)} points\nResampled: {len(resampled_episode)} points", 
                    xy=(0.02, 0.96), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'./demo/comparison_episode{episode_num}.png')
        plt.close()

def main():
    xml_path = "./model/universal_robots_ur5e/test_scene_complete.xml"
    traj_path = "./demo/traj.txt"  # Adjust path as needed

    camera_related = True

    sim = Projectile(xml_path, traj_path, initial_delay=2, display_camera=camera_related, enable_cameras=camera_related)
    sim.reset(RANDOM_EPISODE_TMP)
    sim.simulate(sys.argv[1])


if __name__ == "__main__":
    main()