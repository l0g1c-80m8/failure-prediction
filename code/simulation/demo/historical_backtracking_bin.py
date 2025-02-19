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


N_TRAIN_EPISODES = 1
N_VAL_EPISODES = 10
EPISODE_LENGTH = 1200  # Number of points in trajectory

# Thresholds for action calculation
DISPLACEMENT_THRESHOLD_HIGH = 0.01
DISPLACEMENT_THRESHOLD_LOW = 0

RANDOM_EPISODE_TMP = random.randint(0, EPISODE_LENGTH) # 67 86 #  109

# Define different interpolation methods
def linear_interpolation(first_failure_time_step, failure_time_step_trim):
    return np.linspace(0, 1, first_failure_time_step - failure_time_step_trim + 1)

def sin_interpolation(first_failure_time_step, failure_time_step_trim):
    x = np.linspace(0, 1, first_failure_time_step - failure_time_step_trim + 1)
    return np.sin(x)

class Projectile(MuJoCoBase):
    def __init__(self, xml_path, traj_file, initial_delay=3.0):
        super().__init__(xml_path)
        
        # self.init_angular_speed = 1.0  # Angular speed in radians per second
        self.initial_delay = initial_delay  # Delay before starting movement
        self.speed_scale = random.uniform(0.5, 1.0)  # New parameter to control joint speed
        self.joint_pause = random.uniform(0.2, 0.8)  # Duration of pause between movements
        self.start_time = None  # To track the start time
        self.positions = []  # To store the position data
        self.episode = [] # To store the episode data
        # self.human_intervene = False  # New attribute to track cube drop
        # self.intervene_step = 0  # Step when human intervention occurs
        # self.is_intervene_step_set = False
        self.high_threshold_step = 0  # Step when displacement > DISPLACEMENT_THRESHOLD_HIGH
        self.is_high_threshold_step_set = False
        
        self.traj_file = traj_file
        self.trajectory = self.load_trajectory()
        self.current_step = 0
        self.current_target = None
        self.next_target = None
        self.transition_start_time = None
        self.ncam = self.model.ncam  # Get number of cameras in the model
        self.cam_position = [None] * self.ncam  # Initialize with None for each camera
        self.cam_position_read = [False] * self.ncam  # Initialize all as unread

        # Add visualization markers# Marker appearance settings
        self.marker_size = 0.005  # Increased size - adjust this value as needed
        self.marker_positions = []  # Store positions and colors
        self.max_markers = 100  # Maximum number of position markers to show
        
        # Initialize marker colors
        self.cube_marker_color = [1, 0, 0, 0.8]  # Red for cube
        self.box_marker_color = [0, 1, 0, 0.8]   # Green for fixed box

        # Create windows
        # cv2.namedWindow('Top Camera View', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Front Camera View', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Side Camera View', cv2.WINDOW_NORMAL)
        
        # Set window sizes
        # cv2.resizeWindow('Top Camera View', 640, 640)
        # cv2.resizeWindow('Front Camera View', 640, 640)
        # cv2.resizeWindow('Side Camera View', 640, 640)

    def add_position_markers(self, cube_pos, box_pos):
        """Add markers for both cube and fixed box positions"""
        # Add new positions and colors
        self.marker_positions.append((cube_pos, self.cube_marker_color))
        self.marker_positions.append((box_pos, self.box_marker_color))
        
        # Keep only the most recent markers
        while len(self.marker_positions) > self.max_markers * 2:  # *2 because we add two markers at a time
            self.marker_positions.pop(0)

    def add_markers_to_scene(self):
        """Add all stored markers to the scene"""
        if not self.marker_positions:
            return
            
        for pos, color in self.marker_positions:
            self.scene.ngeom += 1
            g = self.scene.geoms[self.scene.ngeom - 1]
            
            # Set geometry properties
            g.type = mj.mjtGeom.mjGEOM_SPHERE
            g.size[:] = [self.marker_size, self.marker_size, self.marker_size]  # Use the configurable size
            g.pos[:] = pos
            g.mat[:,:] = np.eye(3)
            g.rgba[:] = color
            g.dataid = -1
            g.objtype = mj.mjtObj.mjOBJ_UNKNOWN
            g.objid = -1

    def load_trajectory(self):
        """Load trajectory from a text file containing lists of joint positions"""
        with open(self.traj_file, "r") as file:
            trajectory = []
            for line in file:
                try:
                    # Strip whitespace and convert string representation of list to numpy array
                    joint_positions = np.array(ast.literal_eval(line.strip()))
                    trajectory.append(joint_positions)
                except Exception as e:
                    print(f"Error parsing line: {line.strip()}. Error: {e}")
                    continue
                    
            print(f"Loaded {len(trajectory)} waypoints")
            return trajectory

    def reset(self, seed):
        random.seed(seed)


        # Set camera configuration
        self.cam.azimuth = 300 # -250 random.uniform(-225, -315)
        self.cam.distance = 2.5 # random.uniform(2, 3)
        self.cam.elevation = -40 # random.uniform(-50, -30)
        # print("self.cam", self.cam.azimuth, self.cam.distance, self.cam.elevation)

        self.randomize_camera_position('top_camera')
        self.randomize_camera_position('front_camera')

        # Add random rotation parameters
        # Subtle random rotation parameters
        self.rotation_speed = random.uniform(-0.01, 0.01)  # Slower random speed between 0.2 and 0.5 rad/s
        self.target_angle = random.uniform(-0.03, 0.03)   # Small random angle between -0.3 and 0.3 rad (about ±17 degrees)
        self.reverse_on_target = random.choice([True, False])  # Randomly decide to stop or reverse
        self.reached_target = False  # Track if we've reached target
        # print(f"New rotation parameters - Speed: {self.rotation_speed:.2f} rad/s, Target angle: {self.target_angle:.2f} rad, Reverse: {self.reverse_on_target}")

        # self.angular_speed = self.init_angular_speed
        # self.human_intervene = False  # New attribute to track cube drop
        # self.intervene_step = 0  # Step when human intervention occurs
        # self.is_intervene_step_set = False
        self.high_threshold_step = 0  # Step when displacement > DISPLACEMENT_THRESHOLD_HIGH
        self.is_high_threshold_step_set = False

        # Initialize start time
        self.start_time = None

        self.current_step = 0
        self.current_target = None
        self.next_target = None
        self.transition_start_time = None
        self.speed_scale = random.uniform(0.5, 1.0)  # New parameter to control joint speed
        self.joint_pause = random.uniform(0.2, 0.8)  # Duration of pause between movements

        self.emergency_stop = False  # Flag to trigger the emergency stop
        self.emergency_stop_pos_first_set = False  # Flag to first set the emergency stop pose
        self.current_qpos = None

        # Add floor randomization
        self.randomize_floor()
        # Randomize fixed_box and free_cube
        self.randomize_object_colors()

        # cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')
        # fixed_box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_box')
        # self.data.qpos[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3] = [random.uniform(0.35, 0.42), random.uniform(-0.5, -0.35), random.uniform(0.2, 0.35)]
        self.randomize_free_cube()

        # Clear markers when resetting
        self.marker_geoms = []
        self.marker_counter = 0

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


    def randomize_free_cube(self):
        """Randomize the free cube's size, mass, friction, and other physical properties."""
        # Get cube body and geom IDs
        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')
        cube_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'sliding_cube')
        
        # Randomize cube size (within reasonable bounds)
        base_size = 0.015  # Original size
        size_variation = random.uniform(0.8, 3)  # n% variation
        new_size = base_size * size_variation
        self.model.geom_size[cube_geom_id] = [new_size, new_size, new_size]
        
        # Randomize mass (scaled with size)
        base_mass = 0.5  # Original mass
        mass_variation = random.uniform(0.8, 1.2)  # ±20% variation
        new_mass = base_mass * mass_variation  # base_mass * size_variation * mass_variation  # Scale mass with size
        self.model.body_mass[cube_body_id] = new_mass
        
        # Adjust inertia based on new mass and size
        new_inertia = (new_mass * new_size**2) / 6  # Simple box inertia approximation
        self.model.body_inertia[cube_body_id] = [new_inertia, new_inertia, new_inertia]
        
        # Randomize friction properties
        friction_variation = random.uniform(0.25, 0.35)  # Base friction is 0.2
        # Find the contact pair involving the sliding cube
        for i in range(self.model.npair):
            if (self.model.pair_geom1[i] == cube_geom_id or 
                self.model.pair_geom2[i] == cube_geom_id):
                self.model.pair_friction[i, 0] = friction_variation  # Sliding friction
                self.model.pair_friction[i, 1] = friction_variation * 2.5  # Rolling friction
                self.model.pair_friction[i, 2] = friction_variation * 0.005  # Torsional friction
        
        # Randomize initial position (within reasonable bounds)
        x_pos = random.uniform(0.35, 0.35) # random.uniform(0.35, 0.42)
        y_pos = random.uniform(0.65, 0.65) # random.uniform(-0.5, -0.35)
        z_pos = random.uniform(0.3, 0.4)
        self.data.qpos[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3] = [x_pos, y_pos, z_pos]
        
        # Randomize initial orientation (uncomment when needed)
        # quat = [random.uniform(-1, 1) for _ in range(4)]
        # quat = quat / np.linalg.norm(quat)  # Normalize quaternion
        # self.data.qpos[self.model.body_jntadr[cube_body_id]+3:self.model.body_jntadr[cube_body_id]+7] = quat

    def randomize_object_colors(self):
        """Randomize colors for fixed box and free cube"""
        # Get geom IDs
        fixed_box_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "panel")
        free_cube_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "sliding_cube")
        
        # Generate new colors
        box_color = [random.uniform(0.01, 1.0), random.uniform(0.01, 1.0), random.uniform(0.01, 1.0), 1.0]
        cube_color = np.array([1.0, 1.0, 1.0, 2.0]) - box_color  # [random.uniform(0.01, 1.0), random.uniform(0.01, 1.0), random.uniform(0.01, 1.0), 1.0]
        
        # Set new colors
        self.model.geom_rgba[fixed_box_geom_id] = box_color
        self.model.geom_rgba[free_cube_geom_id] = cube_color

    def activate_emergency_stop(self):
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
            # self.joint_pause = 0.5  # Duration of pause at each key pose in seconds
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
        box_rotation = self.rotation_speed * np.sign(angle_diff) * smoothing_factor
        
        # Set the box rotation control
        data.ctrl[6] = box_rotation
        # if int(elapsed_time * 10) % 10 == 0:  # Print every 0.1 seconds
        #     print(f"Time: {elapsed_time:.2f}, Current angle: {current_angle:.2f}, Target: {self.target_angle:.2f}, Speed: {self.rotation_speed:.2f}, Reverse: {self.reverse_on_target}")


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


    def data_collection(self, cube_body_id, fixed_box_body_id):
        """
        Collect transformation data between adjacent timesteps for both objects.
        Returns positions, rotations, and relative transforms between timesteps.
        """
        # Get current positions and orientations
        cube_pos = self.data.xpos[cube_body_id].copy()  # 3D position
        cube_quat = self.data.xquat[cube_body_id].copy()  # Quaternion orientation
        fixed_box_pos = self.data.xpos[fixed_box_body_id].copy()
        fixed_box_quat = self.data.xquat[fixed_box_body_id].copy()
        

        # Add position markers
        self.add_position_markers(cube_pos, fixed_box_pos)


        # Store velocities
        cube_lin_vel = self.data.qvel[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3].copy()
        cube_ang_vel = self.data.qvel[self.model.body_jntadr[cube_body_id]+3:self.model.body_jntadr[cube_body_id]+6].copy()
        fixed_box_lin_vel = self.data.cvel[fixed_box_body_id].reshape((6,))[:3].copy()
        
        # Convert quaternions to rotation matrices (3x3)
        def quat_to_mat(quat):
            w, x, y, z = quat
            return np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
            ])
        
        cube_rot = quat_to_mat(cube_quat)
        fixed_box_rot = quat_to_mat(fixed_box_quat)
        
        # If this is not the first frame, calculate transforms between frames
        if hasattr(self, 'prev_cube_pos'):
            # Calculate translation vectors (movement since last frame)
            cube_translation = cube_pos - self.prev_cube_pos
            fixed_box_translation = fixed_box_pos - self.prev_fixed_box_pos
            
            # Calculate rotation matrices between frames
            # R2 = dR * R1 -> dR = R2 * R1^T
            cube_rot_delta = cube_rot @ self.prev_cube_rot.T
            fixed_box_rot_delta = fixed_box_rot @ self.prev_fixed_box_rot.T
            
            # Calculate relative transform between cube and fixed box
            relative_pos = fixed_box_pos - cube_pos
            relative_rot = fixed_box_rot @ cube_rot.T
            
            transform_data = {
                'cube_translation': cube_translation,
                'cube_rotation_delta': cube_rot_delta,
                'fixed_box_translation': fixed_box_translation,
                'fixed_box_rotation_delta': fixed_box_rot_delta,
                'relative_position': relative_pos,
                'relative_rotation': relative_rot
            }
        else:
            # For first frame, set deltas to identity/zero
            transform_data = {
                'cube_translation': np.zeros(3),
                'cube_rotation_delta': np.eye(3),
                'fixed_box_translation': np.zeros(3),
                'fixed_box_rotation_delta': np.eye(3),
                'relative_position': fixed_box_pos - cube_pos,
                'relative_rotation': fixed_box_rot @ cube_rot.T
            }
        
        # Store current transforms for next frame
        self.prev_cube_pos = cube_pos
        self.prev_cube_rot = cube_rot
        self.prev_fixed_box_pos = fixed_box_pos
        self.prev_fixed_box_rot = fixed_box_rot
        
        # Combine state information (keeping existing format)
        state = np.hstack([
            transform_data['relative_position'],  # 3 values
            cube_lin_vel,                        # 3 values
            cube_ang_vel                         # 3 values
        ])
        
        return state, fixed_box_lin_vel, transform_data


    # Function to calculate action value based on displacement, linear speed, and angular speed
    def calculate_action(self, displacement, linear_speed, angular_speed):
        # Check if all values are in the lower range
        # print("DISPLACEMENT_THRESHOLD_LOW:", DISPLACEMENT_THRESHOLD_LOW, "displacement:", displacement, "DISPLACEMENT_THRESHOLD_HIGH", DISPLACEMENT_THRESHOLD_HIGH)
        # print("LINEAR_SPEED_THRESHOLD_LOW:", LINEAR_SPEED_THRESHOLD_LOW, "linear_speed:", linear_speed, "LINEAR_SPEED_THRESHOLD_HIGH", LINEAR_SPEED_THRESHOLD_HIGH)
        # print("ANGULAR_SPEED_THRESHOLD_LOW:", ANGULAR_SPEED_THRESHOLD_LOW, "angular_speed:", abs(angular_speed), "ANGULAR_SPEED_THRESHOLD_HIGH", ANGULAR_SPEED_THRESHOLD_HIGH)


        # if displacement < DISPLACEMENT_THRESHOLD_LOW and \
        # linear_speed < LINEAR_SPEED_THRESHOLD_LOW and \
        # abs(angular_speed) < ANGULAR_SPEED_THRESHOLD_LOW:
        if displacement[2] < DISPLACEMENT_THRESHOLD_LOW:
            # print("Safe")
            return 0.0  # Set action to 0

        # Check if any value exceeds the higher thresholds
        # if displacement > DISPLACEMENT_THRESHOLD_HIGH or \
        # linear_speed > LINEAR_SPEED_THRESHOLD_HIGH or \
        # abs(angular_speed) > ANGULAR_SPEED_THRESHOLD_HIGH:
        if displacement[2] > DISPLACEMENT_THRESHOLD_HIGH:
            # print("Failure")
            return 1.0  # Set action to 1

        # If the values fall between thresholds, interpolate a reasonable action
        # Normalize between 0 and 1 based on distance from the lower to upper threshold
        # displacement_factor = (displacement - DISPLACEMENT_THRESHOLD_LOW) / \
        #                     (DISPLACEMENT_THRESHOLD_HIGH - DISPLACEMENT_THRESHOLD_LOW)
        # linear_speed_factor = (linear_speed - LINEAR_SPEED_THRESHOLD_LOW) / \
        #                     (LINEAR_SPEED_THRESHOLD_HIGH - LINEAR_SPEED_THRESHOLD_LOW)
        # angular_speed_factor = (abs(angular_speed) - ANGULAR_SPEED_THRESHOLD_LOW) / \
        #                     (ANGULAR_SPEED_THRESHOLD_HIGH - ANGULAR_SPEED_THRESHOLD_LOW)

        # Combine the factors with an average, assuming equal importance
        # action_value = (displacement_factor + linear_speed_factor + angular_speed_factor) / 3.0
        # print("Risk")
        # return action_value
        
        # Just for debugging
        return 0.5

    def randomize_camera_position(self, camera_name='top_camera'):
        """
        Improved camera position randomization with proper orientation handling
        and direct camera parameter updates.
        """
        # Get camera id
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)


        if not self.cam_position_read[cam_id]:
            self.cam_position[cam_id] = self.model.cam_pos[cam_id]
            print(self.cam_position[cam_id])
            self.cam_position_read[cam_id] = True
        
        # Define ranges for camera randomization
        # Wider ranges for more noticeable variation
        pos_ranges = {
            'x': (self.cam_position[cam_id][0]-0.001, self.cam_position[cam_id][0]+0.001),    # Wider range for x offset
            'y': (self.cam_position[cam_id][1]-0.001, self.cam_position[cam_id][1]+0.001),    # Wider range for y offset
            'z': (self.cam_position[cam_id][2]-0.01, self.cam_position[cam_id][2]+0.01),     # Height variation
            'azimuth': (-5, 5),  # Degrees of rotation around vertical axis
            'elevation': (-1, 0), # Degrees of tilt
        }
        
        # Randomly sample camera parameters
        pos_x = random.uniform(*pos_ranges['x'])
        pos_y = random.uniform(*pos_ranges['y'])
        pos_z = random.uniform(*pos_ranges['z'])
        
        # Convert angles to radians for rotation calculation
        # azimuth = np.radians(random.uniform(*pos_ranges['azimuth']))
        # elevation = np.radians(random.uniform(*pos_ranges['elevation']))
        
        # Update camera self.cam_position
        self.model.cam_pos[cam_id] = np.array([pos_x, pos_y, pos_z])
        
        # Calculate rotation matrix
        # First rotate around Y axis (elevation)
        # Ry = np.array([
        #     [np.cos(elevation), 0, np.sin(elevation)],
        #     [0, 1, 0],
        #     [-np.sin(elevation), 0, np.cos(elevation)]
        # ])
        
        # # Then rotate around Z axis (azimuth)
        # Rz = np.array([
        #     [np.cos(azimuth), -np.sin(azimuth), 0],
        #     [np.sin(azimuth), np.cos(azimuth), 0],
        #     [0, 0, 1]
        # ])
        
        # # Combine rotations
        # R = Rz @ Ry
        
        # # Convert rotation matrix to quaternion
        # # Using simplified method since we know R is orthogonal
        # trace = np.trace(R)
        # if trace > 0:
        #     S = np.sqrt(trace + 1.0) * 2
        #     qw = 0.25 * S
        #     qx = (R[2, 1] - R[1, 2]) / S
        #     qy = (R[0, 2] - R[2, 0]) / S
        #     qz = (R[1, 0] - R[0, 1]) / S
        # else:
        #     if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        #         S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        #         qw = (R[2, 1] - R[1, 2]) / S
        #         qx = 0.25 * S
        #         qy = (R[0, 1] + R[1, 0]) / S
        #         qz = (R[0, 2] + R[2, 0]) / S
        #     elif R[1, 1] > R[2, 2]:
        #         S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        #         qw = (R[0, 2] - R[2, 0]) / S
        #         qx = (R[0, 1] + R[1, 0]) / S
        #         qy = 0.25 * S
        #         qz = (R[1, 2] + R[2, 1]) / S
        #     else:
        #         S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        #         qw = (R[1, 0] - R[0, 1]) / S
        #         qx = (R[0, 2] + R[2, 0]) / S
        #         qy = (R[1, 2] + R[2, 1]) / S
        #         qz = 0.25 * S
        
        # # Update camera quaternion
        # self.model.cam_quat[cam_id] = np.array([qw, qx, qy, qz])
        
        # Make sure camera is looking at the scene center
        # target_pos = np.array([0, 0, 0])  # Scene center
        # cam_pos = self.model.cam_pos[cam_id]
        # forward = target_pos - cam_pos
        # forward = forward / np.linalg.norm(forward)
        
        # Update camera orientation to look at target
        # self.model.cam_pos[cam_id] = cam_pos
        
        # Force update scene
        mj.mj_forward(self.model, self.data)

    def get_camera_image(self, camera_name):
        """
        Get image from the specified camera with selective rendering.
        
        Args:
            camera_name (str): Name of the camera to capture from
            
        Returns:
            np.ndarray: RGB image array of shape (height, width, 3)
        """
        # Get camera id
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")
        
        # Get image dimensions
        width = 640
        height = 640
        
        # Initialize image array
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create camera instance
        cam = mj.MjvCamera()
        
        # Copy camera configuration from model to camera instance
        cam.type = mj.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
        
        # Configure scene to show specific groups for different cameras
        if camera_name in ['top_camera', 'front_camera']:
            # Save current geom groups visibility
            original_geomgroup = self.opt.geomgroup.copy()
            
            # For top and front cameras, show only groups 2 (cube) and 3 (fixed box)
            # self.opt.geomgroup[:] = 0  # Hide all groups
            # self.opt.geomgroup[1] = 1  # Show fixed box
            # self.opt.geomgroup[2] = 1  # Show cube
        
        # Update scene
        mj.mjv_updateScene(self.model, self.data, self.opt, None, cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        
        # Set up viewport
        viewport = mj.MjrRect(0, 0, width, height)
        
        # Render scene
        mj.mjr_render(viewport, self.scene, self.context)
        
        # Read pixels
        mj.mjr_readPixels(img, None, viewport, self.context)
        
        # Restore original visibility settings if modified
        if camera_name in ['top_camera', 'front_camera']:
            self.opt.geomgroup[:] = original_geomgroup
        
        return img


    def validate_episode(self, episode):
        required_keys = ['image', 'state', 'action', 'language_instruction']
        
        for i, item in enumerate(episode):
            # Check if all required keys are present
            for key in required_keys:
                if key not in item:
                    print(f"Episode {i} is missing key: {key}")
                    return False
                
                # Check if the value is not None or empty
                value = item[key]
                if value is None:
                    print(f"Episode {i} has None value for key: {key}")
                    return False
                
                # Additional checks for specific keys
                if key == 'image' and not isinstance(value, (np.ndarray, list)):
                    print(f"Episode {i} has invalid type for 'image': {type(value)}")
                    return False
                
                if key == 'state' and (not isinstance(value, np.ndarray) or value.size == 0):
                    print(f"Episode {i} has invalid or empty 'state': {value}")
                    return False
                
                # if i == 124:
                # if key == 'action':
                #     print("isinstance(value, np.ndarray)", isinstance(value, np.ndarray))
                #     print("value.shape", value.shape)
                if key == 'action' and (not isinstance(value, np.ndarray) or value.shape != (1,)):
                    print(f"Episode {i} has invalid 'action': {value}")
                    return False
                
                if key == 'language_instruction' and not isinstance(value, str):
                    print(f"Episode {i} has invalid 'language_instruction': {value}")
                    return False

        print("All episodes are valid.")
        return True

    def simulate(self, dataset="train"):
        video_dir = './demo'
        video_filename = os.path.join(video_dir, 'simulation_video.mp4')
        top_camera_video_filename = os.path.join(video_dir, 'top_camera_video.mp4')
        front_camera_video_filename = os.path.join(video_dir, 'front_camera_video.mp4')

        writer = imageio.get_writer(video_filename, fps=60)
        top_camera_writer = imageio.get_writer(top_camera_video_filename, fps=60, macro_block_size=16)
        front_camera_writer = imageio.get_writer(front_camera_video_filename, fps=60, macro_block_size=16)

        # Create directory if it does not exist
        os.makedirs(video_dir, exist_ok=True)

        # Transition to the 'start' joint pose from traj.txt
        start_pose = self.trajectory[0]  # 'start' joint pose from traj.txt
        print(f"Moving to the 'start' pose: {start_pose}")
        # for joint_idx, position in enumerate(start_pose):
        #     self.data.qpos[joint_idx] = position  # Directly set the joint positions

        # mj.mj_forward(self.model, self.data)  # Forward dynamics to update the simulation state

        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')
        fixed_box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_box')
        # self.data.qpos[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3] = [0.4, 0.45, 0.6]

        self.positions = []
        self.episode = []  # Reset episode data for each new episode

        # while not glfw.window_should_close(self.window):
        if dataset == "train":
            N_EPISODES = N_TRAIN_EPISODES
        elif dataset == "val":
            N_EPISODES = N_VAL_EPISODES

        for episode_num in range(N_EPISODES):
            
            random_seed_tmp = random.randint(0, EPISODE_LENGTH)
            print("random_seed_tmp", random_seed_tmp)
            failure_time_step = -1
            first_failure_time_step = -1
            cube_drop_time = 50
            # Clear position data
            episode_filled_tag = False

            # Initialize lists to store metrics
            linear_velocities = []
            angular_velocities = []
            displacements = []
            action_values = []
            fixed_box_velocities = []

            for overal_step_num in range(EPISODE_LENGTH):

                episode_failed_tag = False
                backtracking_steps = 5

                # Load the 'home' keyframe for initial position
                keyframe_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, 'home')
                if keyframe_id >= 0:
                    mj.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
                # Reset to start pose again at the beginning of each episode
                for joint_idx, position in enumerate(start_pose):
                    self.data.qpos[joint_idx] = position

                mj.mj_forward(self.model, self.data)
                
                # random_seed_tmp = RANDOM_EPISODE_TMP
                # if overal_step_num > 0:
                #     random_seed_tmp = random.randint(0, EPISODE_LENGTH)

                self.reset(RANDOM_EPISODE_TMP if episode_num == 0 else random_seed_tmp)  # Reset simulation
                # DEBUG
                # self.reset(RANDOM_EPISODE_TMP[1])

                for step_num in range(EPISODE_LENGTH):

                    simstart = self.data.time

                    while (self.data.time - simstart < 1.0/60.0):
                        # Step simulation environment
                        mj.mj_step(self.model, self.data)

                        # Record the cube's position
                        cube_pos = self.data.qpos[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')]
                        self.positions.append(cube_pos.copy())

                    # get framebuffer viewport
                    viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
                    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

                    # Update scene
                    mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                    mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                    
                    # Add markers to scene
                    self.add_markers_to_scene()
                    
                    # get framebuffer viewport
                    viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
                    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

                    mj.mjr_render(viewport, self.scene, self.context)

                    # Capture the current frame
                    framebuffer = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)
                    mj.mjr_readPixels(framebuffer, None, viewport, self.context)
                    framebuffer = framebuffer[::-1, :, :]  # Reverse the order of rows
                    writer.append_data(framebuffer)

                    # Get top camera frame
                    top_camera_frame = self.get_camera_image('top_camera')
                    # top_camera_frame = top_camera_frame[::-1, :, :]
                    top_camera_writer.append_data(top_camera_frame)
                    top_camera_view = cv2.cvtColor(top_camera_frame, cv2.COLOR_RGB2BGR)
                    # side_view = self.get_camera_image('side_camera')

                    # Get front camera frame
                    front_camera_frame = self.get_camera_image('front_camera')
                    front_camera_frame = cv2.rotate(front_camera_frame, cv2.ROTATE_90_CLOCKWISE)
                    front_camera_writer.append_data(front_camera_frame)
                    front_camera_view = cv2.cvtColor(front_camera_frame, cv2.COLOR_RGB2BGR)
                    # side_view = cv2.cvtColor(side_view, cv2.COLOR_RGB2BGR)


                    # Display images
                    cv2.imshow('Top Camera View', top_camera_view)
                    cv2.imshow('Front Camera View', front_camera_view)
                    # cv2.imshow('Side Camera View', side_view)

                    # Check for 'q' key press to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


                    # swap OpenGL buffers (blocking call due to v-sync)
                    glfw.swap_buffers(self.window)

                    # process pending GUI events, call GLFW callbacks
                    glfw.poll_events()

                    # skip the first cube_drop_time steps because the cube is falling from the sky
                    if step_num >= cube_drop_time:
                        # print("step_num", step_num)

                        state, fixed_box_linear_speed, transforms = self.data_collection(cube_body_id, fixed_box_body_id)
                        displacement = [state[0], state[1], state[2]]
                        linear_speed = [state[3], state[4], state[5]]
                        angular_speed = [state[6], state[7], state[8]]

                        if displacement[2] >= DISPLACEMENT_THRESHOLD_HIGH:
                            if not self.is_high_threshold_step_set:
                                self.high_threshold_step = step_num  # Mark the step for interpolation endpoint
                                self.is_high_threshold_step_set = True

                        action_value = self.calculate_action(displacement, linear_speed, angular_speed)


                        # Historical backtracking for backtracking_steps time steps
                        # print("episode_filled_tag!!!!!!", episode_filled_tag)
                        if episode_filled_tag:
                            if step_num < (failure_time_step - backtracking_steps):
                                continue
                            elif step_num == (failure_time_step - backtracking_steps):
                                self.activate_emergency_stop()
                            elif action_value == 1.0: # still failed enven e-stop in advance
                                episode_failed_tag = True
                                continue

                        # Once if action_value == 1 for the first time, it will always == 1.
                        # If failed, apply e-stop
                        # print("failure_time_step!!!!!!", failure_time_step)
                        if action_value == 1.0 and not self.emergency_stop:
                            # print("!!!!!!!!self.emergency_stop", self.emergency_stop)
                            self.activate_emergency_stop()
                            episode_failed_tag = True
                            failure_time_step = step_num
                            # cube_drop_time is for avoid counting in the initial falling of the cube
                            first_failure_time_step = step_num - cube_drop_time # + episode_num * (EPISODE_LENGTH - cube_drop_time)

                        if not episode_filled_tag:
                            self.episode.append({
                            'image': top_camera_frame,
                            # 'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
                            # 'state': np.asarray(state, dtype=np.float32),  # Save the padded state
                            'action': np.asarray([action_value], dtype=np.float32),  # Ensure action is a tensor of shape (1,)
                            'language_instruction': 'dummy instruction',
                                })
                            # For plot     
                            print("action_value!!!!!!!!!!!!step_num - cube_drop_time", step_num - cube_drop_time, action_value) # + episode_num * (EPISODE_LENGTH - cube_drop_time), action_value)
                            # displacements.append([state[0], state[1], state[2]])  # Displacement
                            # linear_velocities.append([state[3], state[4], state[5]])  # Linear velocity
                            # angular_velocities.append([state[6], state[7], state[8]])  # Angular velocity
                            action_values.append(action_value) # Action value
                            # fixed_box_velocities.append(fixed_box_linear_speed)
                episode_filled_tag = True
                print("len(self.episode)", len(self.episode), "episode_filled_tag", episode_filled_tag, "episode_failed_tag", episode_failed_tag, "failure_time_step", failure_time_step)

                if episode_failed_tag:
                    failure_time_step -= backtracking_steps
                    # cube_drop_time is for avoid counting in the initial falling of the cube
                    failure_time_step_trim = failure_time_step - cube_drop_time #  + episode_num * (EPISODE_LENGTH - cube_drop_time)
                    # print("episode_num", episode_num)
                    # print("N_EPISODES", N_EPISODES)
                    print("first_failure_time_step", first_failure_time_step)
                    print("failure_time_step_trim", failure_time_step_trim)
                    print("self.episode[failure_time_step_trim-1]['action']", self.episode[failure_time_step_trim-1]['action'])
                    print("self.episode[failure_time_step_trim]['action']", self.episode[failure_time_step_trim]['action'])
                    self.episode[failure_time_step_trim]['action'] = np.asarray([1.0], dtype=np.float32)
                    action_values[failure_time_step_trim] = 1
                    for idx in range(failure_time_step_trim, first_failure_time_step):
                        # print("idx", idx)
                        self.episode[idx]['action'] = np.asarray([1.0], dtype=np.float32)
                        action_values[idx] = 1  
                elif episode_filled_tag and first_failure_time_step != -1: # if not failed after backtracking, the time step to apply e-stop is safe, backtracking over
                    self.episode[failure_time_step_trim]['action'] = np.asarray([0.0], dtype=np.float32)
                    action_values[failure_time_step_trim] = 0

                    # Interpolate values from 0 to 1 for latest failure_time_step to first_failure_time_step
                    interpolated_values = linear_interpolation(first_failure_time_step, failure_time_step_trim)
                    # Update the "action" key for the dictionaries between i and k
                    for idx, value in enumerate(interpolated_values, start=failure_time_step_trim):
                        # print("value", value)
                        self.episode[idx]["action"] = np.asarray([value], dtype=np.float32)
                        action_values[idx] = value
                    break
                elif episode_filled_tag and first_failure_time_step == -1:
                    break
            
            # Validate the completeness
            self.validate_episode(self.episode)

            # if dataset == "train":
            #     print("Generating train examples...")
            #     np.save(f'demo/data/train/episode_{episode_num}.npy', self.episode)
            # elif dataset == "val":
            #     print("Generating val examples...")
            #     np.save(f'demo/data/val/episode_{episode_num}.npy', self.episode)

            # Plot after simulation
            # self.plot_metrics(linear_velocities, angular_velocities, displacements, action_values, fixed_box_velocities, episode_num)


        writer.close()
        top_camera_writer.close()
        front_camera_writer.close()
        glfw.terminate()
        # self.save_trajectory()


    def save_trajectory(self):
        """
        Save or plot the trajectory of the cube.
        """
        positions = np.array(self.positions)
        if len(positions) > 0:
            # Save to a file
            np.savetxt('./demo/cube_trajectory.txt', positions)

            # # Plot the trajectory
            # plt.figure(figsize=(10, 6))
            # plt.plot(positions[:, 0], positions[:, 1], label='Trajectory')
            # plt.xlabel('X Position')
            # plt.ylabel('Y Position')
            # plt.title('Trajectory of the Cube')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig('cube_trajectory.png')
            # plt.show()

    def plot_metrics(self, linear_velocities, angular_velocities, displacements, action_values, fixed_box_velocities, episode_num):
        """
        Plot linear velocity, angular velocity, and displacement over time.
        """
        time_steps = range(len(linear_velocities))
        print("range", range(len(linear_velocities)), range(len(angular_velocities)), range(len(displacements)), range(len(action_values)))  # 800

        # Convert to NumPy array
        linear_velocities = np.array(linear_velocities)
        angular_velocities = np.array(angular_velocities)
        displacements = np.array(displacements)
        fixed_box_velocities = np.array(fixed_box_velocities)


        pic_number = 13
        pic_idx = 0
        plt.figure(figsize=(pic_number*4, pic_number*4))

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, linear_velocities[:, 0], label='Linear Velocity X')
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity X (m/s)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, linear_velocities[:, 1], label='Linear Velocity Y')
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity Y (m/s)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, linear_velocities[:, 2], label='Linear Velocity Z')
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity Z (m/s)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, angular_velocities[:, 0], label='Angular Velocity Theta', color='orange')
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity Theta (rad/s)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, angular_velocities[:, 1], label='Angular Velocity Phi', color='orange')
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity Phi (rad/s)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, angular_velocities[:, 2], label='Angular Velocity Psi', color='orange')
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity Psi (rad/s)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, displacements[:, 0], label='Displacement X', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Displacement X (m)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, displacements[:, 1], label='Displacement Y', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Displacement Y (m)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, displacements[:, 2], label='Displacement Z', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Displacement Z (m)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, fixed_box_velocities[:, 0], label='Fixed_box_values X', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Fixed_box_values X (m/s)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, fixed_box_velocities[:, 1], label='Fixed_box_values Y', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Fixed_box_values Y (m/s)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, fixed_box_velocities[:, 2], label='Fixed_box_values Z', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Fixed_box_values Z (m/s)')
        plt.legend()
        plt.grid()

        pic_idx += 1
        plt.subplot(pic_number, 1, pic_idx)
        plt.plot(time_steps, action_values, label='Risk value', color='red')
        plt.xlabel('Time Step')
        plt.ylabel('Risk value')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        # plt.show()
        plt.savefig('./demo/cube_trajectory_{}.png'.format(episode_num))

def main():
    xml_path = "./model/universal_robots_ur5e/test_scene_complete.xml"
    traj_path = "../ur5-scripts/traj_20250209.txt"  # Adjust path as needed

    os.makedirs('demo/data/train', exist_ok=True)
    os.makedirs('demo/data/val', exist_ok=True)

    sim = Projectile(xml_path, traj_path, initial_delay=2)
    sim.reset(RANDOM_EPISODE_TMP)
    sim.simulate(sys.argv[1])


if __name__ == "__main__":
    main()