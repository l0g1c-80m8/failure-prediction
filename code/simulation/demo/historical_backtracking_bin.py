import os
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
import sys
import time
import argparse
from datetime import datetime

from mujoco_base import MuJoCoBase
import imageio
import random

import tqdm
# from scipy.interpolate import CubicSpline
import ast

import cv2

from common_functions import (flat_interpolation, process_simulation_camera_frame,
                        calculate_failure_phase, plot_raw_metrics,
                        read_config
                        )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Update simulation config JSON')
    parser.add_argument('--config', default='simulation_config.json', help='Path to the simulation_config.json file to modify')
    return parser.parse_args()

class Projectile(MuJoCoBase):
    def __init__(self, config, ramdom_episode):
        super().__init__(config.get('simulation_related', {}).get('xml_path', 'N/A'))
        self.config = config
        self.ramdom_episode = ramdom_episode
        
        # Simulation parameters
        self.initial_delay = self.config.get('simulation_related', {}).get('initial_delay', 'N/A')  # Delay before starting movement
        self.display_camera = self.config.get('camera_related', {}).get('display_camera', 'N/A')
        self.episode = [] # To store the episode data
        self.min_contour_area = self.config.get('simulation_related', {}).get('min_contour_area', 'N/A')
        # self.high_threshold_step = 0  # Step when displacement > DISPLACEMENT_THRESHOLD_HIGH
        # self.is_high_threshold_step_set = False

        # Trajectory loading
        self.traj_file = self.config.get('simulation_related', {}).get('traj_path', 'N/A')
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
        self.enable_cameras = self.config.get('camera_related', {}).get('enable_cameras', 'N/A')

        self.dataset_type = self.config.get('simulation_related', {}).get('dataset_type', "train")
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
            lines = file.readlines()
    
            # Randomize the order of lines if requested
            if self.config.get('input_trajectory_related', {}).get('shuffle', 'N/A'):
                random.shuffle(lines)

            for line in lines:

                try:
                    joint_positions = np.array(ast.literal_eval(line.strip()))
                    
                    # Apply randomization if enabled
                    if randomize:
                        # Define randomization ranges for each joint (in radians)
                        # These values can be adjusted based on how much variation you want
                        joint1_range = self.config.get('input_trajectory_related', {}).get('joint_1', 'N/A')
                        joint2_range = self.config.get('input_trajectory_related', {}).get('joint_2', 'N/A')
                        joint3_range = self.config.get('input_trajectory_related', {}).get('joint_3', 'N/A')
                        joint4_range = self.config.get('input_trajectory_related', {}).get('joint_4', 'N/A')
                        joint5_range = self.config.get('input_trajectory_related', {}).get('joint_5', 'N/A')
                        joint6_range = self.config.get('input_trajectory_related', {}).get('joint_6', 'N/A')
                        variation_ranges = [
                            (joint1_range[0], joint1_range[1]),  # Joint 1: ±0.05 radians
                            (joint2_range[0], joint2_range[1]),  # Joint 2: ±0.03 radians
                            (joint3_range[0], joint3_range[1]),  # Joint 3: ±0.04 radians
                            (joint4_range[0], joint4_range[1]),  # Joint 4: ±0.03 radians
                            (joint5_range[0], joint5_range[1]),  # Joint 5: ±0.05 radians
                            (joint6_range[0], joint6_range[1]),  # Joint 6: ±0.02 radians
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
        self.cam.azimuth = self.config.get('camera_related', {}).get('azimuth', 'N/A') # -216 random.uniform(-225, -315)
        self.cam.distance = self.config.get('camera_related', {}).get('distance', 'N/A') # random.uniform(2, 3)
        self.cam.elevation = self.config.get('camera_related', {}).get('elevation', 'N/A') # random.uniform(-16, -30)
        # print("self.cam", self.cam.azimuth, self.cam.distance, self.cam.elevation)

        # Randomize camera positions
        self.randomize_camera_position('top_camera')
        self.randomize_camera_position('front_camera')

        # Set random rotation parameters
        self.rotation_speed = random.uniform(self.config.get('panel_related', {}).get('rotation_speed', 'N/A')[0], self.config.get('panel_related', {}).get('rotation_speed', 'N/A')[1])  # Slower random speed between 0.2 and 0.5 rad/s
        self.target_angle = random.uniform(self.config.get('panel_related', {}).get('target_angle', 'N/A')[0], self.config.get('panel_related', {}).get('target_angle', 'N/A')[1])   # Small random angle between -0.3 and 0.3 rad (about ±17 degrees)
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
        self.speed_scale = random.uniform(self.config.get('robot_related', {}).get('speed_scale', 'N/A')[0], self.config.get('robot_related', {}).get('speed_scale', 'N/A')[1])  # New parameter to control joint speed
        self.joint_pause = random.uniform(self.config.get('simulation_related', {}).get('joint_pause', 'N/A')[0], self.config.get('simulation_related', {}).get('joint_pause', 'N/A')[1])  # Duration of pause between movements

        # Emergency stop settings
        self.emergency_stop = False  # Flag to trigger the emergency stop
        self.emergency_stop_pos_first_set = False  # Flag to first set the emergency stop pose
        self.current_qpos = None

        # Randomize environment
        self.randomize_floor()
        self.randomize_scene_colors("panel_collision", "object_collision")
        self.randomize_object('object_body', 'object_collision')
        mj.mj_forward(self.model, self.data)

        self.trajectory = self.load_trajectory()

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
        texture_scale = random.uniform(self.config.get('background_related', {}).get('texture_scale', 'N/A')[0], self.config.get('background_related', {}).get('texture_scale', 'N/A')[1])
        self.model.mat_texrepeat[material_id] = [texture_scale, texture_scale]
        
        # Random reflectance
        self.model.mat_reflectance[material_id] = random.uniform(self.config.get('background_related', {}).get('mat_reflectance', 'N/A')[0], self.config.get('background_related', {}).get('mat_reflectance', 'N/A')[1])
        
        # Optional: Add some random noise to the floor color
        rgb_noise = np.random.uniform(self.config.get('background_related', {}).get('rgb_noise', 'N/A')[0], self.config.get('background_related', {}).get('rgb_noise', 'N/A')[1], self.config.get('background_related', {}).get('rgb_noise', 'N/A')[2])
        self.model.mat_rgba[material_id][:3] += rgb_noise
        self.model.mat_rgba[material_id][:3] = np.clip(self.model.mat_rgba[material_id][:3], 0, 1)

    def randomize_object(self, object_body_id, object_geom_id):
        """Randomize the object's size, mass, friction, and other physical properties."""
        # Debug: print all mesh names in the model
        # for i in range(self.model.nmesh):
        #     mesh_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_MESH, i)
        #     print(f"Mesh ID {i}: {mesh_name}")

        # Get object body and geom IDs
        object_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, object_body_id)
        object_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, object_geom_id)
        current_object_name = self.config.get('object_related', {}).get('current_object', 'N/A')
        object_mesh_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_MESH, current_object_name)
        print("object_mesh_id", object_mesh_id)

        fixed_panel_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_panel')
        fixed_panel_pos = self.data.xpos[fixed_panel_body_id]
        # print("fixed_panel_pos", fixed_panel_pos)
        
        # Randomize object size (within reasonable bounds)
        print("object_geom_id", object_geom_id)
        base_size = self.config.get('object_related', {}).get(current_object_name, {}).get('base_size', 'N/A') # Original size
        # print("base_size", base_size)
        size_variation = random.uniform(self.config.get('object_related', {}).get(current_object_name, {}).get('size_variation', 'N/A')[0], self.config.get('object_related', {}).get(current_object_name, {}).get('size_variation', 'N/A')[1])  # n% variation
        new_size = base_size * size_variation
        # print("new_size", new_size)
        self.model.geom_size[object_geom_id] = [new_size, new_size, new_size]
        if object_mesh_id >= 0:
            # Scale the mesh
            self.model.mesh_scale[object_mesh_id] = [new_size, new_size, new_size]
        
        # Randomize mass (scaled with size)
        base_mass = self.config.get('object_related', {}).get(current_object_name, {}).get('base_mass', 'N/A') # Original mass
        mass_variation = random.uniform(self.config.get('object_related', {}).get(current_object_name, {}).get('mass_variation', 'N/A')[0], self.config.get('object_related', {}).get(current_object_name, {}).get('mass_variation', 'N/A')[1])  # ±20% variation
        new_mass = base_mass * mass_variation  # base_mass * size_variation * mass_variation  # Scale mass with size
        self.model.body_mass[object_body_id] = new_mass
        
        # Adjust inertia based on new mass and size
        new_inertia = (new_mass * new_size**2) / 6  # Simple panel inertia approximation
        self.model.body_inertia[object_body_id] = [new_inertia, new_inertia, new_inertia]
        
        # Randomize friction properties
        friction_variation = random.uniform(self.config.get('object_related', {}).get(current_object_name, {}).get('friction_variation', 'N/A')[0], self.config.get('object_related', {}).get(current_object_name, {}).get('friction_variation', 'N/A')[1])  # Base friction is 0.2
        # Find the contact pair involving the sliding object
        for i in range(self.model.npair):
            if (self.model.pair_geom1[i] == object_geom_id or 
                self.model.pair_geom2[i] == object_geom_id):
                self.model.pair_friction[i, 0] = friction_variation  # Sliding friction
                self.model.pair_friction[i, 1] = friction_variation * 2.5  # Rolling friction
                self.model.pair_friction[i, 2] = friction_variation * 0.005  # Torsional friction
        
        # Randomize initial position (within reasonable bounds)
        x_offset = self.config.get('object_related', {}).get(current_object_name, {}).get('x_offset', 'N/A')
        y_offset = self.config.get('object_related', {}).get(current_object_name, {}).get('y_offset', 'N/A')
        z_offset = self.config.get('object_related', {}).get(current_object_name, {}).get('z_offset', 'N/A')
        x_pos = random.uniform(fixed_panel_pos[0]+x_offset[0], fixed_panel_pos[0]+x_offset[1])
        y_pos = random.uniform(fixed_panel_pos[1]+y_offset[0], fixed_panel_pos[1]+y_offset[1])
        z_pos = random.uniform(fixed_panel_pos[2]+z_offset[0], fixed_panel_pos[2]+z_offset[1])
        self.data.qpos[self.model.body_jntadr[object_body_id]:self.model.body_jntadr[object_body_id]+3] = [x_pos, y_pos, z_pos]
        
        # Randomize initial orientation (uncomment when needed)
        quat = self.config.get('object_related', {}).get(current_object_name, {}).get('qpos', 'N/A') #[random.uniform(-1, 1) for _ in range(4)]
        quat = quat / np.linalg.norm(quat)  # Normalize quaternion
        self.data.qpos[self.model.body_jntadr[object_body_id]+3:self.model.body_jntadr[object_body_id]+7] = quat

    def randomize_scene_colors(self, panel_geom_name, object_geom_name):
        """Randomize colors for fixed panel and free object"""
        # Zeyu: need to be revised, since combined object and the panel, should be decoupled
        # Get geom IDs
        fixed_panel_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, panel_geom_name)
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
        x_offset = self.config.get('camera_related', {}).get('x_offset', 'N/A')
        y_offset = self.config.get('camera_related', {}).get('y_offset', 'N/A')
        z_offset = self.config.get('camera_related', {}).get('z_offset', 'N/A')
        pos_ranges = {
            'x': (self.cam_position_init[cam_id][0]-x_offset[0], self.cam_position_init[cam_id][0]+x_offset[1]),    # Wider range for x offset
            'y': (self.cam_position_init[cam_id][1]-y_offset[0], self.cam_position_init[cam_id][1]+y_offset[1]),    # Wider range for y offset
            'z': (self.cam_position_init[cam_id][2]-z_offset[0], self.cam_position_init[cam_id][2]+z_offset[1]),     # Height variation
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
        angle_rad_threshold = self.config.get('panel_related', {}).get('angle_rad_threshold', 'N/A')  # About 0.6 degrees threshold
        
        if abs(angle_diff) < angle_rad_threshold:
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

    def get_camera_image(self, camera_name):
        """
        Get images from the specified camera - processed (panel/object separate) and non-processed (full scene).
        
        Args:
            camera_name (str): Name of the camera to capture from
            
        Returns:
            tuple: Four RGB image arrays (panel_img, object_img, full_img_rgb, full_img_depth) 
                each of shape (height, width, 3) or (height, width) for depth
        """
        # Skip if cameras are disabled
        if not self.enable_cameras:
            return None, None, None, None
        
        # Get camera id
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")
        
        # Get image dimensions
        image_size = self.config.get('camera_related', {}).get('camera_image_size', 'N/A')
        
        # Initialize image arrays
        panel_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        object_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        full_img_rgb = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        full_img_depth = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
        
        # Create camera instance
        cam = mj.MjvCamera()
        # Copy camera configuration from model to camera instance
        cam.type = mj.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
        
        # Save current geom groups visibility
        original_geomgroup = self.opt.geomgroup.copy()
        
        if camera_name in ['top_camera', 'front_camera']:
            # Set up viewport
            viewport = mj.MjrRect(0, 0, image_size[1], image_size[0])
            
            # First render - full scene (non-processed image)
            # Restore original visibility settings to show everything
            self.opt.geomgroup[:] = original_geomgroup
            
            # Update and render scene for full view
            mj.mjv_updateScene(self.model, self.data, self.opt, None, cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            
            # Read pixels for full RGB image
            mj.mjr_readPixels(full_img_rgb, full_img_depth, viewport, self.context)
            
            # Now do the processed renders as before
            # First render - fixed panel only
            self.opt.geomgroup[:] = 0  # Hide all groups
            self.opt.geomgroup[1] = 1  # Show only fixed panel
            
            # Update and render scene for fixed panel
            mj.mjv_updateScene(self.model, self.data, self.opt, None, cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            
            # Read pixels for fixed panel image
            mj.mjr_readPixels(panel_img, None, viewport, self.context)
            
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
            
            return panel_img, object_img, full_img_rgb, full_img_depth
        
        return None, None, None, None  # Return None for all images if camera name not recognized

    def simulate(self):
        video_dir = './demo'
        os.makedirs(video_dir, exist_ok=True)
        if self.enable_cameras:
            top_camera_video_filename = os.path.join(video_dir, 'top_camera_video.mp4')
            front_camera_video_filename = os.path.join(video_dir, 'front_camera_video.mp4')

            # writer = imageio.get_writer(video_filename, fps=60)
            fps = self.config.get('simulation_related', {}).get('video_writer_fps', 'N/A')
            top_panel_writer = imageio.get_writer(top_camera_video_filename, fps=fps, macro_block_size=16)
            top_object_writer = imageio.get_writer(top_camera_video_filename, fps=fps, macro_block_size=16)
            front_panel_writer = imageio.get_writer(front_camera_video_filename, fps=fps, macro_block_size=16)
            front_object_writer = imageio.get_writer(top_camera_video_filename, fps=fps, macro_block_size=16)

        # Create data directories
        os.makedirs(f"{self.config.get('simulation_related', {}).get('save_path', 'N/A')}/data/{self.dataset_type}_raw", exist_ok=True)
        save_path = f"{self.config.get('simulation_related', {}).get('save_path', 'N/A')}/data/{self.dataset_type}_raw"

        # Get object IDs
        object_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'object_body')
        fixed_panel_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_panel')

        # while not glfw.window_should_close(self.window):
        if self.dataset_type == "train":
            n_episodes = self.config.get('simulation_related', {}).get('trajectories', {}).get('n_train_episodes', 'N/A')
        elif self.dataset_type == "val":
            n_episodes = self.config.get('simulation_related', {}).get('trajectories', {}).get('n_val_episodes', 'N/A')
        else:
            print(f"Unknown dataset type: {self.dataset_type}")
            return

        episode_length = self.config.get('simulation_related', {}).get('trajectories', {}).get('episode_length', 'N/A')

        for episode_num in range(n_episodes):
            
            random_seed_tmp = random.randint(0, episode_length)
            print(f"Episode {episode_num+1}/{n_episodes}, random seed: {random_seed_tmp}")

            # Initialize episode variables
            failure_time_step = -1
            first_failure_time_step = -1
            object_drop_time = self.config.get('simulation_related', {}).get('object_drop_time', 'N/A')  # Steps to ignore while object is initially falling
            episode_filled_tag = False

            self.episode = []  # Reset episode data for each new episode
        
            # Initialize dummy contours for use when cameras are disabled
            dummy_contour = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)

            # Main simulation loop for backtracking
            for overal_step_num in range(episode_length):
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
                
                self.reset(self.ramdom_episode if episode_num == 0 else random_seed_tmp)  # Reset simulation

                # Inner simulation loop
                for step_num in range(episode_length):
                    simstart = self.data.time
                    while (self.data.time - simstart < 1.0/60.0):
                        # Step simulation environment
                        mj.mj_step(self.model, self.data)

                        # Record the object's position
                        current_object_name = self.config.get('object_related', {}).get('current_object', 'N/A')
                        object_pos = self.data.qpos[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, current_object_name)]

                    # get framebuffer viewport
                    viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
                    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
                    # Update scene
                    mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                    mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                    # Render scene
                    mj.mjr_render(viewport, self.scene, self.context)
                    
                    # Create an array to store the rendered image
                    viewport_img = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)

                    # Read pixels from the framebuffer into the array
                    mj.mjr_readPixels(viewport_img, None, viewport, self.context)

                    # Convert from OpenGL format (bottom-origin) to OpenCV format (top-origin)
                    viewport_img = viewport_img[::-1, :, :]

                    # Convert from RGB to BGR for OpenCV
                    viewport_img_bgr = cv2.cvtColor(viewport_img, cv2.COLOR_RGB2BGR)

                    # Initialize contour variables with default values
                    top_panel_contour = dummy_contour
                    top_object_contour = dummy_contour
                    front_panel_contour = dummy_contour
                    front_object_contour = dummy_contour

                    # Camera operations only if enabled
                    if self.enable_cameras:
                        # Get top camera frame
                        top_panel_frame, top_object_frame, full_top_frame_rgb, full_top_frame_depth = self.get_camera_image('top_camera')
                        # top_camera_frame = top_camera_frame[::-1, :, :]
                        if top_panel_frame is not None and top_object_frame is not None:
                            # Process panel frame
                            # print("top_panel_frame", type(top_panel_frame))  # <class 'numpy.ndarray'>
                            top_panel_writer.append_data(top_panel_frame)
                            top_panel_view = cv2.cvtColor(top_panel_frame, cv2.COLOR_RGB2BGR)
                            top_panel_mask, top_panel_contour, top_panel_filtered = process_simulation_camera_frame(top_panel_view, min_contour_area=self.min_contour_area)
                            
                            # Process object frame
                            top_object_writer.append_data(top_object_frame)
                            top_object_view = cv2.cvtColor(top_object_frame, cv2.COLOR_RGB2BGR)
                            top_object_mask, top_object_contour, top_object_filtered = process_simulation_camera_frame(top_object_view, min_contour_area=self.min_contour_area)

                        # Get front camera frames
                        front_panel_frame, front_object_frame, full_front_frame_rgb, full_front_frame_depth = self.get_camera_image('front_camera')
                        # Add this line to rotate the full front frame before storing it in the episode
                        if full_front_frame_rgb is not None:
                            full_front_frame_rgb = cv2.rotate(full_front_frame_rgb, cv2.ROTATE_90_CLOCKWISE)
                        if front_panel_frame is not None and front_object_frame is not None:
                            # Rotate and process panel frame
                            front_panel_frame = cv2.rotate(front_panel_frame, cv2.ROTATE_90_CLOCKWISE)
                            front_panel_writer.append_data(front_panel_frame)
                            front_panel_view = cv2.cvtColor(front_panel_frame, cv2.COLOR_RGB2BGR)
                            front_panel_mask, front_panel_contour, front_panel_filtered = process_simulation_camera_frame(front_panel_view, min_contour_area=self.min_contour_area)
                            
                            # Rotate and process object frame
                            front_object_frame = cv2.rotate(front_object_frame, cv2.ROTATE_90_CLOCKWISE)
                            front_object_writer.append_data(front_object_frame)
                            front_object_view = cv2.cvtColor(front_object_frame, cv2.COLOR_RGB2BGR)
                        front_object_mask, front_object_contour, front_object_filtered = process_simulation_camera_frame(front_object_view, min_contour_area=self.min_contour_area)


                    # Display images
                    if self.display_camera:
                        if self.display_camera:
                            cv2.imshow('MuJoCo Main View', viewport_img_bgr)
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
                        # Check for empty or invalid contours
                        if (np.asarray(top_object_contour, dtype=np.float32).shape[0] == 0 or np.asarray(top_panel_contour, dtype=np.float32).shape[0] == 0 or 
                            np.asarray(front_object_contour, dtype=np.float32).shape[0] == 0 or np.asarray(front_panel_contour, dtype=np.float32).shape[0] == 0):
                            print(f"Warning: Empty contours detected at step_num {step_num}, skipping processing")
                            # print("np.asarray(top_panel_contour, dtype=np.float32).shape", np.asarray(top_panel_contour, dtype=np.float32).shape)
                            # print("np.asarray(top_object_contour, dtype=np.float32).shape", np.asarray(top_object_contour, dtype=np.float32).shape)
                            # print("np.asarray(front_panel_contour, dtype=np.float32).shape", np.asarray(front_panel_contour, dtype=np.float32).shape)
                            # print("np.asarray(front_object_contour, dtype=np.float32).shape", np.asarray(front_object_contour, dtype=np.float32).shape)
                            # sys.exit()
                            continue
                        

                        # print("step_num", step_num)
                        end_effector_pos, transform_data_3D = self.data_collection(object_body_id, fixed_panel_body_id)
                        displacement = transform_data_3D['relative_position']

                        # Get current positions
                        object_pos = self.data.xpos[object_body_id].copy()
                        panel_pos = self.data.xpos[fixed_panel_body_id].copy()

                        failure_phase_value = calculate_failure_phase(displacement, object_pos, panel_pos)

                        # Historical backtracking for backtracking_steps time steps
                        # print("episode_filled_tag!!!!!!", episode_filled_tag)
                        if episode_filled_tag:
                            if step_num == (failure_time_step - backtracking_steps):
                                self.activate_emergency_stop()  # manually active the e-stop, robot pose hold
                                continue
                            if step_num > (failure_time_step - backtracking_steps) and failure_phase_value == 1.0: # after e-stop, still failed enven e-stop in advance
                                episode_failed_tag = True
                            continue
                        else:
                            # Once if failure_phase_value == 1 for the first time, it will always == 1.
                            # If failed, apply e-stop
                            # print("failure_time_step!!!!!!", failure_time_step)
                            if failure_phase_value == 1.0 and not self.emergency_stop and not self.emergency_stop_pos_first_set:
                                # print("!!!!!!!!self.emergency_stop", self.emergency_stop)
                                self.activate_emergency_stop()  # automatically active the e-stop, robot pose hold
                                episode_failed_tag = True
                                failure_time_step = step_num
                                # object_drop_time is for avoid counting in the initial falling of the object
                                first_failure_time_step = step_num - object_drop_time # + episode_num * (self.config.get('trajectories', {}).get('episode_length', 'N/A') - object_drop_time)

                            self.episode.append({
                            'full_top_frame_rgb': full_top_frame_rgb,
                            'full_front_frame_rgb': full_front_frame_rgb,
                            # 'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
                            'time_step': np.asarray(step_num, dtype=np.float32),
                            'object_top_contour': np.asarray(top_object_contour, dtype=np.float32),
                            'object_front_contour': np.asarray(front_object_contour, dtype=np.float32),
                            'gripper_top_contour': np.asarray(top_panel_contour, dtype=np.float32),
                            'gripper_front_contour': np.asarray(front_panel_contour, dtype=np.float32),
                            'end_effector_pos': np.asarray(end_effector_pos, dtype=np.float32),
                            'failure_phase_value': np.asarray([failure_phase_value], dtype=np.float32),  # Ensure action is a tensor of shape (1,)
                            'risk': np.asarray([0.0], dtype=np.float32)  # Placeholder for risk prediction
                            # 'language_instruction': 'dummy instruction',
                                })
                            # For plot     
                            print("failure_phase_value!!!!!!!!!!!!step_num - object_drop_time", step_num - object_drop_time, failure_phase_value) # + episode_num * (self.config.get('trajectories', {}).get('episode_length', 'N/A') - object_drop_time), failure_phase_value)
                            
                # After a round of simulation: 
                episode_filled_tag = True
                print("len(self.episode)", len(self.episode), "episode_filled_tag", episode_filled_tag, "episode_failed_tag", episode_failed_tag, "failure_time_step", failure_time_step)

                failure_time_step -= backtracking_steps
                # object_drop_time is for avoid counting in the initial falling of the object
                failure_time_step_trim = failure_time_step - object_drop_time #  + episode_num * (self.config.get('trajectories', {}).get('episode_length', 'N/A') - object_drop_time)
                # print("episode_num", episode_num)
                # print("n_episodes", n_episodes)
                print("first_failure_time_step", first_failure_time_step)
                print("failure_time_step_trim", failure_time_step_trim)

                # If no failure occurred at all during the first attempt
                if first_failure_time_step <= -1:
                    print("No failure occurred at all")
                    break
                
                # There is failure occurred
                if episode_failed_tag:
                    # The initial condition will lead to failure itself
                    if failure_time_step_trim <= -1:
                        break
                    else:
                        continue
                # After a few rounds of backtracking (no failure after applying e-stop)
                elif first_failure_time_step != -1: # if not failed after backtracking, the time step to apply e-stop is safe, backtracking over
                    interpolated_values = flat_interpolation(first_failure_time_step, failure_time_step_trim)
                    # Update the 'failure_phase_value' key for the dictionaries between i and k
                    for idx, value in enumerate(interpolated_values, start=failure_time_step_trim):
                        # print("value", value)
                        self.episode[idx]['failure_phase_value'] = np.asarray([value], dtype=np.float32)
                    break
                
            
            # Plot after simulation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_raw_metrics(self.episode, episode_num, self.dataset_type, save_path, current_object_name, timestamp)

            if self.config.get('simulation_related', {}).get('save_data', 'N/A'):
                print(f"Generating {self.dataset_type} raw examples...")
                np.save(f"{save_path}/episode{episode_num}_{current_object_name}_{self.dataset_type}_{timestamp}_raw", self.episode)
                

        # writer.close()
        if self.display_camera:
            top_panel_writer.close()
            top_object_writer.close()
            front_panel_writer.close()
            front_object_writer.close()
        glfw.terminate()

if __name__ == "__main__":
    args = parse_arguments()
    config = read_config(args.config)
    if not config:  # Check if config is not None before trying to access values
        print("Could not load configuration. Using default values.")
    
    ramdom_episode = random.randint(0, config.get('simulation_related', {}).get('trajectories', {}).get('episode_length', 'N/A')) # 67 86 #  109 282 731 344

    sim = Projectile(config, ramdom_episode)
    sim.reset(ramdom_episode)
    sim.simulate()