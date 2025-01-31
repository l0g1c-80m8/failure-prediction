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


N_TRAIN_EPISODES = 20
N_VAL_EPISODES = 20
EPISODE_LENGTH = 350  # Number of points in trajectory

# Thresholds for action calculation
DISPLACEMENT_THRESHOLD_HIGH = 0.01
DISPLACEMENT_THRESHOLD_LOW = 0
LINEAR_SPEED_THRESHOLD_HIGH = 0.5
LINEAR_SPEED_THRESHOLD_LOW = 0.05
ANGULAR_SPEED_THRESHOLD_HIGH = 0.8
ANGULAR_SPEED_THRESHOLD_LOW = 0.1

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
        self.init_angular_speed = 1.0  # Angular speed in radians per second
        self.initial_delay = initial_delay  # Delay before starting movement
        self.speed_scale = random.uniform(1.5, 2.0)  # New parameter to control joint speed
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

    def load_trajectory(self):
        with open(self.traj_file, "r") as file:
            trajectory = []
            for line in file:
                print("line", line)
                if ":" in line:
                    key, value = line.strip().split(":")
                    try:
                        trajectory.append(np.array(ast.literal_eval(value.strip())))
                    except Exception as e:
                        print(f"Error parsing line: {line.strip()}. Error: {e}")
                    print("key", key, "value", value)
            print(f"Loaded trajectory: {trajectory}")
            return trajectory

    def reset(self, seed):
        random.seed(seed)


        # Set camera configuration
        self.cam.azimuth = -250 # random.uniform(-225, -315)
        self.cam.distance = 2.5 # random.uniform(2, 3)
        self.cam.elevation = -40 # random.uniform(-50, -30)
        # print("self.cam", self.cam.azimuth, self.cam.distance, self.cam.elevation)

        self.randomize_camera_position()

        # Add random rotation parameters
        # Subtle random rotation parameters
        self.rotation_speed = random.uniform(-0.01, 0.01)  # Slower random speed between 0.2 and 0.5 rad/s
        self.target_angle = random.uniform(-0.03, 0.03)   # Small random angle between -0.3 and 0.3 rad (about ±17 degrees)
        self.reverse_on_target = random.choice([True, False])  # Randomly decide to stop or reverse
        self.reached_target = False  # Track if we've reached target
        # print(f"New rotation parameters - Speed: {self.rotation_speed:.2f} rad/s, Target angle: {self.target_angle:.2f} rad, Reverse: {self.reverse_on_target}")

        self.angular_speed = self.init_angular_speed
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
        self.speed_scale = random.uniform(2.5, 3)  # New parameter to control joint speed
        self.joint_pause = random.uniform(0.2, 0.8)  # Duration of pause between movements

        self.emergency_stop = False  # Flag to trigger the emergency stop
        self.emergency_stop_pos_first_set = False  # Flag to first set the emergency stop pose
        self.current_qpos = None

        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')
        fixed_box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_box')
        self.data.qpos[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3] = [random.uniform(0.35, 0.42), random.uniform(-0.5, -0.35), random.uniform(0.2, 0.35)]

        mj.set_mjcb_control(self.controller)

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

        # Calculate linear and angular speeds of `free_cube`
        cube_lin_vel = self.data.qvel[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3]
        cube_ang_vel = self.data.qvel[self.model.body_jntadr[cube_body_id]+3:self.model.body_jntadr[cube_body_id]+6]


        # Calculate linear and angular speeds of `fixed_box`
        fixed_box_lin_vel = self.data.qvel[self.model.body_jntadr[fixed_box_body_id]:self.model.body_jntadr[fixed_box_body_id]+3]
        # fixed_box_linear_speed = np.linalg.norm(fixed_box_lin_vel)
        
        # cube_linear_speed = np.linalg.norm(cube_lin_vel)
        # cube_angular_speed = np.linalg.norm(cube_ang_vel)

        # Get positions of `free_cube` and `fixed_box`
        cube_pos = self.data.xpos[cube_body_id]
        fixed_box_pos = self.data.xpos[fixed_box_body_id]

        # Calculate the relative displacement
        # relative_displacement = np.linalg.norm(cube_pos - fixed_box_pos)
        # print("!!!!!!!!!!!!!cube_pos", cube_pos[0], cube_pos[1], cube_pos[2])
        # print("!!!!!!!!!!!!!fixed_box_pos", fixed_box_pos[0], fixed_box_pos[1], fixed_box_pos[2])
        # relative_displacement_z = fixed_box_pos[2] - cube_pos[2]
        relative_displacement = fixed_box_pos - cube_pos

        # Optionally print or log these values
        fixed_box_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_box')
        # print("!!!!!!!!!!!!!self.data.xquat[fixed_box_id]", self.data.xquat[fixed_box_id])
        # print(f"Time: {self.data.time:.2f}s | Linear Speed: {cube_linear_speed:.2f} m/s | Angular Speed: {cube_angular_speed:.2f} rad/s | Relative Displacement: {relative_displacement:.2f} m")
        
        # # Calculate the action based on the thresholds and interpolation logic
        # action_value = self.calculate_action(relative_displacement, cube_linear_speed, cube_angular_speed)

        # # Ensure action is stored as a (1,) tensor, not as a scalar
        # action_value = np.asarray([action_value], dtype=np.float32)

        # Just for debugging!!!!!!!!!!!!!!!!!!!!!
        # cube_linear_speed = 0
        # cube_angular_speed = 0


        # print("!!!!!!!!!!!!!relative_displacement", relative_displacement, cube_lin_vel, cube_ang_vel)
        # Combine displacement, linear speed, and angular speed into a state array
        state = np.hstack([relative_displacement, cube_lin_vel, cube_ang_vel])
        # Pad state to 9 values (for compatibility)
        # state_padded = np.pad(state, (0, 9 - len(state)), 'constant')

        return state, fixed_box_lin_vel


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

    def randomize_camera_position(self):
        """
        Improved camera position randomization with proper orientation handling
        and direct camera parameter updates.
        """
        # Get camera id
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, 'top_camera')
        
        # Define ranges for camera randomization
        # Wider ranges for more noticeable variation
        pos_ranges = {
            'x': (-0.3, 0.3),    # Wider range for x offset
            'y': (-0.5, -0.3),    # Wider range for y offset
            'z': (2.0, 2.5),     # Height variation
            'azimuth': (-15, 15),  # Degrees of rotation around vertical axis
            'elevation': (-1, 0), # Degrees of tilt
        }
        
        # Randomly sample camera parameters
        pos_x = random.uniform(*pos_ranges['x'])
        pos_y = random.uniform(*pos_ranges['y'])
        pos_z = random.uniform(*pos_ranges['z'])
        
        # Convert angles to radians for rotation calculation
        azimuth = np.radians(random.uniform(*pos_ranges['azimuth']))
        elevation = np.radians(random.uniform(*pos_ranges['elevation']))
        
        # Update camera position
        self.model.cam_pos[cam_id] = np.array([pos_x, pos_y, pos_z])
        
        # Calculate rotation matrix
        # First rotate around Y axis (elevation)
        Ry = np.array([
            [np.cos(elevation), 0, np.sin(elevation)],
            [0, 1, 0],
            [-np.sin(elevation), 0, np.cos(elevation)]
        ])
        
        # Then rotate around Z axis (azimuth)
        Rz = np.array([
            [np.cos(azimuth), -np.sin(azimuth), 0],
            [np.sin(azimuth), np.cos(azimuth), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations
        R = Rz @ Ry
        
        # Convert rotation matrix to quaternion
        # Using simplified method since we know R is orthogonal
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S
        
        # Update camera quaternion
        self.model.cam_quat[cam_id] = np.array([qw, qx, qy, qz])
        
        # Make sure camera is looking at the scene center
        target_pos = np.array([0, 0, 0])  # Scene center
        cam_pos = self.model.cam_pos[cam_id]
        forward = target_pos - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Update camera orientation to look at target
        self.model.cam_pos[cam_id] = cam_pos
        
        # Force update scene
        mj.mj_forward(self.model, self.data)

    def get_camera_image(self, camera_name):
        """
        Get image from the specified camera with proper handling of randomized parameters.
        
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
        width = 224 #640  # Set to be divisible by 16
        height = 224 #640  # Set to be divisible by 16
        
        # Initialize image array
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create camera instance
        cam = mj.MjvCamera()
        
        # Copy camera configuration from model to camera instance
        cam.type = mj.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
        
        # Ensure camera parameters are properly set
        cam.distance = np.linalg.norm(self.model.cam_pos[cam_id])  # Distance from target
        cam.azimuth = np.degrees(np.arctan2(self.model.cam_pos[cam_id][1], 
                                        self.model.cam_pos[cam_id][0]))  # Rotation around z-axis
        cam.elevation = -np.degrees(np.arctan2(self.model.cam_pos[cam_id][2], 
                                            np.sqrt(np.sum(self.model.cam_pos[cam_id][:2]**2))))  # Angle from xy-plane
        
        # Update scene with camera
        mj.mjv_updateScene(self.model, self.data, self.opt, None, cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        
        # Set up viewport
        viewport = mj.MjrRect(0, 0, width, height)
        
        # Render scene
        mj.mjr_render(viewport, self.scene, self.context)
        
        # Read pixels from framebuffer
        mj.mjr_readPixels(img, None, viewport, self.context)
        
        # Flip image vertically (MuJoCo returns image upside down)
        img = img[::-1, :, :]
        
        # Optional: Add debug information to verify camera parameters
        if False:  # Set to True when debugging
            print(f"Camera {camera_name} parameters:")
            print(f"Position: {self.model.cam_pos[cam_id]}")
            print(f"Quaternion: {self.model.cam_quat[cam_id]}")
            print(f"Distance: {cam.distance}")
            print(f"Azimuth: {cam.azimuth}")
            print(f"Elevation: {cam.elevation}")
        
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

        writer = imageio.get_writer(video_filename, fps=60)
        top_camera_writer = imageio.get_writer(top_camera_video_filename, fps=60, macro_block_size=16)

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

        # Initialize lists to store metrics
        linear_velocities = []
        angular_velocities = []
        displacements = []
        action_values = []
        fixed_box_velocities = []
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

            for overal_step_num in range(EPISODE_LENGTH):

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

                episode_failed_tag = False
                backtracking_steps = 5

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

                    # Update scene and render
                    mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                    mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                    mj.mjr_render(viewport, self.scene, self.context)

                    # Capture the current frame
                    framebuffer = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)
                    mj.mjr_readPixels(framebuffer, None, viewport, self.context)
                    framebuffer = framebuffer[::-1, :, :]  # Reverse the order of rows
                    writer.append_data(framebuffer)

                    # Get top camera frame
                    top_camera_frame = self.get_camera_image('top_camera')
                    top_camera_writer.append_data(top_camera_frame)

                    # swap OpenGL buffers (blocking call due to v-sync)
                    glfw.swap_buffers(self.window)

                    # process pending GUI events, call GLFW callbacks
                    glfw.poll_events()

                    # skip the first cube_drop_time steps because the cube is falling from the sky
                    if step_num >= cube_drop_time:
                        # print("step_num", step_num)

                        state, fixed_box_linear_speed = self.data_collection(cube_body_id, fixed_box_body_id)
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
                            first_failure_time_step = step_num - cube_drop_time + episode_num * (EPISODE_LENGTH - cube_drop_time)

                        if not episode_filled_tag:
                            self.episode.append({
                            'image': top_camera_frame,
                            # 'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
                            'state': np.asarray(state, dtype=np.float32),  # Save the padded state
                            'action': np.asarray([action_value], dtype=np.float32),  # Ensure action is a tensor of shape (1,)
                            'language_instruction': 'dummy instruction',
                                })
                            # For plot     
                            print("action_value!!!!!!!!!!!!step_num - cube_drop_time", step_num - cube_drop_time + episode_num * (EPISODE_LENGTH - cube_drop_time), action_value)
                            displacements.append([state[0], state[1], state[2]])  # Displacement
                            linear_velocities.append([state[3], state[4], state[5]])  # Linear velocity
                            angular_velocities.append([state[6], state[7], state[8]])  # Angular velocity
                            action_values.append(action_value) # Action value
                            fixed_box_velocities.append(fixed_box_linear_speed)
                episode_filled_tag = True
                print("len(self.episode)", len(self.episode), "episode_filled_tag", episode_filled_tag, "episode_failed_tag", episode_failed_tag, "failure_time_step", failure_time_step)

                failure_time_step -= backtracking_steps
                # cube_drop_time is for avoid counting in the initial falling of the cube
                failure_time_step_trim = failure_time_step - cube_drop_time  + episode_num * (EPISODE_LENGTH - cube_drop_time)
                # print("episode_num", episode_num)
                # print("N_EPISODES", N_EPISODES)
                print("first_failure_time_step", first_failure_time_step)
                print("failure_time_step_trim", failure_time_step_trim)
                print("self.episode[failure_time_step_trim-1]['action']", self.episode[failure_time_step_trim-1]['action'])
                print("self.episode[failure_time_step_trim]['action']", self.episode[failure_time_step_trim]['action'])
                if episode_failed_tag:
                    self.episode[failure_time_step_trim]['action'] = np.asarray([1.0], dtype=np.float32)
                    action_values[failure_time_step_trim] = 1
                    for idx in range(failure_time_step_trim, first_failure_time_step):
                        # print("idx", idx)
                        self.episode[idx]['action'] = np.asarray([1.0], dtype=np.float32)
                        action_values[idx] = 1  
                elif episode_filled_tag and first_failure_time_step != -1: # if not failed, the time step to apply e-stop is safe, backtracking over
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
            
            self.validate_episode(self.episode)
            if dataset == "train":
                print("Generating train examples...")
                np.save(f'demo/data/train/episode_{episode_num}.npy', self.episode)
            elif dataset == "val":
                print("Generating val examples...")
                np.save(f'demo/data/val/episode_{episode_num}.npy', self.episode)

        # Plot after simulation


        writer.close()
        top_camera_writer.close()
        glfw.terminate()
        # self.save_trajectory()
        self.plot_metrics(linear_velocities, angular_velocities, displacements, action_values, fixed_box_velocities)


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

    def plot_metrics(self, linear_velocities, angular_velocities, displacements, action_values, fixed_box_velocities):
        """
        Plot linear velocity, angular velocity, and displacement over time.
        """
        time_steps = range(len(linear_velocities))
        # print("range", range(len(linear_velocities)), range(len(angular_velocities)), range(len(displacements)), range(len(action_values)))  # 800

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
        plt.savefig('./demo/cube_trajectory.png')

def main():
    xml_path = "./model/universal_robots_ur5e/test_scene.xml"
    traj_path = "../ur5-scripts/traj.txt"  # Adjust path as needed

    os.makedirs('demo/data/train', exist_ok=True)
    os.makedirs('demo/data/val', exist_ok=True)

    sim = Projectile(xml_path, traj_path, initial_delay=2)
    sim.reset(RANDOM_EPISODE_TMP)
    sim.simulate(sys.argv[1])


if __name__ == "__main__":
    main()