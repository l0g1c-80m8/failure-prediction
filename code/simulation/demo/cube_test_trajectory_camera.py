import os
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
import sys

from mujoco_base import MuJoCoBase
import imageio
import random

import tqdm
from scipy.interpolate import CubicSpline
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


N_TRAIN_EPISODES = 1
N_VAL_EPISODES = 1
EPISODE_LENGTH = 10  # Number of points in trajectory

# Thresholds for action calculation
DISPLACEMENT_THRESHOLD_HIGH = 0.12
DISPLACEMENT_THRESHOLD_LOW = 0.05
LINEAR_SPEED_THRESHOLD_HIGH = 1.0
LINEAR_SPEED_THRESHOLD_LOW = 0.05
ANGULAR_SPEED_THRESHOLD_HIGH = 0.8
ANGULAR_SPEED_THRESHOLD_LOW = 0.1


class Projectile(MuJoCoBase):
    def __init__(self, xml_path, traj_file, output_dir, initial_delay=3.0):
        super().__init__(xml_path)
        self.init_angular_speed = 1.0  # Angular speed in radians per second
        self.initial_delay = initial_delay  # Delay before starting movement
        self.speed_scale = random.uniform(0.5, 2.0)  # New parameter to control joint speed
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
        
        self.output_dir = output_dir

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

    def reset(self):

        # Set camera configuration
        self.cam.azimuth = -250 # random.uniform(-225, -315)
        self.cam.distance = 2.5 # random.uniform(2, 3)
        self.cam.elevation = -40 # random.uniform(-50, -30)
        # print("self.cam", self.cam.azimuth, self.cam.distance, self.cam.elevation)

        self.randomize_camera_position()

        self.angular_speed = self.init_angular_speed
        # self.human_intervene = False  # New attribute to track cube drop
        # self.intervene_step = 0  # Step when human intervention occurs
        # self.is_intervene_step_set = False
        self.high_threshold_step = 0  # Step when displacement > DISPLACEMENT_THRESHOLD_HIGH
        self.is_high_threshold_step_set = False

        # Initialize start time
        self.start_time = None
        # Clear position data
        self.positions = []
        self.episode = []  # Reset episode data for each new episode

        self.current_step = 0
        self.current_target = None
        self.next_target = None
        self.transition_start_time = None
        self.speed_scale = random.uniform(0.5, 2.0)  # New parameter to control joint speed
        self.joint_pause = random.uniform(0.2, 0.8)  # Duration of pause between movements

        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')
        fixed_box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_box')
        self.data.qpos[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3] = [0.4, -0.45, 0.35]

        mj.set_mjcb_control(self.controller)


    def controller(self, model, data):
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

        elapsed_time = data.time - self.start_time
        if elapsed_time < self.initial_delay:
            return

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
        
        linear_speed = np.linalg.norm(cube_lin_vel)
        angular_speed = np.linalg.norm(cube_ang_vel)

        # Get positions of `free_cube` and `fixed_box`
        cube_pos = self.data.xpos[cube_body_id]
        fixed_box_pos = self.data.xpos[fixed_box_body_id]

        # Calculate the relative displacement
        relative_displacement = np.linalg.norm(cube_pos - fixed_box_pos)

        # Optionally print or log these values
        print(f"Time: {self.data.time:.2f}s | Linear Speed: {linear_speed:.2f} m/s | Angular Speed: {angular_speed:.2f} rad/s | Relative Displacement: {relative_displacement:.2f} m | ", end="")
        
        # # Calculate the action based on the thresholds and interpolation logic
        # action_value = self.calculate_action(relative_displacement, linear_speed, angular_speed)

        # # Ensure action is stored as a (1,) tensor, not as a scalar
        # action_value = np.asarray([action_value], dtype=np.float32)

        # Combine displacement, linear speed, and angular speed into a state array
        state = np.hstack([relative_displacement, [linear_speed, angular_speed]])
        # Pad state to 9 values (for compatibility)
        state_padded = np.pad(state, (0, 9 - len(state)), 'constant')

        return state_padded


    # Function to calculate action value based on displacement, linear speed, and angular speed
    def calculate_action(self, displacement, linear_speed, angular_speed):
        # Check if all values are in the lower range
        if np.linalg.norm(displacement) < DISPLACEMENT_THRESHOLD_LOW and \
        linear_speed < LINEAR_SPEED_THRESHOLD_LOW and \
        abs(angular_speed) < ANGULAR_SPEED_THRESHOLD_LOW:
            return 0  # Set action to 0

        # Check if any value exceeds the higher thresholds
        if np.linalg.norm(displacement) > DISPLACEMENT_THRESHOLD_HIGH or \
        linear_speed > LINEAR_SPEED_THRESHOLD_HIGH or \
        abs(angular_speed) > ANGULAR_SPEED_THRESHOLD_HIGH:
            return 1  # Set action to 1

        # If the values fall between thresholds, interpolate a reasonable action
        # Normalize between 0 and 1 based on distance from the lower to upper threshold
        displacement_factor = (np.linalg.norm(displacement) - DISPLACEMENT_THRESHOLD_LOW) / \
                            (DISPLACEMENT_THRESHOLD_HIGH - DISPLACEMENT_THRESHOLD_LOW)
        linear_speed_factor = (linear_speed - LINEAR_SPEED_THRESHOLD_LOW) / \
                            (LINEAR_SPEED_THRESHOLD_HIGH - LINEAR_SPEED_THRESHOLD_LOW)
        angular_speed_factor = (abs(angular_speed) - ANGULAR_SPEED_THRESHOLD_LOW) / \
                            (ANGULAR_SPEED_THRESHOLD_HIGH - ANGULAR_SPEED_THRESHOLD_LOW)
        print(np.linalg.norm(displacement), linear_speed_factor, abs(angular_speed))
        # Combine the factors with an average, assuming equal importance
        action_value = (displacement_factor + linear_speed_factor + angular_speed_factor) / 3.0
        print("Action:", f"{round(action_value, 4)}...")
        return action_value

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
        for joint_idx, position in enumerate(start_pose):
            self.data.qpos[joint_idx] = position  # Directly set the joint positions
        mj.mj_forward(self.model, self.data)  # Forward dynamics to update the simulation state

        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')
        fixed_box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_box')
        self.data.qpos[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3] = [0.4, 0.45, 0.6]


        # while not glfw.window_should_close(self.window):
        if dataset == "train":
            N_EPISODES = N_TRAIN_EPISODES
        elif dataset == "val":
            N_EPISODES = N_VAL_EPISODES

        # Initialize lists to store metrics
        linear_velocities = []
        angular_velocities = []
        displacements = []
        action_values = []


        for episode_num in range(N_EPISODES):

            # Load the 'home' keyframe for initial position
            keyframe_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, 'home')
            if keyframe_id >= 0:
                mj.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
            # Reset to start pose again at the beginning of each episode
            for joint_idx, position in enumerate(start_pose):
                self.data.qpos[joint_idx] = position
            mj.mj_forward(self.model, self.data)
            
            self.reset()  # Reset simulation

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


                state_padded = self.data_collection(cube_body_id, fixed_box_body_id)
                displacements.append(state_padded[0])  # Displacement
                linear_velocities.append(state_padded[1])  # Linear velocity
                angular_velocities.append(state_padded[2])  # Angular velocity
                displacement = state_padded[0]
                linear_speed = state_padded[1]
                angular_speed = state_padded[2]

                # if self.human_intervene and not self.is_intervene_step_set:
                #     self.intervene_step = step_num
                #     self.is_intervene_step_set = True
                #     print("intervene_step", self.intervene_step)

                if displacement >= DISPLACEMENT_THRESHOLD_HIGH:
                    if not self.is_high_threshold_step_set:
                        self.high_threshold_step = step_num  # Mark the step for interpolation endpoint
                        self.is_high_threshold_step_set = True

                action_value = self.calculate_action(displacement, linear_speed, angular_speed)
                action_values.append(action_value) # Action value

                self.episode.append({
                'image': top_camera_frame,
                # 'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
                'state': np.asarray(state_padded, dtype=np.float32),  # Save the padded state
                'action': np.asarray([action_value], dtype=np.float32),  # Ensure action is a tensor of shape (1,)
                'language_instruction': 'dummy instruction',
                    })
            
            # for step_num in range(EPISODE_LENGTH):
            #     displacement = self.episode[step_num]["state"][0]
            #     action_value = self.calculate_action(displacement, step_num)
            #     print("step_num", step_num, "action_value", action_value)
            #     # Ensure action is stored as a (1,) tensor, not as a scalar
            #     self.episode[step_num]["action"] = np.asarray([action_value], dtype=np.float32)  # Ensure action is a tensor of shape (1,)
            if dataset == "train":
                print("Generating train examples...")
                np.save(f'{self.output_dir}/data/train/episode_{episode_num}.npy', self.episode)
            elif dataset == "val":
                print("Generating val examples...")
                np.save(f'{self.output_dir}/data/val/episode_{episode_num}.npy', self.episode)

        # Plot after simulation


        writer.close()
        top_camera_writer.close()
        glfw.terminate()
        self.save_trajectory()
        self.plot_metrics(linear_velocities, angular_velocities, displacements, action_values)


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

    def plot_metrics(self, linear_velocities, angular_velocities, displacements, action_values):
        """
        Plot linear velocity, angular velocity, and displacement over time.
        """
        time_steps = range(len(linear_velocities))
        # print("range", range(len(linear_velocities)), range(len(angular_velocities)), range(len(displacements)), range(len(action_values)))  # 800

        plt.figure(figsize=(16, 8))

        plt.subplot(4, 1, 1)
        plt.plot(time_steps, linear_velocities, label='Linear Velocity')
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity (m/s)')
        plt.legend()
        plt.grid()

        plt.subplot(4, 1, 2)
        plt.plot(time_steps, angular_velocities, label='Angular Velocity', color='orange')
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.legend()
        plt.grid()

        plt.subplot(4, 1, 3)
        plt.plot(time_steps, displacements, label='Displacement', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Displacement (m)')
        plt.legend()
        plt.grid()

        plt.subplot(4, 1, 4)
        plt.plot(time_steps, action_values, label='Action_values', color='red')
        plt.xlabel('Time Step')
        plt.ylabel('Action_values')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        # plt.show()
        plt.savefig('./demo/cube_trajectory.png')

# def main():
#     xml_path = "./model/universal_robots_ur5e/test_scene.xml"

#     os.makedirs('data/train', exist_ok=True)
#     os.makedirs('data/val', exist_ok=True)

#     sim = Projectile(xml_path, initial_delay=0.5)  # Set the delay to 3 seconds
#     sim.reset()
#     sim.simulate(sys.argv[1])

def main():
    xml_path = "./model/universal_robots_ur5e/test_scene.xml"
    traj_path = "../ur5-scripts/traj.txt"  # Adjust path as needed
    root_output_dir = "../rlds_dataset_builder-main/zeyu_example_dataset/"

    os.makedirs(os.path.join(root_output_dir, 'data/train'), exist_ok=True)
    os.makedirs(os.path.join(root_output_dir, 'data/val'), exist_ok=True)

    sim = Projectile(xml_path, traj_path, root_output_dir, initial_delay=0.5)
    sim.reset()
    sim.simulate(sys.argv[1])


if __name__ == "__main__":
    main()