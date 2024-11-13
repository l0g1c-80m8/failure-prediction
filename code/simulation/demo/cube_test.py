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

N_TRAIN_EPISODES = 10
N_VAL_EPISODES = 10
EPISODE_LENGTH = 200  # Number of points in trajectory

# Thresholds for action calculation
DISPLACEMENT_THRESHOLD_HIGH = 0.2


class Projectile(MuJoCoBase):
    def __init__(self, xml_path, initial_delay=3.0):
        super().__init__(xml_path)
        self.init_angular_speed = 1.0  # Angular speed in radians per second
        self.initial_delay = initial_delay  # Delay before starting movement
        self.start_time = None  # To track the start time
        self.positions = []  # To store the position data
        self.episode = [] # To store the episode data
        self.human_intervene = False  # New attribute to track cube drop
        self.intervene_step = 0  # Step when human intervention occurs
        self.is_intervene_step_set = False
        self.high_threshold_step = 0  # Step when displacement > DISPLACEMENT_THRESHOLD_HIGH
        self.is_high_threshold_step_set = False

    def reset(self):

        # Set camera configuration
        self.cam.azimuth = random.uniform(225, 315)
        self.cam.distance = random.uniform(2, 3)
        self.cam.elevation = random.uniform(-50, -30)
        self.speed_up = random.uniform(-1, 1)
        # print("self.cam", self.cam.azimuth, self.cam.distance, self.cam.elevation)

        self.angular_speed = self.init_angular_speed
        self.human_intervene = False  # New attribute to track cube drop
        self.intervene_step = 0  # Step when human intervention occurs
        self.is_intervene_step_set = False
        self.high_threshold_step = 0  # Step when displacement > DISPLACEMENT_THRESHOLD_HIGH
        self.is_high_threshold_step_set = False

        # Initialize start time
        self.start_time = None
        # Clear position data
        self.positions = []
        self.episode = []  # Reset episode data for each new episode

        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')
        fixed_box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_box')
        self.data.qpos[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3] = [0.4, 0.45, 0.6]

        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        if self.start_time is None:
            # Set start time and initial position when controller is first called
            self.start_time = data.time
            # Get the initial position of the shoulder_pan_joint
            joint_index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'shoulder_pan_joint')
            self.initial_position = data.qpos[joint_index]
            return

        elapsed_time = data.time - self.start_time
        if elapsed_time < self.initial_delay:
            # Delay not yet passed, do nothing
            return

        # Define the angular position in radians and calculate the offset
        angular_position = self.initial_position + (elapsed_time - self.initial_delay) * self.angular_speed

        # Apply a sudden speed increase
        if angular_position >= self.speed_up:
            self.angular_speed *= 2.0  # Double the speed at 90 degrees
            self.human_intervene = True

        # Clamp the desired position to 0–π radians
        desired_position = np.clip(angular_position, -np.pi/2, np.pi/2)

        # Set the control for `shoulder_pan_joint`
        joint_index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'shoulder_pan_joint')
        data.ctrl[joint_index] = desired_position

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
        print(f"Time: {self.data.time:.2f}s | Linear Speed: {linear_speed:.2f} m/s | Angular Speed: {angular_speed:.2f} rad/s | Relative Displacement: {relative_displacement:.2f} m")
        
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
    def calculate_action(self, displacement, current_step):
        # Set action to 0 if no human intervention
        if current_step <= self.intervene_step:
            return 0

        # Set action to 1 if displacement exceeds the high threshold
        if displacement >= DISPLACEMENT_THRESHOLD_HIGH:
            return 1

        # print("current_step > self.intervene_step", current_step > self.intervene_step)
        # print("self.high_threshold_step", self.high_threshold_step)
        # print("self.intervene_step", self.intervene_step)
        # Exponential interpolation for action between intervention and high threshold
        if current_step > self.intervene_step and self.high_threshold_step > self.intervene_step:
            # Normalize the current step between intervene and high threshold steps
            normalized_step = (current_step - self.intervene_step) / (self.high_threshold_step - self.intervene_step)
            # print("normalized_step", normalized_step)
            # Exponential growth from 0 to 1
            action_value = 1 - np.exp(-5 * normalized_step)
            return action_value


    def simulate(self, dataset="train"):
        video_dir = './demo'
        video_filename = os.path.join(video_dir, 'simulation_video.mp4')

        # Create directory if it does not exist
        os.makedirs(video_dir, exist_ok=True)

        # Load the 'home' keyframe for initial position
        keyframe_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, 'home')
        if keyframe_id >= 0:
            mj.mj_resetDataKeyframe(self.model, self.data, keyframe_id)

        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')
        fixed_box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'fixed_box')
        self.data.qpos[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3] = [0.4, 0.45, 0.6]

        writer = imageio.get_writer(video_filename, fps=60)

        # while not glfw.window_should_close(self.window):
        if dataset == "train":
            N_EPISODES = N_TRAIN_EPISODES
        elif dataset == "val":
            N_EPISODES = N_VAL_EPISODES


        for episode_num in range(N_EPISODES):

            # Load the 'home' keyframe for initial position
            keyframe_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, 'home')
            if keyframe_id >= 0:
                mj.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
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

                # swap OpenGL buffers (blocking call due to v-sync)
                glfw.swap_buffers(self.window)

                # process pending GUI events, call GLFW callbacks
                glfw.poll_events()


                state_padded = self.data_collection(cube_body_id, fixed_box_body_id)
                displacement = state_padded[0]

                if self.human_intervene and not self.is_intervene_step_set:
                    self.intervene_step = step_num
                    self.is_intervene_step_set = True
                    print("intervene_step", self.intervene_step)

                if displacement >= DISPLACEMENT_THRESHOLD_HIGH:
                    if not self.is_high_threshold_step_set:
                        self.high_threshold_step = step_num  # Mark the step for interpolation endpoint
                        self.is_high_threshold_step_set = True

                self.episode.append({
                'image': framebuffer,
                'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
                'state': np.asarray(state_padded, dtype=np.float32),  # Save the padded state
                # 'action': action_value,  # Ensure action is a tensor of shape (1,)
                'language_instruction': 'dummy instruction',
                    })
            
            for step_num in range(EPISODE_LENGTH):
                displacement = self.episode[step_num]["state"][0]
                action_value = self.calculate_action(displacement, step_num)
                print("step_num", step_num, "action_value", action_value)
                # Ensure action is stored as a (1,) tensor, not as a scalar
                self.episode[step_num]["action"] = np.asarray([action_value], dtype=np.float32)  # Ensure action is a tensor of shape (1,)
            if dataset == "train":
                print("Generating train examples...")
                np.save(f'data/train/episode_{episode_num}.npy', self.episode)
            elif dataset == "val":
                print("Generating val examples...")
                np.save(f'data/val/episode_{episode_num}.npy', self.episode)


        writer.close()
        glfw.terminate()
        self.save_trajectory()


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


def main():
    xml_path = "./model/universal_robots_ur5e/test_scene.xml"

    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/val', exist_ok=True)

    sim = Projectile(xml_path, initial_delay=0.5)  # Set the delay to 3 seconds
    sim.reset()
    sim.simulate(sys.argv[1])


if __name__ == "__main__":
    main()
