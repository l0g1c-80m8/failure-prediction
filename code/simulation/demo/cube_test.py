import os
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw

from mujoco_base import MuJoCoBase
import imageio


class Projectile(MuJoCoBase):
    def __init__(self, xml_path, initial_delay=3.0):
        super().__init__(xml_path)
        self.angular_speed = 1.0  # Angular speed in radians per second
        self.initial_delay = initial_delay  # Delay before starting movement
        self.start_time = None  # To track the start time
        self.positions = []  # To store the position data

    def reset(self):

        # Set camera configuration
        self.cam.azimuth = 270.0
        self.cam.distance = 2.5
        self.cam.elevation = -40.0

        # Initialize start time
        self.start_time = None
        # Clear position data
        self.positions = []

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

        # Calculate the desired position of the shoulder_pan_joint starting from initial_position
        angular_velocity = self.angular_speed  # rad/s
        offset = np.sin(angular_velocity * (elapsed_time - self.initial_delay))  # Sinusoidal offset
        desired_position = self.initial_position + offset

        # Set the control input for `shoulder_pan_joint`
        joint_index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'shoulder_pan_joint')
        data.ctrl[joint_index] = desired_position

    def simulate(self):
        video_dir = './demo'
        video_filename = os.path.join(video_dir, 'simulation_video.mp4')

        # Create directory if it does not exist
        os.makedirs(video_dir, exist_ok=True)

        # Load the 'home' keyframe for initial position
        keyframe_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, 'home')
        if keyframe_id >= 0:
            mj.mj_resetDataKeyframe(self.model, self.data, keyframe_id)

        cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'free_cube')
        self.data.qpos[self.model.body_jntadr[cube_body_id]:self.model.body_jntadr[cube_body_id]+3] = [0.4, 0.45, 0.6]

        writer = imageio.get_writer(video_filename, fps=60)

        while not glfw.window_should_close(self.window):
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
    sim = Projectile(xml_path, initial_delay=3.0)  # Set the delay to 3 seconds
    sim.reset()
    sim.simulate()


if __name__ == "__main__":
    main()
