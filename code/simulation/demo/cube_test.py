import mujoco as mj
import numpy as np
from mujoco.glfw import glfw

from mujoco_base import MuJoCoBase


class Projectile(MuJoCoBase):
    def __init__(self, xml_path, initial_delay=3.0):
        super().__init__(xml_path)
        self.angular_speed = 1.0  # Angular speed in radians per second
        self.initial_delay = initial_delay  # Delay before starting movement
        self.start_time = None  # To track the start time
        self.positions = []  # To store the position data

    def reset(self):
        # Set initial position of the cube
        self.data.qpos[2] = 1.0  # Adjust according to your cube's z position

        # Reset cube velocity
        cube_joint_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, 'free_cube')
        self.data.qvel[cube_joint_index] = 0.0

        # Set camera configuration
        self.cam.azimuth = 90.0
        self.cam.distance = 2.5
        self.cam.elevation = -40.0

        # Initialize start time
        self.start_time = None
        # Clear position data
        self.positions = []

        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        if self.start_time is None:
            # Set start time when controller is first called
            self.start_time = data.time
            return

        elapsed_time = data.time - self.start_time
        if elapsed_time < self.initial_delay:
            # Delay not yet passed, do nothing
            return

        # Apply gravity or velocity to make the cube drop after the delay
        # Set the free_cube joint velocity to simulate dropping
        joint_index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'free_cube')
        # gravity = -9.81  # Gravity constant

        # Apply initial downward velocity or force if needed
        # Adjust this to fit your exact scenario
        # self.data.qvel[joint_index] = gravity

        # Calculate the desired position of the shoulder_pan_joint
        angular_velocity = self.angular_speed  # rad/s
        desired_position = np.sin(angular_velocity * (elapsed_time - self.initial_delay))  # Sinusoidal motion

        # Set the control input for `shoulder_pan_joint`
        joint_index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'shoulder_pan_joint')
        data.ctrl[joint_index] = desired_position

    def simulate(self):
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

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()
        # Save the trajectory data to a file or plot it
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
