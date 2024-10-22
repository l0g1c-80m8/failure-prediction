import mujoco
import mujoco.viewer
import numpy as np
import time
from numpy.typing import NDArray
from typing import List, Optional

# local import
from src.logger import LOGGER


class MjSimulation:
    def __init__(self, model_path: str, trajectory: List[NDArray[np.float64]], speed: float = 1.0) -> None:
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(model_path)
        self._data: mujoco.MjData = mujoco.MjData(self._model)
        self._trajectory = trajectory
        self._speed = speed
        
        # Get indices for robot joints and cube
        self._robot_joint_indices = list(range(6))  # First 6 DOF for UR5e
        self._cube_start_idx = 6  # Index where cube DOF starts
        
        # Reset the simulation
        mujoco.mj_resetData(self._model, self._data)
        
        # Initialize cube position
        self.reset_cube()

    def reset_cube(self):
        """Reset cube to initial position above panel"""
        # The first 6 values are robot joints, next 7 are cube position (3 pos + 4 quat)
        self._data.qpos[self._cube_start_idx:self._cube_start_idx+3] = [0, 0.25, 0.04]  # x, y, z
        self._data.qpos[self._cube_start_idx+3:self._cube_start_idx+7] = [1, 0, 0, 0]   # quaternion
        
    def set_robot_position(self, positions: List[float]) -> None:
        """Set only robot joint positions"""
        if len(positions) != len(self._robot_joint_indices):
            raise ValueError(f'Expected {len(self._robot_joint_indices)} joint positions, got {len(positions)}')
        for i, pos in enumerate(self._robot_joint_indices):
            self._data.qpos[pos] = positions[i]
            # Also set the position control target
            self._data.ctrl[i] = positions[i]

    def run_physics_with_fixed_robot(self, duration: float, robot_positions: List[float], viewer) -> None:
        """Run physics simulation while holding robot position fixed"""
        steps = int(duration / self._model.opt.timestep)
        for _ in range(steps):
            # Reset robot position before each physics step
            self.set_robot_position(robot_positions)
            mujoco.mj_step(self._model, self._data)
            viewer.sync()
            time.sleep(self._model.opt.timestep)

    def interpolate_positions(self, start_pos: NDArray[np.float64], 
                            end_pos: NDArray[np.float64], 
                            t: float) -> NDArray[np.float64]:
        """Smooth interpolation between positions"""
        # Simple linear interpolation
        return start_pos + (end_pos - start_pos) * t

    def execute_trajectory_segment(self, start_pos: NDArray[np.float64], 
                                 end_pos: NDArray[np.float64], 
                                 duration: float,
                                 viewer) -> None:
        """Execute a single trajectory segment with smooth motion"""
        steps = int(duration / self._model.opt.timestep)
        for step in range(steps):
            t = step / steps
            
            # Smooth interpolation
            current_pos = self.interpolate_positions(start_pos, end_pos, t)
            self.set_robot_position(current_pos.tolist())
            
            # Physics step
            mujoco.mj_step(self._model, self._data)
            viewer.sync()
            
            # Maintain real-time
            time.sleep(self._model.opt.timestep)

    def run_trajectory(self) -> None:
        if not self._trajectory or len(self._trajectory) == 0:
            LOGGER.warning("No trajectory to run.")
            return

        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            initial_position = self._trajectory[0].tolist()
            final_position = self._trajectory[-1].tolist()
            
            # 1. Set initial robot position and let it settle
            LOGGER.info("Setting initial position...")
            self.set_robot_position(initial_position)
            self.run_physics_with_fixed_robot(0.5, initial_position, viewer)
            
            # 2. Let cube fall and settle while holding robot position
            LOGGER.info("Letting cube fall and settle...")
            self.run_physics_with_fixed_robot(5.0, initial_position, viewer)
            
            # 3. Execute trajectory with smooth motion
            LOGGER.info("Executing robot trajectory...")
            start_time = time.time()
            
            for i in range(len(self._trajectory) - 1):
                start_pos = self._trajectory[i]
                end_pos = self._trajectory[i + 1]
                
                # Calculate duration based on distance and speed
                distance = np.linalg.norm(end_pos - start_pos)
                duration = distance / self._speed
                
                # Execute segment
                self.execute_trajectory_segment(start_pos, end_pos, duration, viewer)
                
                LOGGER.info(f'Completed segment {i+1}/{len(self._trajectory)-1}')
            
            # 4. Hold final position while continuing physics
            LOGGER.info("Final wait period...")
            self.run_physics_with_fixed_robot(3.0, final_position, viewer)
            
            LOGGER.info(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
