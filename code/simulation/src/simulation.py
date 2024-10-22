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

    def set_joint_positions(self, positions: List[float]) -> None:
        if len(positions) != self._model.nq:
            raise ValueError(f'Expected {self._model.nq} joint positions, got {len(positions)}')
        self._data.qpos[:] = positions
    
    def wait(self, count: int):
        time.sleep(1 * count)

    def run_trajectory(self) -> None:
        if not self._trajectory or len(self._trajectory) == 0:
            LOGGER.warning("No trajectory to run.")
            return

        self.set_joint_positions(self._trajectory[0].tolist())

        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            dt: float = self._model.opt.timestep
            total_time: float = 0.0
            trajectory_length = len(self._trajectory)
            start_time = time.time()

            segment_durations = [np.linalg.norm(self._trajectory[i + 1] - self._trajectory[i]) / self._speed
                                 for i in range(trajectory_length - 1)]
            total_duration = sum(segment_durations)

            self.wait(2)

            while viewer.is_running() and total_time < total_duration:
                step_start = time.time()

                # find the current segment
                segment_idx = 0
                accumulated_time = 0.0
                for i, duration in enumerate(segment_durations):
                    if accumulated_time + duration > total_time:
                        segment_idx = i
                        break
                    accumulated_time += duration

                current_pos = self._trajectory[segment_idx]
                next_pos = self._trajectory[segment_idx + 1]
                segment_duration = segment_durations[segment_idx]
                segment_progress = (total_time - accumulated_time) / segment_duration

                # smooth interpolation using cubic hermite spline
                t = segment_progress
                h00 = 2 * t ** 3 - 3 * t ** 2 + 1
                h10 = t ** 3 - 2 * t ** 2 + t
                h01 = -2 * t ** 3 + 3 * t ** 2
                h11 = t ** 3 - t ** 2

                # estimate velocity
                vel_scale = 0.5
                current_vel = (next_pos - current_pos) * vel_scale
                next_vel = current_vel  # assuming continuous velocity

                interpolated_pos = (h00 * current_pos +
                                    h10 * current_vel * segment_duration +
                                    h01 * next_pos +
                                    h11 * next_vel * segment_duration)

                self.set_joint_positions(interpolated_pos.tolist())
                mujoco.mj_step(self._model, self._data)
                viewer.sync()

                total_time += dt
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                if int(total_time / dt) % 100 == 0:
                    LOGGER.info(f'Simulation progress: {total_time:.2f}s / {total_duration:.2f}s')

            # Ensure the final position is reached
            self.set_joint_positions(self._trajectory[-1].tolist())
            mujoco.mj_step(self._model, self._data)
            viewer.sync()

            LOGGER.info(f"Trajectory completed in {time.time() - start_time:.2f} seconds.")

            self.wait(15)
