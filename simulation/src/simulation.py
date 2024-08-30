import mujoco
import mujoco.viewer
import numpy as np
import time
from numpy.typing import NDArray
from typing import List, Optional

# local import
from src.logger import LOGGER


class MjSimulation:
    def __init__(self, model_path: str, trajectory: List[NDArray[np.float64]], duration: float = 0.5) -> None:
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(model_path)
        self._data: mujoco.MjData = mujoco.MjData(self._model)
        self._trajectory = trajectory
        self._duration = duration

    def set_joint_positions(self, positions: List[float]) -> None:
        if len(positions) != self._model.nq:
            raise ValueError(f'Expected {self._model.nq} joint positions, got {len(positions)}')
        self._data.qpos[:] = positions

    def run_trajectory(self) -> None:
        if not self._trajectory or len(self._trajectory) == 0:
            LOGGER.warning("No trajectory to run.")
            return

        self.set_joint_positions(self._trajectory[0].tolist())

        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            dt: float = self._model.opt.timestep
            idx: int = 0
            sub_step: float = 0.0
            trajectory_length = len(self._trajectory)

            while viewer.is_running() and idx < trajectory_length - 1:
                step_start = time.time()

                LOGGER.info(f'Simulation on step {idx + 1} (sub step {sub_step}) of {trajectory_length}')

                current_pos = self._trajectory[idx]
                next_pos = self._trajectory[idx + 1]
                alpha = sub_step / (self._duration / dt)
                interpolated_pos = current_pos * (1 - alpha) + next_pos * alpha

                self.set_joint_positions(interpolated_pos.tolist())
                mujoco.mj_step(self._model, self._data)
                viewer.sync()

                sub_step += dt
                if sub_step >= self._duration:
                    idx += 1
                    sub_step = 0.0

                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            LOGGER.info("Trajectory completed.")


