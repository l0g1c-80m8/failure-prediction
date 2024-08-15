import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import List, Optional

# local import
from src.logger import LOGGER


class MjSimulation:
    def __init__(self, model_path: str) -> None:
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(model_path)
        self._data: mujoco.MjData = mujoco.MjData(self._model)

    def set_joint_positions(self, positions: List[float]) -> None:
        if len(positions) != self._model.nq:
            raise ValueError(f'Expected {self._model.nq} joint positions, got {len(positions)}')
        self._data.qpos[:] = positions

    def run_trajectory(
            self,
            initial_qpos: List[float],
            target_qpos: List[float],
            duration: float
    ) -> None:
        self.set_joint_positions(initial_qpos)

        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            dt: float = self._model.opt.timestep
            steps: int = int(duration / dt)
            idx = 0

            while viewer.is_running() and (idx := idx + 1) < steps:
                step_start = time.time()

                LOGGER.info(f'on step {1 + idx} of {steps} steps')
                t: float = idx / steps
                current_qpos: np.ndarray = np.array(initial_qpos) + \
                                           t * (np.array(target_qpos) - np.array(initial_qpos))

                self.set_joint_positions(current_qpos.tolist())
                mujoco.mj_step(self._model, self._data)

                viewer.sync()

                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
