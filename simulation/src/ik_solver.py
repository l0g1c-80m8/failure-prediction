import mujoco
import numpy as np
import sys
from typing import List, Tuple, Optional

# local
from constants import KEY, RES
from logger import LOGGER


class MjIkSolver:
    def __init__(self, model_path: str, end_effector_key: str) -> None:
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._end_effector_id = self._model.body(end_effector_key).id

        self._trajectory = None

    def _calculate_ik(self, target_pos: np.ndarray) -> Optional[np.ndarray]:
        def objective(q: np.ndarray) -> float:
            self._data.qpos[:] = q
            mujoco.mj_forwardPosition(self._model, self._data)
            current_pos = self._data.xpos[self._end_effector_id]
            return np.linalg.norm(current_pos - target_pos)

        initial_guess = self._data.qpos.copy()
        result = mujoco.mj_inversePosition(self._model, self._data, target_pos, initial_guess)

        if result.success:
            return self._data.qpos
        else:
            return None

    def solve_trajectory(self, cartesian_waypoints: List[Tuple[float, float, float]]) -> List[np.ndarray]:
        joint_trajectory = []
        for waypoint in cartesian_waypoints:
            ik_solution = self._calculate_ik(np.array(waypoint))
            if ik_solution is not None:
                joint_trajectory.append(ik_solution)
            else:
                LOGGER.warning(f'Warning: No IK solution found for waypoint {waypoint}')
        return trajectory


if __name__ == "__main__":
    iks = MjIkSolver(RES.UR5_MODEL, KEY.UR5_EE)
    waypoints = [
        (0.5, 0.5, 0.5),
        (0.7, 0.3, 0.6),
        (0.4, 0.6, 0.4)
    ]
    trajectory = iks.solve_trajectory(waypoints)
    print(f'Final end effector position: {trajectory}')

    # exit without error
    sys.exit(0)
