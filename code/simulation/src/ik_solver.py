import mujoco
import numpy as np
import sys
from numpy.typing import NDArray
from typing import List, Optional
from scipy.optimize import minimize

# local
from src.constants import KEY, RES
from src.logger import LOGGER


class IkSolver:
    def __init__(self, model_path: str, end_effector_key: str) -> None:
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._end_effector_id = self._model.body(end_effector_key).id

    def _get_end_effector_position(self, joint_configs: np.ndarray) -> np.ndarray:
        self._data.qpos[:] = joint_configs
        mujoco.mj_forward(self._model, self._data)
        return self._data.xpos[self._end_effector_id]

    def _calculate_ik(self, target_pos: np.ndarray) -> Optional[np.ndarray]:
        def objective(q: np.ndarray) -> float:
            self._data.qpos[:] = q
            mujoco.mj_forward(self._model, self._data)
            current_pos = self._data.xpos[self._end_effector_id]
            return np.linalg.norm(current_pos - target_pos)

        initial_guess = self._data.qpos.copy()
        result = minimize(objective, initial_guess, method='BFGS', options={'gtol': 1e-4, 'maxiter': 1000})

        if result.success or result.fun < 1e-3:  # Allow solutions that are close enough
            LOGGER.info(f"IK solved. Error: {result.fun}")
            return result.x
        else:
            LOGGER.warning(f"IK failed. Error: {result.fun}")
            return None

    def solve_trajectory(self, cartesian_waypoints: List[NDArray[np.float64]]) -> List[NDArray[np.float64]]:
        joint_trajectory = []
        for waypoint in cartesian_waypoints:
            LOGGER.info(f'Solving IK for waypoint: {waypoint}')
            ik_solution = self._calculate_ik(np.array(waypoint))
            if ik_solution is not None:
                joint_trajectory.append(ik_solution)
            else:
                LOGGER.warning(f'Warning: No IK solution found for waypoint {waypoint}')

        if joint_trajectory:
            LOGGER.info(f'Trajectory found with {len(joint_trajectory)} points.')
            for idx, joint_config in enumerate(joint_trajectory):
                ee_pos = self._get_end_effector_position(joint_config)
                LOGGER.info(f'Waypoint {idx}: Joint config: {joint_trajectory}, End effector position: {ee_pos}')
        else:
            LOGGER.info('Failed to find a trajectory.')

        return joint_trajectory


if __name__ == "__main__":
    iks = IkSolver(UR5_MODEL, KEY.UR5_EE)
    waypoints = [
        (0.4, 0.2, 0.5),  # front, slightly to the right
        (0.4, -0.2, 0.5),  # front, slightly to the left
        (0.6, 0.0, 0.3),  # further front, lower
        (0.5, 0.3, 0.7),  # higher position
        (0.4, 0.0, 0.5)  # back to a central position
    ]
    trajectory = iks.solve_trajectory(waypoints)

    if trajectory:
        print(f'Trajectory found with {len(trajectory)} points.')

    # exit without error
    sys.exit(0)
