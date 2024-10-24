import mujoco
import numpy as np
import sys
from numpy.typing import NDArray
from typing import List, Optional
from scipy.optimize import minimize

# local
from src.constants import KEY, RES
from src.logger import LOGGER


import numpy as np
import mujoco
from typing import Optional, List, Dict, Tuple
from scipy.optimize import minimize
from numpy.typing import NDArray
import logging

LOGGER = logging.getLogger(__name__)

class IkSolver:
    def __init__(self, 
                 model_path: str, 
                 end_effector_key: str,
                 joint_constraints: Optional[Dict[int, Tuple[float, float]]] = None) -> None:
        """
        Initialize IK solver with optional joint constraints.
        
        Args:
            model_path: Path to the MuJoCo XML model file
            end_effector_key: Name of the end effector body in the model
            joint_constraints: Dictionary mapping joint indices to their (min, max) angle limits
        """
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._end_effector_id = self._model.body(end_effector_key).id
        self._joint_constraints = joint_constraints or {}
        
        # Create bounds for all joints
        self._bounds = []
        for i in range(self._model.nq):
            if i in self._joint_constraints:
                self._bounds.append(self._joint_constraints[i])
            else:
                # Use default bounds if not specified (-2π to 2π)
                self._bounds.append((-2 * np.pi, 2 * np.pi))
    
    def _get_end_effector_position(self, joint_configs: np.ndarray) -> np.ndarray:
        self._data.qpos[:] = joint_configs
        mujoco.mj_forward(self._model, self._data)
        return self._data.xpos[self._end_effector_id]
    
    def set_joint_constraint(self, joint_idx: int, min_angle: float, max_angle: float) -> None:
        """
        Set constraint for a specific joint.
        
        Args:
            joint_idx: Index of the joint to constrain
            min_angle: Minimum allowed angle (in radians)
            max_angle: Maximum allowed angle (in radians)
        """
        self._joint_constraints[joint_idx] = (min_angle, max_angle)
        self._bounds[joint_idx] = (min_angle, max_angle)
    
    def _calculate_ik(self, target_pos: np.ndarray) -> Optional[np.ndarray]:
        def objective(q: np.ndarray) -> float:
            self._data.qpos[:] = q
            mujoco.mj_forward(self._model, self._data)
            current_pos = self._data.xpos[self._end_effector_id]
            return np.linalg.norm(current_pos - target_pos)
        
        initial_guess = self._data.qpos.copy()
        
        # Ensure initial guess satisfies constraints
        for i, (min_val, max_val) in enumerate(self._bounds):
            initial_guess[i] = np.clip(initial_guess[i], min_val, max_val)
        
        # Use L-BFGS-B method which supports bounds
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=self._bounds,
            options={'gtol': 1e-4, 'maxiter': 1000}
        )
        
        if result.success or result.fun < 1e-6:
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
                LOGGER.info(f'Waypoint {idx}: Joint config: {joint_config}, End effector position: {ee_pos}')
        else:
            LOGGER.info('Failed to find a trajectory.')
        
        return joint_trajectory
