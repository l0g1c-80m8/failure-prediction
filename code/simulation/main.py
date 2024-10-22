import sys
import numpy as np
from datetime import datetime
from numpy.typing import NDArray
from typing import List, Optional, Dict, Tuple

# local imports
from src.simulation import MjSimulation
from src.constants import KEY, RES, LOGGER_OPTIONS
from src.logger import LOGGER
from src.planner import RRTPlanner, JointTrajectoryPlanner
from src.ik_solver import IkSolver


def get_joint_space_trajectory(
        start_pos: NDArray[np.float64],
        goal_pos: NDArray[np.float64],
        bounds: NDArray[np.float64],
        constraints: Dict[int, Tuple[float, float]],
        max_tries: int = 10
) -> Optional[List[NDArray[np.float64]]]:
    ik_solver: IkSolver = IkSolver(RES.UR5_MODEL, KEY.UR5_EE, joint_constraints=constraints)
    trajectory: Optional[List[NDArray[np.float64]]] = None
    tries: int = 0

    while trajectory is None and (tries := tries + 1) <= max_tries:
        LOGGER.info(f'Generating trajectory - trial #{tries}')
        planner: RRTPlanner = RRTPlanner(start_pos=start_pos, goal_pos=goal_pos, bounds=bounds)
        if planner.plan:
            trajectory = ik_solver.solve_trajectory(planner.plan)

    if tries == max_tries:
        LOGGER.info(f'Exhausted max trials at #{tries}')
        return None

    LOGGER.info(f'Generated trajectory at trial #{tries}')

    return trajectory

def get_simple_joint_trajectory(
    start_pos: NDArray[np.float64],
    goal_pos: NDArray[np.float64],
    constraints: Dict[int, Tuple[float, float]]) -> Optional[List[NDArray[np.float64]]]:
    planner = JointTrajectoryPlanner(constraints, num_joints=6)
    
    trajectory = planner.plan_trajectory(start_pos, goal_pos, velocity=1.0)
    
    return trajectory


def main() -> None:
    start_pos: NDArray[np.float64] = np.array([0, -0.997, 0.089], dtype=np.float64)
    goal_pos: NDArray[np.float64] = np.array([0, 0.997, 0.089], dtype=np.float64)
    bounds: NDArray[np.float64] = np.array([[-2, 2], [-2, 2], [-2, 2]], dtype=np.float64)
    constraints: Dict[int, Tuple[float, float]] = {
        0: (-np.pi / 2, np.pi / 2),
        1: (-np.pi / 2, -np.pi / 2),
        2: (np.pi / 2, np.pi / 2),
        3: (0, 0),
        4: (0, 0),
        5: (0, 0),
    }
    # in joint angles
    start_joint_pos: NDArray[np.float64] = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, 0, 0, 0], dtype=np.float64)
    goal_joint_pos: NDArray[np.float64] = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, 0, 0, 0], dtype=np.float64)


    trajectory: List[NDArray[np.float64]] = get_joint_space_trajectory(start_pos, goal_pos, bounds, constraints)
    trajectory: List[NDArray[np.float64]] = get_simple_joint_trajectory(start_joint_pos, goal_joint_pos, constraints)

    simulator: MjSimulation = MjSimulation(model_path=RES.UR5_MODEL, trajectory=trajectory, speed=0.7)
    simulator.run_trajectory()


if __name__ == '__main__':
    # 🚀🚀🚀
    print(f'logging in this file: {LOGGER_OPTIONS.FILE}')
    LOGGER.info(f'start - {datetime.now()}')
    main()
    LOGGER.info(f'end - {datetime.now()}')

    # exit without error
    sys.exit(0)
