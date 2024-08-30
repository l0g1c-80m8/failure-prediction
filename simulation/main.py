import sys
import numpy as np
from datetime import datetime
from numpy.typing import NDArray
from typing import List, Optional

# local imports
from src.simulation import MjSimulation
from src.constants import KEY, RES, LOGGER_OPTIONS
from src.logger import LOGGER
from src.planner import RRTPlanner
from src.ik_solver import IkSolver


def get_joint_space_trajectory(start_pos, goal_pos, bounds, max_tries=10):
    ik_solver: IkSolver = IkSolver(RES.UR5_MODEL, KEY.UR5_EE)
    trajectory: Optional[List[np.ndarray]] = None
    tries = 0

    while trajectory is not None and (tries := tries + 1) <= max_tries:
        LOGGER.info(f'Generating trajectory - trial #{tries}')
        planner: RRTPlanner = RRTPlanner(start_pos=start_pos, goal_pos=goal_pos, bounds=bounds)
        if planner.plan:
            trajectory = ik_solver.solve_trajectory(planner.plan)

    if tries == max_tries:
        LOGGER.info(f'Exhausted max trials at #{tries}')
        return None

    LOGGER.info(f'Generated trajectory at trial #{tries}')

    return trajectory


def main() -> None:
    start_pos: NDArray[np.float64] = np.array([0.0, -0.4, 0.5], dtype=np.float64)
    goal_pos: NDArray[np.float64] = np.array([0.3, 0.4, 0.3], dtype=np.float64)
    bounds: NDArray[np.float64] = np.array([[-2, 2], [-2, 2], [-2, 2]], dtype=np.float64)

    trajectory = get_joint_space_trajectory(start_pos, goal_pos, bounds)

    simulator: MjSimulation = MjSimulation(model_path=RES.UR5_MODEL)

    initial_qpos: List[float] = [0, -1.57, 1.57, -1.57, -1.57, 0]
    target_qpos: List[float] = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]

    simulator.run_trajectory(initial_qpos, target_qpos, duration=5.0)


if __name__ == '__main__':
    # 🚀🚀🚀
    print(f'logging in this file: {LOGGER_OPTIONS.FILE}')
    LOGGER.info(f'start - {datetime.now()}')
    main()
    LOGGER.info(f'end - {datetime.now()}')

    # exit without error
    sys.exit(0)
