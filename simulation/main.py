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


def get_joint_space_trajectory(
        start_pos: NDArray[np.float64],
        goal_pos: NDArray[np.float64],
        bounds: NDArray[np.float64],
        max_tries: int = 10
) -> Optional[List[NDArray[np.float64]]]:
    ik_solver: IkSolver = IkSolver(RES.UR5_MODEL, KEY.UR5_EE)
    trajectory: Optional[List[NDArray[np.float64]]] = None
    tries = 0

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


def main() -> None:
    start_pos: NDArray[np.float64] = np.array([-0.4, -0.4, 0.5], dtype=np.float64)
    goal_pos: NDArray[np.float64] = np.array([0.4, 0.4, 0.5], dtype=np.float64)
    bounds: NDArray[np.float64] = np.array([[-2, 2], [-2, 2], [0.5, 0.5]], dtype=np.float64)

    trajectory: List[NDArray[np.float64]] = get_joint_space_trajectory(start_pos, goal_pos, bounds)

    simulator: MjSimulation = MjSimulation(model_path=RES.UR5_MODEL, trajectory=trajectory)

    simulator.run_trajectory()


if __name__ == '__main__':
    # 🚀🚀🚀
    print(f'logging in this file: {LOGGER_OPTIONS.FILE}')
    LOGGER.info(f'start - {datetime.now()}')
    main()
    LOGGER.info(f'end - {datetime.now()}')

    # exit without error
    sys.exit(0)
