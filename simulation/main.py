import sys
from datetime import datetime
from typing import List

# local imports
from src.simulation import MjSimulation
from src.constants import RES, LOGGER_OPTIONS
from src.logger import LOGGER

def main() -> None:
    simulator: MjSimulation = MjSimulation(RES.UR5_MODEL)

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
    