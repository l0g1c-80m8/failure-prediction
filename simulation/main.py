import sys
from typing import List

# local imports
from src.simulation import MjSimulation
from src.constants import RES

def main() -> None:
    simulator: MjSimulation = MjSimulation(RES.UR5_MODEL)

    initial_qpos: List[float] = [0, -1.57, 1.57, -1.57, -1.57, 0]
    target_qpos: List[float] = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]

    simulator.create_viewer()

    simulator.run_trajectory(initial_qpos, target_qpos, duration=5.0)

    simulator.keep_viewer_open()

if __name__ == '__main__':
    main()
    
    # exit without error
    sys.exit(0)
    