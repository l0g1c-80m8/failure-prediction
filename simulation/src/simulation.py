import mujoco
import numpy as np
import time
from typing import List, Optional

class Simulation:
    def __init__(self, model_path: str) -> None:
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(model_path)
        self._data: mujoco.MjData = mujoco.MjData(self._model)
        self._viewer: mujoco.MjModel = None
    
    def set_joint_positions(self, positions: List[float]) -> None:
        self._data.qpos[:] = positions
    
    def create_viewer(self) -> None:
        self._viewer = mujoco.Viewer(self._model, self._data)
    
    def run_trajectory(
        self, 
        initial_qpos: List[float], 
        target_qpos: List[float], 
        duration: float
    ) -> None:
        self.set_joint_positions(initial_qpos)
        
        dt: float = self._model.opt.timestep
        steps: int = int(duration / dt)

        for idx in range(steps):
            t: float = idx / steps
            current_qpos: np.ndarray = np.array(initial_qpos) + \
                t * (np.array(target_qpos) - np.array(initial_qpos))
            
            self.set_joint_positions(current_qpos.tolist())
            mujoco.mj_step(self._model, self._data)
            
            if self._viewer:
                self._viewer.render()
            
            time.sleep(dt)

    def keep_viewer_open(self) -> None:
        if self._viewer:
            while True:
                self._viewer.render()
