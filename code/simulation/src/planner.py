import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
# import matplotlib
# matplotlib.use('TkAgg')  # Choose an interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
import sys


class RRTNode:
    def __init__(self, pose: NDArray[np.float64]):
        self.pose: NDArray[np.float64] = np.array(pose)
        self.parent: Optional['RRTNode'] = None


class RRTPlanner:
    def __init__(
            self,
            start_pos: NDArray[np.float64],
            goal_pos: NDArray[np.float64],
            bounds: NDArray[np.float64],
            max_iterations: int = 3000,
            step_size: float = 0.1
    ):
        self._start_pos: RRTNode = RRTNode(start_pos)
        self._goal_pos: RRTNode = RRTNode(goal_pos)
        self._bounds: NDArray[np.float64] = bounds
        self._max_iterations: int = max_iterations
        self._step_size: float = step_size
        self._nodes: List[RRTNode] = [self._start_pos]
        self._tree: KDTree = KDTree([start_pos])

        self._plan: Optional[List[NDArray[np.float64]]] = None

    def _random_pose(self) -> NDArray[np.float64]:
        return np.random.uniform(self._bounds[:, 0], self._bounds[:, 1])

    def _nearest_node(self, pose: NDArray[np.float64]) -> RRTNode:
        _, index = self._tree.query(pose)
        return self._nodes[index]

    def _new_pose(self, from_pose: NDArray[np.float64], to_pose: NDArray[np.float64]) -> NDArray[np.float64]:
        direction: NDArray[np.float64] = to_pose - from_pose
        length: float = np.linalg.norm(direction, ord=2)
        if length > self._step_size:
            direction = direction / length * self._step_size
        return from_pose + direction

    def _add_node(self, pose: NDArray[np.float64], parent: RRTNode) -> RRTNode:
        node = RRTNode(pose)
        node.parent = parent
        self._nodes.append(node)
        self._tree = KDTree([n.pose for n in self._nodes])
        return node

    def _generate_plan(self) -> None:
        self._plan = None

        for _ in range(self._max_iterations):
            random_pose: NDArray[np.float64] = self._random_pose()
            nearest: RRTNode = self._nearest_node(random_pose)
            new_pose: NDArray[np.float64] = self._new_pose(nearest.pose, random_pose)
            new_node: RRTNode = self._add_node(new_pose, nearest)

            if np.linalg.norm(new_pose - self._goal_pos.pose) < self._step_size:
                self._goal_pos.parent = new_node
                self._plan = self._extract_path()

    def _extract_path(self) -> List[NDArray[np.float64]]:
        path: List[NDArray[np.float64]] = []
        node: Optional[RRTNode] = self._goal_pos
        while node:
            path.append(node.pose)
            node = node.parent
        return path[::-1]

    @property
    def plan(self) -> Optional[List[NDArray[np.float64]]]:
        if self._plan is None:
            self._generate_plan()

        return self._plan

    def visualize(self, path: Optional[List[NDArray[np.float64]]] = None) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        poses: NDArray[np.float64] = np.array([node.pose for node in self._nodes])
        ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], c='b', s=2)

        for node in self._nodes[1:]:
            pose: NDArray[np.float64] = node.pose
            parent_pose: NDArray[np.float64] = node.parent.pose
            ax.plot(
                [pose[0], parent_pose[0]],
                [pose[1], parent_pose[1]],
                [pose[2], parent_pose[2]], 'b-', linewidth=0.5
            )

        ax.scatter(*self._start_pos.pose, c='g', s=50, label='Start')
        ax.scatter(*self._goal_pos.pose, c='r', s=50, label='Goal')

        if self.plan:
            path_array: NDArray[np.float64] = np.array(self.plan)
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'r-', linewidth=2, label='Path')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

import numpy as np
from typing import List, Dict, Tuple, Optional
from numpy.typing import NDArray
import logging

LOGGER = logging.getLogger(__name__)

class JointTrajectoryPlanner:
    def __init__(self, 
                 joint_constraints: Dict[int, Tuple[float, float]], 
                 num_joints: int = 13,  # Updated to match UR5 MuJoCo model
                 dt: float = 0.01):
        """
        Initialize trajectory planner with joint constraints for UR5 robot.
        
        Args:
            joint_constraints: Dictionary mapping joint indices to (min, max) angle limits
            num_joints: Number of joints in the robot (13 for UR5 MuJoCo model)
            dt: Time step for trajectory discretization
        """
        self.dt = dt
        self.num_joints = num_joints
        
        # Initialize constraints for all joints
        self.joint_constraints = {}
        for i in range(num_joints):
            if i in joint_constraints:
                self.joint_constraints[i] = joint_constraints[i]
            else:
                # Default constraints if not specified
                # More conservative defaults for unspecified joints
                self.joint_constraints[i] = (-0.1, 0.1)
    
    def _check_joint_limits(self, joint_angles: np.ndarray) -> bool:
        """
        Check if joint angles are within constraints.
        
        Args:
            joint_angles: Array of joint angles to check
            
        Returns:
            bool: True if all joints are within limits
        """
        for i, angle in enumerate(joint_angles):
            min_angle, max_angle = self.joint_constraints[i]
            if angle < min_angle or angle > max_angle:
                return False
        return True
    
    def _clip_to_joint_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Clip joint angles to within their constraints.
        
        Args:
            joint_angles: Array of joint angles to clip
            
        Returns:
            np.ndarray: Clipped joint angles
        """
        clipped = joint_angles.copy()
        for i in range(len(joint_angles)):
            min_angle, max_angle = self.joint_constraints[i]
            clipped[i] = np.clip(joint_angles[i], min_angle, max_angle)
        return clipped
    
    def _pad_joint_state(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Pad joint angles to match MuJoCo model's expected dimensions.
        
        Args:
            joint_angles: Array of primary joint angles
            
        Returns:
            np.ndarray: Padded joint state vector
        """
        padded = np.zeros(self.num_joints)
        active_joints = min(len(joint_angles), 6)  # UR5 has 6 main joints
        padded[:active_joints] = joint_angles[:active_joints]
        return padded
    
    def plan_trajectory(self, 
                       start_pose: np.ndarray, 
                       end_pose: np.ndarray, 
                       velocity: float = 1.0,
                       check_constraints: bool = True) -> Optional[List[np.ndarray]]:
        """
        Plan a trajectory from start pose to end pose using linear interpolation.
        
        Args:
            start_pose: Starting joint angles (6 DOF for UR5)
            end_pose: Target joint angles (6 DOF for UR5)
            velocity: Desired velocity scaling factor (1.0 = normal speed)
            check_constraints: Whether to verify and enforce joint constraints
            
        Returns:
            List of joint configurations forming the trajectory, or None if invalid
        """
        # Pad input poses to match MuJoCo model dimensions
        start_pose_padded = self._pad_joint_state(start_pose)
        end_pose_padded = self._pad_joint_state(end_pose)
        
        # Validate input poses
        if len(start_pose_padded) != self.num_joints or len(end_pose_padded) != self.num_joints:
            LOGGER.error(f"Invalid pose dimensions. Expected {self.num_joints}")
            return None
            
        # Check if start and end poses are within constraints
        if check_constraints:
            if not self._check_joint_limits(start_pose_padded):
                LOGGER.error("Start pose violates joint constraints")
                return None
            if not self._check_joint_limits(end_pose_padded):
                LOGGER.error("End pose violates joint constraints")
                return None
                
        # Calculate the maximum joint difference to determine trajectory duration
        max_diff = np.max(np.abs(end_pose_padded - start_pose_padded))
        duration = max_diff / velocity  # Scale duration by velocity factor
        
        # Calculate number of waypoints based on duration and timestep
        num_waypoints = int(np.ceil(duration / self.dt)) + 1
        
        # Generate trajectory
        trajectory = []
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)  # Normalized time from 0 to 1
            # Linear interpolation
            waypoint = (1 - t) * start_pose_padded + t * end_pose_padded
            
            if check_constraints:
                # Clip to joint limits if needed
                waypoint = self._clip_to_joint_limits(waypoint)
                
            trajectory.append(waypoint)
            
        return trajectory
    
    def get_trajectory_statistics(self, trajectory: List[np.ndarray]) -> Dict:
        """
        Calculate basic statistics about the trajectory.
        
        Args:
            trajectory: List of joint configurations
            
        Returns:
            Dictionary containing trajectory statistics
        """
        if not trajectory:
            return {}
            
        # Convert to numpy array for easier calculations
        traj_array = np.array(trajectory)
        
        # Calculate statistics
        stats = {
            "num_waypoints": len(trajectory),
            "total_time": (len(trajectory) - 1) * self.dt,
            "joint_ranges": [],
            "max_joint_velocities": []
        }
        
        # Calculate range of motion for each joint
        for joint in range(self.num_joints):
            joint_positions = traj_array[:, joint]
            stats["joint_ranges"].append({
                "joint": joint,
                "min": np.min(joint_positions),
                "max": np.max(joint_positions),
                "range": np.max(joint_positions) - np.min(joint_positions)
            })
            
        # Calculate maximum velocities
        velocities = np.diff(traj_array, axis=0) / self.dt
        max_velocities = np.max(np.abs(velocities), axis=0)
        for joint in range(self.num_joints):
            stats["max_joint_velocities"].append({
                "joint": joint,
                "max_velocity": max_velocities[joint]
            })
            
        return stats
