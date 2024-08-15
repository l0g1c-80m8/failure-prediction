import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
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
            max_iterations: int = 1000,
            step_size: float = 0.1
    ):
        self._start_pos: RRTNode = RRTNode(start_pos)
        self._goal_pos: RRTNode = RRTNode(goal_pos)
        self._bounds: NDArray[np.float64] = bounds
        self._max_iterations: int = max_iterations
        self._step_size: float = step_size
        self._nodes: List[RRTNode] = [self._start_pos]
        self._tree: KDTree = KDTree([start_pos])

        self._plan = Optional[List[NDArray[np.float64]]]

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
        for _ in range(self._max_iterations):
            random_pose: NDArray[np.float64] = self._random_pose()
            nearest: RRTNode = self._nearest_node(random_pose)
            new_pose: NDArray[np.float64] = self._new_pose(nearest.pose, random_pose)
            new_node: RRTNode = self._add_node(new_pose, nearest)

            if np.linalg.norm(new_pose - self._goal_pos.pose) < self._step_size:
                self._goal_pos.parent = new_node
                self._plan = self._extract_path()

        self._plan = None

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

        print(self._plan)
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


if __name__ == '__main__':
    rrt_planner: RRTPlanner = RRTPlanner(
        np.array([0, 0, 0], dtype=np.float64),
        np.array([10, 10, 10], dtype=np.float64),
        np.array([[-10, 10], [-10, 10], [-10, 10]])  # [min, max] for each dim
    )

    if rrt_planner.plan:
        print('Path found!')
        rrt_planner.visualize()
    else:
        print('No path found.')
        rrt_planner.visualize()

    # exit without error
    sys.exit(0)
