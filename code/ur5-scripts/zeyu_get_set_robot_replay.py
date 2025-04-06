import urx
import time
import random
import math

def parse_trajectory(file_path):
    trajectories = []
    with open(file_path, 'r') as f:
        for line in f:
            # Remove brackets and split by comma
            values = line.strip('[] \n').split(',')
            # Convert strings to floats
            trajectory = list(map(float, values))
            trajectories.append(trajectory)
    return trajectories


def execute_trajectory(robot, trajectories, n):
    for _ in range(n):
        # Forward sequence (top to bottom)
        for i, trajectory in enumerate(trajectories):
            randomized_trajectory = trajectory[:]  # Create a copy of the trajectory
            # Add a random gain of ±10 degrees to the 6th joint
            random_gain = random.uniform(-10, 10)  # Generate random angle in degrees
            randomized_trajectory[5] += math.radians(random_gain)  # Convert to radians and apply
            
            print(f"Moving to position {i} with randomized trajectory: {randomized_trajectory}")
            t_tmp = random.uniform(1.4, 2.0)
            robot.servoj(randomized_trajectory, vel=0.1, acc=0.1, t=t_tmp, lookahead_time=0.2, gain=100, wait=True)
        
        # Reverse sequence (bottom to top)
        for i, trajectory in enumerate(reversed(trajectories)):
            randomized_trajectory = trajectory[:]  # Create a copy of the trajectory
            random_gain = random.uniform(-10, 10)
            randomized_trajectory[5] += math.radians(random_gain)
            
            print(f"Moving to position {len(trajectories)-1-i} with randomized trajectory: {randomized_trajectory}")
            t_tmp = random.uniform(1.4, 2.0)
            robot.servoj(randomized_trajectory, vel=0.1, acc=0.1, t=t_tmp, lookahead_time=0.2, gain=100, wait=True)
            
        time.sleep(0.5)  # Pause between rounds

    print('Finished moving the robot through all positions.')

if __name__ == '__main__':
    robot_left = urx.Robot("192.10.0.11")

    # Parse the trajectory from traj.txt
    trajectory = parse_trajectory("./traj.txt")
    print("Parsed trajectory:", trajectory)

    joints_left = robot_left.getj()
    print('Left robot joints: ', joints_left)
    time.sleep(1)

    # Number of rounds
    n_rounds = 2  # Adjust as needed

    # Execute the trajectory
    execute_trajectory(robot_left, trajectory, n_rounds)

    print("Finished executing the trajectory.")
