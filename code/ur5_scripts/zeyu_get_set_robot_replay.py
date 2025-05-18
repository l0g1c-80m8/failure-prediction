import urx
import time
import random
import math

def parse_trajectory(file_path, shuffle=True):
    trajectories = []
    with open(file_path, 'r') as f:
        for line in f:
            # Remove brackets and split by comma
            values = line.strip('[] \n').split(',')
            # Convert strings to floats
            trajectory = list(map(float, values))
            trajectories.append(trajectory)
    if shuffle:
        random.shuffle(trajectories)  # Shuffle the trajectory
    return trajectories


def execute_trajectory(robot, trajectories, n):
    print("Robot execution thread starting...")
    t_tmp = random.uniform(4.4, 5.0)
    print(f"Using time parameter: {t_tmp}")

    vel_scale = random.uniform(0.05, 0.1)
    acc_scale = random.uniform(0.05, 0.1)
    
    for _ in range(n):
        # Forward sequence (top to bottom)
        tilt_angle = 15
        for i, trajectory in enumerate(trajectories):
            randomized_trajectory = trajectory[:]  # Create a copy of the trajectory
            # Add a random gain of ±10 degrees to the 6th joint
            random_gain = random.uniform(0, tilt_angle)  # Generate random tilt angle in degrees
            randomized_trajectory[5] += math.radians(random_gain)  # Convert to radians and apply
            
            print(f"Moving to position {i} with randomized trajectory: {randomized_trajectory}")
            robot.servoj(randomized_trajectory, vel=vel_scale, acc=acc_scale, t=t_tmp, lookahead_time=0.2, gain=100, wait=True)
        
        # Reverse sequence (bottom to top)
        for i, trajectory in enumerate(reversed(trajectories)):
            randomized_trajectory = trajectory[:]  # Create a copy of the trajectory
            random_gain = random.uniform(-tilt_angle, 0)
            randomized_trajectory[5] += math.radians(random_gain)
            
            print(f"Moving to position {len(trajectories)-1-i} with randomized trajectory: {randomized_trajectory}")
            robot.servoj(randomized_trajectory, vel=vel_scale, acc=acc_scale, t=t_tmp, lookahead_time=0.2, gain=100, wait=True)
            
        time.sleep(0.5)  # Pause between rounds

    robot.servoj(trajectories[0], vel=vel_scale, acc=acc_scale, t=t_tmp, lookahead_time=0.2, gain=100, wait=True)
    print('Finished moving the robot through all positions.')

if __name__ == '__main__':
    robot_left = urx.Robot("192.10.0.11")

    # Parse the trajectory from traj.txt
    trajectory = parse_trajectory("./traj_20250409.txt", shuffle=False)
    print("Parsed trajectory:", trajectory)

    joints_left = robot_left.getj()
    print('Left robot joints: ', joints_left)
    time.sleep(1)

    # Number of rounds
    n_rounds = 2  # Adjust as needed

    # Execute the trajectory
    execute_trajectory(robot_left, trajectory, n_rounds)

    print("Finished executing the trajectory.")
