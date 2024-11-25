import urx
import time

def parse_trajectory(file_path):
    trajectory = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(':')
            trajectory[key] = list(map(float, value.strip('[]').split(',')))
    return trajectory


def execute_trajectory(robot, trajectory, n):
    sequence = ["end", "end_up", "start_up", "start"]
    reverse_sequence = list(reversed(sequence))
    full_sequence = sequence + reverse_sequence

    for _ in range(n):
        for pose_name in full_sequence:
            print(f"Moving to {pose_name}", trajectory[pose_name])
            # for i in range(100):
            robot.servoj(trajectory[pose_name], vel=0.1, acc=0.1, t=2, lookahead_time=0.2, gain=100, wait=True)

        time.sleep(1)  # Pause between rounds

    print('Finished moving the left arm to starting states.')

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
