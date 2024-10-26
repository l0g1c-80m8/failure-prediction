import urx

def move_to_target(robot, target_positions, tolerance, acceleration, max_speed, Kp):
    # Get current joint positions
    current_positions = robot.getj()

    # While loop to adjust speeds until target is reached
    while not all(abs(current_positions[i] - target_positions[i]) < tolerance for i in range(6)):
        # Calculate speed for each joint based on proportional control
        speeds = [Kp * (target_positions[i] - current_positions[i]) for i in range(6)]
        
        # Limit speeds to max_speed
        speeds = [max(min(speed, max_speed), -max_speed) for speed in speeds]
        
        # Apply the calculated speeds
        robot.speedj(speeds, acceleration, 0.1)  # apply for a short time frame
        
        # Update current positions
        current_positions = robot.getj()

    # Stop all joints smoothly
    robot.speedj([0, 0, 0, 0, 0, 0], acceleration, 0.5)
    print('Done!')


if __name__ == '__main__':
    robot = urx.Robot("192.10.0.11")
    joints = robot.getj()
    print('Robot joints before moving: ', joints)
    joints[2] -= 0.1
    print('Robot target joints: ', joints)


    velocity = 0.5
    acceleration = 0.5
    # dt = 1.0 / 500  # 2ms
    dt = 1.0 / 1000 
    lookahead_time = 0.2
    gain = 100

    # example 1: using servoj to move to target joint positions
    robot.set_freedrive(True)

    # Get current joint positions
    current_positions = robot.getj()
    tolerance = 0.08

    # While loop to adjust speeds until target is reached
    while not all(abs(current_positions[i] - joints[i]) < tolerance for i in range(6)):
        robot.servoj(
            joints, vel=velocity, acc=acceleration, t=0.5, lookahead_time=lookahead_time, gain=gain, wait=False
        )
        current_positions = robot.getj()
    print('Done!!!')

    # example 2: use movej to move to target joint positions
    # robot.movej(joints, acc=1, vel=0.1) 

    # example 3: use speedj to move to target joint positions
    # move_to_target(robot, joints, tolerance=0.01, acceleration=0.2, max_speed=0.5, Kp=10)

    