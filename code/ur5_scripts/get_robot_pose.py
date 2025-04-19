import urx_local

if __name__ == '__main__':
    robot_left = urx_local.Robot("192.10.0.11")
    
    # Get the complete pose (position and orientation)
    pose = robot_left.get_pose()
    print('Left robot pose: ', pose)
    
    # Extract the position (x, y, z coordinates)
    position = pose.pos
    print('End-effector position (x, y, z): ', position)
    
    # If you want individual coordinates
    x, y, z = position
    print(f'X: {x}, Y: {y}, Z: {z}')

    # Get the joint angles
    joint_angles = robot_left.getj()
    
    # Print the joint angles
    print('Current joint angles: ', joint_angles)

    # Close the connection to the robot
    robot_left.close()

