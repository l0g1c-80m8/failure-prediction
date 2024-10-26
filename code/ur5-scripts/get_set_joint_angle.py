import urx
import keyboard
import time

def move_robot(robot, joint_angle, joint_idx):
    """Move the robot to the specified joint angle for joint 0."""
    joint_angles = robot.getj()  # Get current joint angles
    joint_angles[joint_idx] = joint_angle  # Update only joint 0
    robot.movej(joint_angles, acc=0.3, vel=0.1)

if __name__ == '__main__':
    # Connect to the robot
    robot = urx.Robot("192.10.0.11")
    
    # Get initial joint angles
    current_angles = robot.getj()
    print('Initial joint angles:', current_angles)

    print("Use the following keys to control joint 0:")
    print("W: Increase joint 0 (+0.1 rad)")
    print("S: Decrease joint 0 (-0.1 rad)")
    print("E: Increase joint 1 (+0.1 rad)")
    print("D: Decrease joint 1 (-0.1 rad)")
    print("R: Increase joint 2 (+0.1 rad)")
    print("F: Decrease joint 2 (-0.1 rad)")
    print("T: Increase joint 3 (+0.1 rad)")
    print("G: Decrease joint 3 (-0.1 rad)")
    print("Y: Increase joint 4 (+0.1 rad)")
    print("H: Decrease joint 4 (-0.1 rad)")
    print("U: Increase joint 5 (+0.1 rad)")
    print("J: Decrease joint 5 (-0.1 rad)")
    print("Q: Quit")
    
    try:
        while True:
            joint_idx = -1
            if keyboard.is_pressed('w'):
                joint_idx = 0
                current_angles[joint_idx] += 0.01  # Increase joint 0

            elif keyboard.is_pressed('s'):
                joint_idx = 0
                current_angles[joint_idx] -= 0.01  # Decrease joint 0

            elif keyboard.is_pressed('e'):
                joint_idx = 1
                current_angles[joint_idx] += 0.01  # Decrease joint 0
            
            elif keyboard.is_pressed('d'):
                joint_idx = 1
                current_angles[joint_idx] -= 0.01  # Decrease joint 0

            elif keyboard.is_pressed('r'):
                joint_idx = 2
                current_angles[joint_idx] += 0.01  # Decrease joint 0
            
            elif keyboard.is_pressed('f'):
                joint_idx = 2
                current_angles[joint_idx] -= 0.01  # Decrease joint 0

            elif keyboard.is_pressed('t'):
                joint_idx = 3
                current_angles[joint_idx] += 0.01  # Decrease joint 0
            
            elif keyboard.is_pressed('g'):
                joint_idx = 3
                current_angles[joint_idx] -= 0.01  # Decrease joint 0

            elif keyboard.is_pressed('y'):
                joint_idx = 4
                current_angles[joint_idx] += 0.01  # Decrease joint 0
            
            elif keyboard.is_pressed('h'):
                joint_idx = 4
                current_angles[joint_idx] -= 0.01  # Decrease joint 0

            elif keyboard.is_pressed('u'):
                joint_idx = 5
                current_angles[joint_idx] += 0.01  # Decrease joint 0
            
            elif keyboard.is_pressed('j'):
                joint_idx = 5
                current_angles[joint_idx] -= 0.01  # Decrease joint 0

            if not joint_idx == -1:
                move_robot(robot, current_angles[joint_idx], joint_idx)
                time.sleep(0.1)  # Delay to avoid too fast input
            elif keyboard.is_pressed('q'):
                break  # Exit the loop

    finally:
        # Ensure the robot connection is closed
        robot.close()
        print("Robot connection closed.")
