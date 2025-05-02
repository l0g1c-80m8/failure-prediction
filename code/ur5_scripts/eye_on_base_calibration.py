import urx_local
import math3d as m3d
import time
import math
import copy

def print_tool_tip_pos_orientation(robot):
    tip_pos = robot.get_pos()
    tip_orientation = robot.get_orientation()
    print(f'Tool tip position: {tip_pos}, orientation: {tip_orientation}')

def move_to_new_pose(robot, pose, acceleration, velocity):
    print_tool_tip_pos_orientation(robot)
    input(f"Press enter to move to a new pose {pose}")
    robot.movel(pose, acceleration, velocity)
    time.sleep(1)
    print('Done!')


"""
I used pendent's "Move Tool" and "Move Joints" and get_robot_pose.py to determine the robot's pose.
I found that the “Move Tool” option on the pendent is very useful in moving the robot arm to various poses since it’s less likely to run into joint limits or acceleration/deacceleration issues compared to “Move Joints”. 
Remember to launch easy_handeye to check if the AruCo marker can be detected.
"""
if __name__ == '__main__':
    robot = urx_local.Robot("192.10.0.11")
    acceleration = 0.1
    velocity = 0.05

    home_pos = m3d.Transform((1.318, -0.570, 0.684), (0.519, -0.173, 0.591))
    move_to_new_pose(robot, home_pos, acceleration, velocity)

    pose = m3d.Transform((1.451, -0.052, -0.019), (0.496, -0.185, 0.591))
    move_to_new_pose(robot, pose, acceleration, velocity)

    pose = m3d.Transform((1.406, 0.457, -0.588), (0.496, -0.185, 0.522))
    move_to_new_pose(robot, pose, acceleration, velocity)

    pose = m3d.Transform((1.407, 0.456, -0.587), (0.496, -0.185, 0.470))
    move_to_new_pose(robot, pose, acceleration, velocity)

    pose = m3d.Transform((1.267, 0.709, -0.988), (0.419, -0.184, 0.476))
    move_to_new_pose(robot, pose, acceleration, velocity)

    pose = m3d.Transform((0.688, 1.597, -2.000), (0.419, -0.184, 0.608))
    move_to_new_pose(robot, pose, acceleration, velocity)

    pose = m3d.Transform((0.687, 1.597, -2.000), (0.419, -0.255, 0.608))
    move_to_new_pose(robot, pose, acceleration, velocity)

    pose = m3d.Transform((0.687, 1.598, -2.000), (0.418, -0.044, 0.608))
    move_to_new_pose(robot, pose, acceleration, velocity)
    
    pose = m3d.Transform((0.345, 1.590, -1.717), (0.418, -0.044, 0.608))
    move_to_new_pose(robot, pose, acceleration, velocity)

    # pose = m3d.Transform((-0.078, 1.248, -2.201), (0.419, -0.044, 0.606))
    # move_to_new_pose(robot, pose, acceleration, velocity)

    # pose = m3d.Transform((0.077, 2.114, -1.547), (0.422, -0.043, 0.572))
    # move_to_new_pose(robot, pose, acceleration, velocity)

    pose = m3d.Transform((-0.086, 1.757, -1.526), (0.416, 0.070, 0.600))
    move_to_new_pose(robot, pose, acceleration, velocity)

    # pose = m3d.Transform((0.563, 1.886, -1.984), (0.368, -0.207, 0.600))
    # move_to_new_pose(robot, pose, acceleration, velocity)

    pose = m3d.Transform((0.474, 1.804, -1.786), (0.425, -0.130, 0.663))
    move_to_new_pose(robot, pose, acceleration, velocity)

    # pose = m3d.Transform((0.393, 2.131, -1.566), (0.432, -0.126, 0.632))
    # move_to_new_pose(robot, pose, acceleration, velocity)

    # go back to picking pose
    # pick_pose = m3d.Transform((0.289, 2.910, 0.048),(0.434, -0.047, 0.572))
    # move_to_new_pose(robot, pick_pose, acceleration, velocity)
    