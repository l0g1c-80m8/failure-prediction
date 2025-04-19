"""
Author: Gayathri Rajesh
"""

import urx_local
import threading
import math3d as m3d
import time
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import numpy as np

def left_arm():
    robot = urx_local.Robot("192.10.0.11")
    trans = robot.get_pose()
    print('Robot pose left: ', trans)
    # trans.pos.z += -0.05
    camera_point = np.array([-0.1168656,   0.15222021,  0.83700001])
    camera_destination = np.array([0.22979144, 0.18172226, 0.81900001])
    #final point [0.22979144 0.18172226 0.81900001]


    move_robot_to_last_picked_point(camera_point, camera_destination, robot, "left")
    
    # robot.set_pose(trans, acc=0.1, vel=0.1)
    return robot

def right_arm():
    robot = urx_local.Robot("192.10.0.12")
    trans = robot.get_pose()
    print('Robot pose right: ', trans)
    # trans.pos.z += -0.05
    # robot.set_pose(trans, acc=0.1, vel=0.1)

    #final point 0.1610227  -0.15659803  0.83499998
    camera_point = np.array([-0.1255439,  -0.1227893,   0.62400001])
    camera_destination = np.array([0.1610227,  -0.15659803,  0.83499998])
    move_robot_to_last_picked_point(camera_point, camera_destination, robot, "right")
    
    #0.25107363 -0.2008733   0.634
    return robot

def move_robot_to_last_picked_point(camera_point, camera_destination, robot, arm):
    acceleration = 0.2
    velocity = 0.05

    if arm == "left":
        # uncomment these to control the left robot arm
        # left robot arm
        translation = [0.55266, 0.065468, 0.825] # x y z
        rotation_quat = [0.73081, -0.68209, -0.00063677, -0.02574] # (x, y, z, w)
        camera_destination[2] = camera_destination[2] - 0.1
    
    else:
        # uncomment these to control the right robot arm
        translation = [-0.60467, 0.052382, 0.6418] # x y z
        rotation_quat = [0.73724, -0.67559, -0.005377, 0.0051025] # (x, y, z, w)
        camera_destination[2] = camera_destination[2] - 0.3
   
    ############## Method 1: use translation and rotation_quat ############## 
    # (w, x, y, z)
    rotation_quat = Quaternion(rotation_quat[-1], rotation_quat[0], rotation_quat[1], rotation_quat[2])
    camera_to_robot_base_trans_matrix = rotation_quat.transformation_matrix

    # this is how we use the result obtained from handeye calibration: https://support.zivid.com/en/latest/academy/applications/hand-eye/how-to-use-the-result-of-hand-eye-calibration.html
    # assign translation values
    camera_to_robot_base_trans_matrix[0][-1] = translation[0]
    camera_to_robot_base_trans_matrix[1][-1] = translation[1]
    camera_to_robot_base_trans_matrix[2][-1] = translation[2]

    ############## Method 2: use the transformation matrix directly ############## 
    # camera_to_robot_base_trans_matrix = np.array([
    #     [0.990668, -0.0666395,   0.118892,   0.582191],
    #     [-0.13553,  -0.389404,   0.911041,  -0.576265],
    #     [-0.0144143,  -0.918653,  -0.394802,     0.3589],
    #     [        0,          0,          0,          1]
    # ])

    #append 1 to point: [x, y, z, 1]
    camera_point = [camera_point[0], camera_point[1], camera_point[2], 1]

    robot_tcp_pos = robot.getl()[:3]
    print('Robot tcp postion: ', robot_tcp_pos)

    cam_to_base_to_tcp_point = np.matmul(camera_to_robot_base_trans_matrix, camera_point)
    print('cam_to_base_to_tcp_point: ', cam_to_base_to_tcp_point)

    # use current robot's z coordinate to avoid collision with the table.
    robot_tcp = np.array(robot_tcp_pos)
    #cam_to_base_to_tcp_point[2] = robot_tcp[2]
    print('Final camera-to-base-to-tcp point: ', cam_to_base_to_tcp_point)
    delta_movement_based_on_tcp = cam_to_base_to_tcp_point[:3] - robot_tcp
    print('delta_movement_based_on_tcp: ', delta_movement_based_on_tcp)

    print('Check everything before executing on the robot!!!')
    #breakpoint() # Pause before executing
    robot.translate((delta_movement_based_on_tcp[0], delta_movement_based_on_tcp[1], delta_movement_based_on_tcp[2]), acceleration, velocity)
    robot.set_tool_voltage(24)

    #Open gripper
    robot.set_digital_out(8, False)
    robot.set_digital_out(9, False)
    time.sleep(0.05)

    #Close gripper
    robot.set_digital_out(8, True)
    robot.set_digital_out(9, False)

    time.sleep(0.05)

    robot_tcp_pos_2 = robot.getl()[:3]
    print('Robot tcp postion: ', robot_tcp_pos_2)

    camera_destination = [camera_destination[0], camera_destination[1], camera_destination[2], 1]

    cam_to_base_to_tcp_point_2 = np.matmul(camera_to_robot_base_trans_matrix, camera_destination)
    print('cam_to_base_to_tcp_point: ', cam_to_base_to_tcp_point_2)

    # use current robot's z coordinate to avoid collision with the table.
    robot_tcp_2 = np.array(robot_tcp_pos_2)
    #cam_to_base_to_tcp_point[2] = robot_tcp[2]
    print('Final camera-to-base-to-tcp point: ', cam_to_base_to_tcp_point_2)
    delta_movement_based_on_tcp_2 = cam_to_base_to_tcp_point_2[:3] - robot_tcp_2
    print('delta_movement_based_on_tcp: ', delta_movement_based_on_tcp_2)

    print('Check everything before executing on the robot!!!')
    #breakpoint() # Pause before executing
    robot.translate((delta_movement_based_on_tcp_2[0], delta_movement_based_on_tcp_2[1], delta_movement_based_on_tcp_2[2]), acceleration, velocity)

    #Open gripper
    robot.set_digital_out(8, False)
    robot.set_digital_out(9, False)
    time.sleep(0.05)

    print('Done!!')
    exit(0)


if __name__ == "__main__":
    t1 = threading.Thread(target = left_arm)
    t2 = threading.Thread(target = right_arm)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Done!")