import urx_local
import time
import numpy as np

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
    return [qx, qy, qz, qw]

def quaternion_to_euler(w, x, y, z):
    # Calculate the roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x**2 + y**2)
    roll_x = np.arctan2(t0, t1)

    # Calculate the pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    # Calculate the yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y**2 + z**2)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def print_tool_tip_pos_orientation(robot):
    # tip_pos.x, tip_pos.y, tip_pos.z
    tip_pos = robot.get_pos()
    tip_orientation = robot.get_orientation()
    euler_angle = tip_orientation.to_euler('ZYX') # roll-pitch-yaw
    quat = get_quaternion_from_euler(euler_angle[0], euler_angle[1], euler_angle[2])
    converted_euler = quaternion_to_euler(quat[-1], quat[0], quat[1], quat[2])
    assert np.allclose(np.array(euler_angle), np.array(converted_euler), atol=1e-6)  # This will return True if `a` and `b` are close within the given tolerance

    converted_tip_orientation = tip_orientation.new_from_euler(converted_euler, 'ZYX')
    # note that tip_orientation and converted_tip_orientation should be exactly the same
    print(f'Tool tip position: {tip_pos}, orientation: {tip_orientation}')
    print('Getl: ', robot.getl())

if __name__ == '__main__':
    # robot = urx_local.Robot("192.10.0.11") # left arm
    robot = urx_local.Robot("192.10.0.12") # right arm
    print_tool_tip_pos_orientation(robot)
    time.sleep(1)
    print('Done')