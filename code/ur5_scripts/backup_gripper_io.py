# import serial


# if __name__ == '__main__':
#     comm_port = '/dev/serial/by-id/usb-ROBOTIQ_INC._I_O_Coupling_333535383038510400460032-if00'
#     try:
#         gripper = serial.Serial(port=str(comm_port),
#             baudrate=115200, bytesize=8,parity='N',stopbits=1,timeout=1)
#         gripper_connected = True
#     except:
#         print("Error connecting to gripper.")
#         gripper_connected = False
    
#     if gripper_connected:

import rospy
from ur_msgs.srv import SetIO, SetIORequest

def set_io(pin, state):
    rospy.wait_for_service('/ur_driver/set_io')
    breakpoint()
    try:
        set_io = rospy.ServiceProxy('/ur_driver/set_io', SetIO)
        req = SetIORequest()
        req.fun = 1  # For digital output; use 2 for analog
        req.pin = pin
        req.state = state
        resp = set_io(req)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == '__main__':
    rospy.init_node('set_io_example')
    pin = 2  # Example pin
    state = 1.0  # State to set: 1.0 for high, 0.0 for low
    success = set_io(pin, state)
    if success:
        rospy.loginfo("Successfully set pin %d to state %f", pin, state)
    else:
        rospy.loginfo("Failed to set pin %d", pin)
