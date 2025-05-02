from pyModbusTCP.client import ModbusClient
import time
import numpy as np 

TCP_x = 400
TCP_y = 401
TCP_z = 402
JOINT_1 = 270
JOINT_2 = 271
JOINT_3 = 272
JOINT_4 = 273
JOINT_5 = 274
JOINT_6 = 275


# def unsigned(a):
#     if a > 32767:
#         a = a - 65535
#     else:
#         a = a
#     return a

def unsigned(a):
    a[a > 32767] = a[a > 32767] - 65536
    return a

def signed(a):
    return np.mod(a, 65536)

def convert_to_regular_joint_metric(j):
    return j / 1000

def convert_to_modbus_joint_metric(j):
    return (j * 1000).astype(np.uintc)

try:
    #connect via modbus
    robot = ModbusClient(host="192.10.0.11", port=502, auto_open=True, debug=False)
    print("connected")
except:
    print("Error with host or port params")

i = 0 
while True:
    # Read the value of the specified register from the UR10
    # register_value = robot.read_holding_registers(TCP_x)
    # reg_var = unsigned(register_value[0])
    # print("TCP-X:",reg_var)

    # register_value = robot.read_holding_registers(TCP_y)
    # reg_var = unsigned(register_value[0])
    # print("TCP-Y:",reg_var)

    # register_value = robot.read_holding_registers(TCP_z)
    # reg_var = unsigned(register_value[0])
    # print("TCP-Z:",reg_var)

    joint_num = 1
    current_joints = []
    # for joint_port in [JOINT_1, JOINT_2, JOINT_3, JOINT_4, JOINT_5, JOINT_6]:

    register_values = robot.read_holding_registers(JOINT_1, 6)
    print('Modbus joint values (current): ', register_values)
    register_values = np.array(register_values)
    register_values = unsigned(register_values)
    current_joints = convert_to_regular_joint_metric(register_values)
    print(f"Joints: {current_joints}")

    # modify joint 3
    current_joints[2] -= 0.1

    current_joints = convert_to_modbus_joint_metric(current_joints)
    current_joints = signed(current_joints).tolist()
    print('Modbus joint values (after): ', current_joints)

    # if robot.write_multiple_registers(JOINT_1, current_joints):
    if robot.write_single_register(JOINT_3, current_joints[2]):
        print('Successfully wrote regigters')
    else:
        print('Failed to write registers')


    # if robot.write_single_register(128, i):
    #     print("write ok", i)
    # else:
    #     print("write error")


    break
    time.sleep(1)
    i+=1
    print("--------"*2)


print('Done!')