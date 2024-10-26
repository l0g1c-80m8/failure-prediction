# import sys
# from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

# if __name__ == '__main__':
#     robot_ip = "192.10.0.11"
#     rob = urx.Robot(robot_ip)
#     robotiqgrip = Robotiq_Two_Finger_Gripper(rob, socket_host="192.168.1.11")

#     robotiqgrip.close_gripper()
#     # robotiqgrip.gripper_action(255)
#     # robotiqgrip.open_gripper()

#     # rob.send_program(robotiqgrip.ret_program_to_run())
#     # rob.close()
#     # sys.exit()
#     print('Done!')


import sys
from urx import Robot
import time

if __name__ == '__main__':
    robot_ip = "192.10.0.11" # left arm
    # robot_ip = "192.10.0.12" # right arm
    robot = Robot(robot_ip)
    print('Robot pose: ', robot.get_pose())
    print('Sending activation sequence...')
    robot.set_tool_voltage(24)
    robot.set_digital_out(8, False)
    robot.set_digital_out(9, False)
    time.sleep(0.05)

    robot.set_digital_out(8, True)
    time.sleep(0.05)

    robot.set_digital_out(8, False)
    time.sleep(0.05)

    robot.set_digital_out(9, True)
    time.sleep(0.05)

    robot.set_digital_out(9, False)
    time.sleep(0.05)
    print('Finish sending the activation sequence.')

    # waiting for user's input
    while True:
        try:
            user_input = input("Enter 1 to open and 0 to close or exit to exit the program.")
            if user_input.lower() == 'exit':  # Allow the user to exit the loop
                print("Exiting the program.")
                break  # Break the loop to exit the program
            
            number = int(user_input)  # Convert the input to an integer
            if number == 1:
                # open gripper
                print('open gripper')
                robot.set_digital_out(8, False)
                robot.set_digital_out(9, False)
                time.sleep(0.05)
            else:
                # close gripper
                print('close gripper')
                # 50%
                robot.set_digital_out(8, True)
                robot.set_digital_out(9, False)

                # 100%
                # robot.set_digital_out(8, True)
                # robot.set_digital_out(9, True)
                time.sleep(0.05)
            # print digital outs
            dout0 = robot.get_digital_out(8)
            dout1 = robot.get_digital_out(9)
            print(f'Dout0 {dout0} Dout1 {dout1}')
        except ValueError:  # Handle the error if the input is not an integer
            print("Please enter a valid number.")
    print('Done!')