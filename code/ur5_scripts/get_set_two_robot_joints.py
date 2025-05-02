import urx
import time

if __name__ == '__main__':
    robot_left = urx.Robot("192.10.0.11")
    joints_left = robot_left.getj()
    print('Left robot joints: ', joints_left)
    time.sleep(1)

    # robot_right = urx.Robot("192.10.0.12")
    # joints_right = robot_right.getj()
    # print('Right robot joints: ', joints_right)

    # starting states for our robots in GELLO and real-world experiments
    starting_robot_joints_left = [0.15057017881446877, -1.2067810306139042, 2.5172985853857583, -4.464267983475822, -1.6961607267145835, -0.031529356942651354]
    # starting_robot_joints_left = [-0.4576745003503593, -1.2852380311658416, -1.9249888475127885, -3.0619757909783183, -2.0758073576539102, 3.143600922204323]
    # starting_robot_joints_right = [-1.6545815648737472, -1.6381940802802397, 1.795800360316563, -1.7138684258221923, -1.7362808437379504, -0.01120622072583366]

    # for i in range(50):
    robot_left.servoj(
        starting_robot_joints_left, vel=0.1, acc=0.1, t=5, lookahead_time=0.2, gain=100, wait=False
    )
    time.sleep(1)
    print('Finished moving the left arm to starting states.')

    # for i in range(100):
    #     robot_right.servoj(
    #         starting_robot_joints_right, vel=0.1, acc=0.3, t=0.35, lookahead_time=0.2, gain=100, wait=False
    #     )
    # time.sleep(1)
    # print('Finished moving the right arm to starting states.')
    # print('Done')
