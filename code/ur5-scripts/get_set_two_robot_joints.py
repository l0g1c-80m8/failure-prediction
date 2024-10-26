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
    starting_robot_joints_left = [6.108484466475944, -0.8206405732561044, -2.292379400175988, -3.087181887648819, -0.3205311336615182, 2.7729978630161582]
    # starting_robot_joints_left = [1.3780459778509433, -1.523749890442781, -1.5899893346228173, -1.523211104456275, 1.722199931031265, -0.20266514321065365]
    # starting_robot_joints_right = [-1.6545815648737472, -1.6381940802802397, 1.795800360316563, -1.7138684258221923, -1.7362808437379504, -0.01120622072583366]

    for i in range(100):
        robot_left.servoj(
            starting_robot_joints_left, vel=0.1, acc=0.3, t=0.35, lookahead_time=0.2, gain=100, wait=False
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
