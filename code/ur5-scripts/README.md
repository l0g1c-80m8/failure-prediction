CB2 UR5s Setup Instructions: https://i-chun-arthur.notion.site/CB2-UR5-Manual-eb8bd315645e422bb441105c421b8e22

The scripts control the UR5s using the [python-urx](https://github.com/SintefManufacturing/python-urx) library. 
* More examples: https://github.com/SintefManufacturing/python-urx.
* A local version of `urx` is used because there's a formatting issue with `tpose`.

**`test_robotiq_gripper.py`**

- Activate and control the Robotiq gripper. Note that every time you start a program you need to activate the gripper to use it.

**`opencv_viewer_example.py`**

- Stream Intel RealSense camera.

**`get_robot_tool_tip_pos.py`**

- Get robot’s tool tip position (x, y, z) in robot base frame.

**`get_set_two_robot_joints.py`**

- Move the robot arms to pre-configured poses. I use this code to move the robot arms to the starting states before my experiments.

**`verify_camera_calibration.py`**

- I use this code to test my camera calibration result. Specifically, it captures RGB-D images from the RealSense camera and display point cloud in Open3D. Then, I select a point in the viewer, and the robot arm would move its end-effector to that position.

**`servoj_example.py`**

- Example code that moves the robot arm using joint-space control.

**`simultaneous_motion.py`**

- Example code that executes both arms simultaneously.

**`run a container:`**

```
sudo docker run -it --device /dev/tty1 --device /dev/input --privileged -v /etc/localtime:/etc/localtime:ro -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --shm-size 8g -e GDK_SCALE -e GDK_DPI_SCALE --network host  --ipc=host  -v /home/:/home --name {container_name} {image_id}
```



