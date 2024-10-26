## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt 
import urx_local
import math3d as m3d
import time
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

def capture_rgb_and_depth_from_realsense():
    """
    Taken from https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    # this line fixes this issue: https://github.com/IntelRealSense/librealsense/issues/6628
    device.hardware_reset()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)   # USB 2.0
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)    # USB 3.0

    # Start streaming
    pipeline_cfg = pipeline.start(config)

    # get camera intrinsics: https://github.com/IntelRealSense/librealsense/issues/869
    profile = pipeline_cfg.get_stream(rs.stream.depth)              # Fetch stream profile for depth stream
    intrinsics = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

    # get depth_scale: https://github.com/IntelRealSense/librealsense/issues/3473
    depth_sensor = pipeline_cfg.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # Stop streaming
        pipeline.stop()

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return color_image, depth_image, intrinsics, depth_scale


def visualize_point_cloud(rgb, depth, intrinsics, depth_scale):
    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1000, convert_rgb_to_intensity=False)
    # plt.imshow(rgbd_image.color); plt.show() # debugging: make sure rgbd image looks reasonable
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy))
    
    # this line is for debugging; it does not have editing ability to add markers
    # NOTE: you can get the camera's information by pressing ctrl + c to copy the current camera information in o3d viewer
    # o3d.visualization.draw_geometries([pcd],
    #                                 zoom=0.47999999999999976,
    #                                 front=[ 0.1172293277274971, 0.094658051301072965, -0.98858339964033515 ],
    #                                 lookat=[ -4.2709908435416708e-05, -1.838764854445177e-05, 0.000339248996169772 ],
    #                                 up=[ 0.007463149155373737, -0.99550299721168745, -0.094435607411763683 ])

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=1280, height=720, left=5, top=5)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_lookat([ -3.1044226837025207e-05, -2.914392155758258e-05, 0.00034436146553498567])
    ctr.set_front([ 0.011940780017795843, 0.076320815640134113, -0.99701181079894508 ])
    ctr.set_up([ 0.017836457145755921, -0.99694051288372565, -0.076101738269383337 ])
    ctr.set_zoom(0.02)
    vis.run()  # user picks points
    vis.destroy_window()
    picked_points = np.asarray(pcd.points)[vis.get_picked_points()]

    # [left to right, height with respect to the camera (below camera is positive, above camera is negative), back to front] in meters
    last_picked_point = picked_points[-1]
    print('Last picked point position: ', last_picked_point)
    return last_picked_point

def move_robot_to_last_picked_point(camera_point):
    acceleration = 0.2
    velocity = 0.05
    
    # uncomment these to control the left robot arm
    # robot = urx_local.Robot("192.10.0.11") # left robot arm
    # translation = [0.14664, -0.38822, 0.37781] # x y z
    # rotation_quat = [-0.75076, 0.17819, -0.19165, 0.60652] # (x, y, z, w)
   

    # uncomment these to control the right robot arm
    robot = urx_local.Robot("192.10.0.12") # right robot arm
    translation = [-1.0403, -0.44696, 0.39837] # x y z
    rotation_quat = [-0.75321, 0.1632, -0.15168, 0.6189] # (x, y, z, w)


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

    # append 1 to point: [x, y, z, 1]
    camera_point = [camera_point[0], camera_point[1], camera_point[2], 1]

    robot_tcp_pos = robot.getl()[:3]
    print('Robot tcp postion: ', robot_tcp_pos)

    cam_to_base_to_tcp_point = np.matmul(camera_to_robot_base_trans_matrix, camera_point)
    print('cam_to_base_to_tcp_point: ', cam_to_base_to_tcp_point)

    # use current robot's z coordinate to avoid collision with the table.
    robot_tcp = np.array(robot_tcp_pos)
    cam_to_base_to_tcp_point[2] = robot_tcp[2]
    print('Final camera-to-base-to-tcp point: ', cam_to_base_to_tcp_point)
    delta_movement_based_on_tcp = cam_to_base_to_tcp_point[:3] - robot_tcp
    print('delta_movement_based_on_tcp: ', delta_movement_based_on_tcp)

    print('Check everything before executing on the robot!!!')
    breakpoint() # Pause before executing
    robot.translate((delta_movement_based_on_tcp[0], delta_movement_based_on_tcp[1], delta_movement_based_on_tcp[2]), acceleration, velocity)
    print('Done!!')
    exit(0)

if __name__ == '__main__':
    rgb, depth, intrinsics, depth_scale = capture_rgb_and_depth_from_realsense()
    last_picked_point = visualize_point_cloud(rgb, depth, intrinsics, depth_scale)
    move_robot_to_last_picked_point(last_picked_point)
