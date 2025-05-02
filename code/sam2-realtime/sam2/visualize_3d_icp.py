import cv2
# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import icp_3d


f1_id = 397
f2_id = 398


def get_3Dmask_coords(pcd_dict):
    """
    Returns the 3D coordinates of the mask from the PCD dictionary.
    """
    mask3D_points = []
    for label, coord in zip(pcd_dict["group"], pcd_dict["coord"]):
        if label == 1:
            mask3D_points.append(coord)
    return np.array(mask3D_points)


def get_pcd(rgb_map, depth_map,  object_mask, extrinsic_params):
    depth_intrinsic = np.array([[892.2883911132812, 0, 643.8308715820312, 0],
                            [0, 892.2883911132812, 376.31494140625, 0],
                            [0, 0, 1, 0]])


    depth_img = depth_map.copy() #cv2.imread(depth, -1) # read 16bit grayscale image
    valid_depth_map = (depth_img != 0)
    color_image = rgb_map.copy() # cv2.imread(color)

    # cv2.imshow("1", depth_img)
    # print(depth_img)
    # cv2.imshow("2", valid_depth_map)

    color_image = np.reshape(color_image[valid_depth_map], [-1,3])
    group_ids = object_mask[valid_depth_map]
    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]

    # pose = np.loadtxt(pose)
    
    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
    
    # intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    extrinsic_fixed = np.linalg.inv(extrinsic_params)
    points_world = np.dot(points, np.transpose(extrinsic_fixed))

    save_dict = dict(coord=points_world[:,:3], color=colors, group=group_ids)

    return save_dict


depth_array = np.load("../../videos/new_recording_0324/record_gray_cylinder/raw_depth_frames.npy")
extrinsic = np.loadtxt("../../videos/new_recording_0324/extrinsic_params.txt")

FRAME_DIR = "../../videos/frames/"

frame1 = cv2.imread(f"{FRAME_DIR}/1_00{f1_id}.jpg", 0)
frame2 = cv2.imread(f"{FRAME_DIR}/1_00{f2_id}.jpg", 0)
dep_frame1 = depth_array[f1_id-1]
dep_frame2 = depth_array[f2_id-1]

print("Nonzero depth1 pixels:", np.count_nonzero(dep_frame1))
print("Nonzero depth2 pixels:", np.count_nonzero(dep_frame2))

ret, frame1 = cv2.threshold(frame1, 20, 1, cv2.THRESH_BINARY)
ret, frame2 = cv2.threshold(frame2, 20, 1, cv2.THRESH_BINARY)

# cv2.imshow("frame1", frame1)
# cv2.imshow("frame2", frame2)

# cv2.imshow("dframe1", dep_frame1)
# cv2.imshow("dframe2", dep_frame2)

cv2.waitKey(0)

pcd_frame1 = get_pcd(frame1, dep_frame1, frame1, extrinsic_params=extrinsic)
pcd_frame2 = get_pcd(frame2, dep_frame2, frame2, extrinsic_params=extrinsic)

# print(pcd_frame1)

print("Unique group labels in frame1:", np.unique(pcd_frame1["group"]))
print("Unique group labels in frame2:", np.unique(pcd_frame2["group"]))

mask3d_f1 = get_3Dmask_coords(pcd_frame1)
mask3d_f2 = get_3Dmask_coords(pcd_frame2)

M = icp_3d.icp_3d(source_mask=mask3d_f1, target_mask=mask3d_f2, mask_coords=True, max_iterations=100)

def apply_transformation(points, T):
    """Applies a 3x3 affine transformation matrix to 2D points."""
    # Convert 2D points to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack((points, ones))
    # print(points.shape)
    # print(homogeneous_points.shape)
    # print(homogeneous_points)
    
    # Apply transformation
    transformed_points = (T @ homogeneous_points.T).T
    
    # Convert back to 2D
    return transformed_points[:, :3]

def visualize_alignment(source_points, target_points, T):
    """Applies transformation and visualizes alignment."""
 
    
    # Transform source points
    transformed_points = apply_transformation(source_points, T)

    # Plot original source, transformed source, and target points
    plt.figure(figsize=(6, 8))
    plt.scatter(target_points[:, 1], target_points[:, 0], c='g', label="Target Edges", s=5)
    plt.scatter(source_points[:, 1], source_points[:, 0], c='r', label="Original Source", s=5)
    plt.scatter(transformed_points[:, 1], transformed_points[:, 0], c='b', label="Transformed Source", s=5, alpha=0.6)
    
    plt.legend()
    plt.gca().invert_yaxis()
    plt.title("ICP Alignment Verification")
    plt.show()

from mpl_toolkits.mplot3d import Axes3D

def visualize_alignment_3d(source_points, target_points, T):
    """Applies transformation and visualizes alignment in 3D."""
    
    # Transform source points using the ICP transformation matrix
    transformed_points = apply_transformation(source_points, T)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original source points (red)
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], 
               c='r', label="Original Source", s=5, alpha=0.5)

    # Plot the transformed source points (blue)
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], 
               c='b', label="Transformed Source", s=5, alpha=0.5)

    # Plot the target points (green)
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
               c='g', label="Target", s=5, alpha=0.5)

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D ICP Alignment")
    ax.legend()
    
    # Show the plot
    plt.show()

# Example usage
# visualize_alignment_3d(mask3d_f1, mask3d_f2, M)

import open3d as o3d

def visualize_open3d(source_points, target_points, transformed_points):
    """Visualizes 3D point clouds using Open3D."""

    # Source
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    source_pcd.paint_uniform_color([1, 0, 0])  # Red

    # Target
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    target_pcd.paint_uniform_color([0, 1, 0])  # Green

    # Transformed
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_pcd.paint_uniform_color([0, 0, 1])  # Blue

    o3d.visualization.draw_geometries([source_pcd, target_pcd, transformed_pcd])

# Example usage
transformed_mask3d_f1 = apply_transformation(mask3d_f1, M)
visualize_open3d(mask3d_f1, mask3d_f2, transformed_mask3d_f1)

# Example usage

# print(mask3d_f1.shape, mask3d_f2.shape, M.shape)
# visualize_alignment(mask3d_f1, mask3d_f2, M)


cv2.waitKey(0)
cv2.destroyAllWindows()



