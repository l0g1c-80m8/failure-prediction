import icp_2d
import cv2
import numpy as np

import matplotlib.pyplot as plt


FRAME_DIR = "../../videos/frames/"

frame1 = cv2.imread(f"{FRAME_DIR}/2_00170.jpg", 0)
frame2 = cv2.imread(f"{FRAME_DIR}/2_00189.jpg", 0)


cv2.imshow("frame1", frame1)
cv2.imshow("frame2", frame2)

M = icp_2d.icp(frame1, frame2, show_mask=True)
print(M)




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
    return transformed_points[:, :2]

def visualize_alignment(source_mask, target_mask, T):
    """Applies transformation and visualizes alignment."""
    source_points = icp_2d.extract_points_from_mask(source_mask)
    target_points = icp_2d.extract_points_from_mask(target_mask)
    
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

# Example usage
visualize_alignment(frame1, frame2, M)

cv2.waitKey(0)
cv2.destroyAllWindows()