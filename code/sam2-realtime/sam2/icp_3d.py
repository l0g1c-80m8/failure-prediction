import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.linalg import svd

def extract_points_from_mask(mask):
    """Extracts nonzero 3D points (x, y, z) from a binary mask."""
    points = np.column_stack(np.where(mask > 0))  # Extract nonzero voxel indices
    return points.astype(np.float32)

def best_fit_transform(A, B):
    """Computes the best-fit affine transformation (rotation + translation) for 3D points."""
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered
    U, _, Vt = svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    # Construct 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def icp_3d(source_mask, target_mask, max_iterations=50, tolerance=1e-5, mask_coords=False):
    """Performs ICP to align source_mask to target_mask and returns a 4x4 transformation matrix."""

    if not mask_coords:
        source_points = extract_points_from_mask(source_mask)
        target_points = extract_points_from_mask(target_mask)
    else:
        source_points = source_mask
        target_points = target_mask

    prev_error = float('inf')
    T = np.eye(4)

    if len(target_points) == 0:
        return np.zeros((4, 4))

    for _ in range(max_iterations):
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)
        matched_target_points = target_points[indices]

        T_iter = best_fit_transform(source_points, matched_target_points)

        # Apply transformation
        source_points_hom = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
        source_points = (T_iter @ source_points_hom.T).T[:, :3]  # Convert back to 3D

        T = T_iter @ T  # Accumulate transformations

        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return T