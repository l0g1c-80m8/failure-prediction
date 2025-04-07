import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.linalg import svd
import datetime

def extract_points_from_mask(mask, show=False):
    """Extracts edge points from a binary mask."""
    
    mask = cv2.medianBlur(mask, 7)
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 100, L2gradient = True)
    if show:
        cv2.imshow(f"{datetime.datetime.now()}", edges)
    points = np.column_stack(np.where(edges > 0))

    # print(len(points))
    return points

def best_fit_transform(A, B):
    """Computes the best-fit affine transformation (rotation + translation)."""
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
    
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T

def icp(source_mask, target_mask, max_iterations=50, tolerance=1e-5, show_mask=False):
    """Performs ICP to align source_mask to target_mask and returns a 3x3 transformation matrix."""
    source_points = extract_points_from_mask(source_mask, show=show_mask)
    target_points = extract_points_from_mask(target_mask, show=show_mask)
    
    prev_error = float('inf')
    T = np.eye(3)

    if len(target_points) == 0:
        return np.zeros((3,3))
    
    for i in range(max_iterations):
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)
        matched_target_points = target_points[indices]
        
        T_iter = best_fit_transform(source_points, matched_target_points)
        
        source_points = (T_iter[:2, :2] @ source_points.T).T + T_iter[:2, 2]
        T = T_iter @ T
        
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    return T
