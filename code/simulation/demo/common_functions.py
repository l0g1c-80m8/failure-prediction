import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
from scipy.spatial.distance import cdist

import json


# Define different interpolation methods
def flat_interpolation(first_failure_time_step, failure_time_step_trim):
    # Calculate number of points needed
    num_points = first_failure_time_step - failure_time_step_trim + 1
    # Return array of constant value 0.5 with the required length
    return np.full(num_points, 0.5)

def linear_interpolation(first_failure_time_step, failure_time_step_trim):
    return np.linspace(0, 1, first_failure_time_step - failure_time_step_trim + 1)

def sin_interpolation(first_failure_time_step, failure_time_step_trim):
    x = np.linspace(0, 1, first_failure_time_step - failure_time_step_trim + 1)
    return np.sin(x)

def find_closest_points(src_points, dst_points):
    """
    Find closest point pairs between source and destination point sets.
    
    Args:
        src_points: (N, 2) array of source points
        dst_points: (M, 2) array of destination points
        
    Returns:
        Tuple of matched points arrays, both of shape (K, 2)
    """
    # Calculate pairwise distances between all points
    distances = cdist(src_points, dst_points)
    
    # Find closest destination point for each source point
    closest_indices = np.argmin(distances, axis=1)
    
    # Create matched pairs
    src_matched = src_points
    dst_matched = dst_points[closest_indices]
    
    return src_matched, dst_matched


def extract_points_from_mask(mask, show=False):
    """Extracts edge points from a binary mask."""
    
    mask = cv2.medianBlur(mask, 7)
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 100, L2gradient = True)
    if show:
        cv2.imshow(f"{datetime.datetime.now()}", edges)
    points = np.column_stack(np.where(edges > 0))

    # print(len(points))
    return points

def calculate_transformation(src_points, dst_points):
    """
    Calculate rigid transformation (R, t) between matched point sets.
    
    Args:
        src_points: (N, 2) array of source points
        dst_points: (N, 2) array of destination points
        
    Returns:
        R: 2x2 rotation matrix
        t: 2D translation vector
    """
    # Calculate centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)
    
    # Center the point sets
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    # Calculate covariance matrix
    H = src_centered.T @ dst_centered
    
    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Calculate rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Calculate translation
    t = dst_centroid - (R @ src_centroid)
    
    return R, t

def create_homogeneous_matrix(R, t):
    """
    Create a 3x3 homogeneous transformation matrix from rotation R and translation t.
    
    Args:
        R: 2x2 rotation matrix
        t: 2D translation vector
        
    Returns:
        3x3 homogeneous transformation matrix
    """
    H = np.eye(3)
    H[:2, :2] = R
    H[:2, 2] = t
    return H

def icp_2d(src_contour, dst_contour, max_iterations=16, tolerance=1e-6, matrix=False):
    """
    Perform 2D ICP algorithm between two contours.
    
    Args:
        src_contour: Source contour points from OpenCV findContours
        dst_contour: Destination contour points from OpenCV findContours
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for mean squared error
        matrix: return R matrix if true, otherwise return rotation angle
        
    Returns:
        rotation_angle: Rotation angle in radians
        translation: 2D translation vector [tx, ty]
        error: Final mean squared error
    """
    # Convert contours to point arrays
    src_points = src_contour.reshape(-1, 2).astype(np.float32)
    dst_points = dst_contour.reshape(-1, 2).astype(np.float32)
    
    # Initialize transformation
    R_total = np.eye(2)
    t_total = np.zeros(2)
    
    prev_error = float('inf')
    current_error = float('inf')  # Initialize current_error
    
    # Check if contours are empty or too small
    if len(src_points) < 2 or len(dst_points) < 2:
        return R_total if matrix else np.arctan2(R_total[1, 0], R_total[0, 0]), t_total, current_error
    
    for iteration in range(max_iterations):
        # Find closest point pairs
        src_matched, dst_matched = find_closest_points(src_points, dst_points)
        
        # Calculate transformation
        R, t = calculate_transformation(src_matched, dst_matched)
        
        # Update total transformation
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Apply transformation to source points
        src_points = (R @ src_points.T).T + t
        
        # Calculate error
        current_error = np.mean(np.sum((src_matched - dst_matched) ** 2, axis=1))
        
        # Check convergence
        if abs(prev_error - current_error) < tolerance:
            break
        
        prev_error = current_error

    # print ("R_total[1, 0], R_total[0, 0]", R_total[1, 0], R_total[0, 0])

    if matrix:
        # Apply log1p scaling to R_total and t_total
        scale_factor = 1e5
        R_total_scaled = np.sign(R_total) * np.log1p(np.abs(R_total) * scale_factor)
        t_total_scaled = np.sign(t_total) * np.log1p(np.abs(t_total) * scale_factor)
        
        # print("Original R_total:", R_total)
        # print("Scaled R_total:", R_total_scaled)
        # print("Original t_total:", t_total)
        # print("Scaled t_total:", t_total_scaled)
        

        # print("H with scaled values:", H)
        
        return R_total_scaled, t_total_scaled, current_error
    else:
        # Extract rotation angle from rotation matrix
        # For a 2D rotation matrix R = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        # The rotation angle can be calculated as θ = atan2(R[1,0], R[0,0])
        rotation_angle = np.arctan2(R_total[1, 0], R_total[0, 0])  # radians
        
        # Apply the same log1p scaling if needed
        # scale_factor = 1e5
        # rotation_angle_scaled = np.sign(rotation_angle) * np.log1p(np.abs(rotation_angle) * scale_factor)
        # t_total_scaled = np.sign(t_total) * np.log1p(np.abs(t_total) * scale_factor)
        
        # print ("rotation_angle, t_total", rotation_angle, t_total)

        return rotation_angle, t_total, current_error

def process_consecutive_frames(contours1, contours2, matrix=False):
    """
    Process consecutive frames and calculate transformations for each contour pair.
    
    Args:
        contours1: List of contours from first frame
        contours2: List of contours from second frame
        matrix: return R matrix if true, otherwise return rotation angle
        
    Returns:
        List of (R, t, error) tuples for each matched contour pair
    """
    results = []
    # print("contours1 shape", contours1.shape)
    
    # Calculate ICP between matched contours
    r_total, t_total, error = icp_2d(contours1[0], contours2[0], matrix=matrix)
    if matrix:
        # Create homogeneous transformation matrix with scaled values
        H = create_homogeneous_matrix(r_total, t_total)
        results=[H, error]
    else:
        results=[r_total, t_total[0], t_total[1]]

    try:
        # Check for empty transforms
        if len(results) == 0:
            raise Exception("All transforms are empty")
    except Exception as e:
        print(f"Error in contour processing: {e}")
    
    if matrix:
        features = extract_transform_features(results)
    else:
        features = results

    return features

def extract_transform_features(transforms):
    if not transforms:
        # Return zeros if no transforms
        return np.zeros(6)  # 4 for rotation matrix elements + 2 for translation
    
    # Take first transform if multiple are present
    H, error = transforms
    
    # Extract rotation matrix and translation vector
    R = H[:2, :2]  # 2x2 rotation matrix
    t = H[:2, 2]   # 2D translation vector
    
    # Convert to numpy arrays if needed
    R = np.array(R)
    t = np.array(t)
    
    # Ensure correct shapes before concatenating
    R_flat = R.flatten()  # Make 2x2 matrix into 1D array of length 4
    t = t.reshape(-1)    # Ensure translation is 1D array
    
    # print("R_flat", R_flat)
    # print("t", t)
    # Combine into single feature vector
    features = np.concatenate([R_flat, t])
    features = [features[0], features[1], features[4], features[5]]
    # print("features", features)
    
    return features

def process_camera_frame(frame, min_contour_area=100):
    """
    Process camera frame to get mask and contours for non-zero pixel values.
    
    Args:
        frame: RGB image array (height, width, 3)
        min_contour_area: 100  # Adjust this threshold as needed
    
    Returns:
        tuple: (mask, contours, filtered_image)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Create mask for non-zero pixels
    # Use a small threshold to filter out near-black pixels
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours (noise)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Create visualization of the filtered image
    filtered_image = frame.copy()
    
    # If we have valid contours, keep only the largest one
    if filtered_contours:
        # Find the contour with the largest area
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        # Use only this contour
        filtered_contours = [largest_contour]
        
        # Draw the contour on the filtered image
        cv2.drawContours(filtered_image, filtered_contours, -1, (0, 255, 0), 2)
    else:
        # If no valid contours found, create a default contour (small square)
        # default_contour = np.array([[[10, 10]], [[10, 20]], [[20, 20]], [[20, 10]]], dtype=np.int32)
        # filtered_contours = [default_contour]
        # cv2.drawContours(filtered_image, filtered_contours, -1, (0, 0, 255), 2)  # Red color for default
        print("No valid contours found")
    
    return mask, filtered_contours, filtered_image

# Function to calculate failure phase based on displacement, linear speed, and angular speed
def calculate_failure_phase(displacement, object_pos, panel_pos):
    """
    Calculate failure phase based on object position relative to panel.
    
    Args:
        displacement: Current displacement vector between object and panel
        object_pos: 3D position of the object (bunny)
        panel_pos: 3D position of the panel
        
    Returns:
        float: failure phase (0.0 = safe on panel, 1.0 = fallen off panel)
    """
    # print("displacement", displacement)
    # Check if object is below the panel surface (has fallen)
    if object_pos[2] < panel_pos[2] - 0.02:  # Small threshold to account for contact
        return 1.0
    
    # Check horizontal displacement (if object center is too far from panel center)
    # panel_radius = 0.15  # Approximate radius of panel - adjust based on your model
    # horizontal_dist = np.sqrt(displacement[0]**2 + displacement[1]**2)
    
    # if horizontal_dist > panel_radius:
    #     return 1.0  # Object center is outside panel radius
        
    # Object is safely on the panel
    return 0.0

def resample_data(episode, cut=True, scale=15):
    episode_resampled = []
    for item_idx in range(len(episode)):
        # print(episode[item_idx]['failure_phase_value'][0])
        if episode[item_idx]['failure_phase_value'][0] == 0.0 and item_idx%scale==0:
            episode_resampled.append(episode[item_idx])
        elif episode[item_idx]['failure_phase_value'][0] > 0.0 and episode[item_idx]['failure_phase_value'][0] < 1.0:
            episode_resampled.append(episode[item_idx])
        elif episode[item_idx]['failure_phase_value'][0] == 1.0 and item_idx%scale==0 and not cut:
            episode_resampled.append(episode[item_idx])
        else:
            pass
    return episode_resampled
                    
def plot_raw_metrics(original_episode, episode_num, dataset_type, save_path, current_object_name, timestamp):
    """
    Visualize original and resampled failure phase curves in a single plot.
    
    Parameters:
    -----------
    original_episode : list
        Original episode data
    resampled_episode : list
        Resampled episode data
    episode_num : int
        Episode number for the filename
    """
    # Extract original data
    original_time_steps = range(len(original_episode))
    original_risk_values = [item['failure_phase_value'][0] for item in original_episode]
    
    # Create figure and plot
    plt.figure(figsize=(12, 6))
    
    # Plot original data as a continuous line
    plt.plot(original_time_steps, original_risk_values, 'b-', 
            linewidth=2, alpha=0.6, label='Original data')
    
    # Enhance the plot
    plt.title('Original failure phases')
    plt.xlabel('Time Step')
    plt.ylabel('Risk Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/episode{episode_num}_{current_object_name}_{dataset_type}_{timestamp}.png')
    plt.close()

def plot_metrics(original_episode, resampled_episode, save_path, episode_name=None):
    """
    Visualize original and resampled failure phase curves in a single plot.
    
    Parameters:
    -----------
    original_episode : list
        Original episode data
    resampled_episode : list
        Resampled episode data
    episode_num : int
        Episode number for the filename
    """
    # Extract original data
    # original_time_steps = range(len(original_episode))
    # original_risk_values = [item['risk'][0] for item in original_episode]
    
    # Map resampled points to their original indices
    resampled_indices = []
    for r_item in resampled_episode:
        # Find matching item in original episode
        for i, o_item in enumerate(original_episode):
            if np.array_equal(r_item['time_step'], o_item['time_step']) and r_item['risk'][0] == o_item['risk'][0]:
                resampled_indices.append(i)
                break
    
    # Extract resampled values
    resampled_risk_values = [item['risk'][0] for item in resampled_episode]
    
    # Create figure and plot
    plt.figure(figsize=(12, 6))
    
    # Plot original data as a continuous line
    # plt.plot(original_time_steps, original_risk_values, 'b-', 
    #         linewidth=2, alpha=0.6, label='Original data')
    
    # Plot resampled data as a dotted line with markers
    plt.plot(resampled_indices, resampled_risk_values, 'r--', 
            linewidth=1.5, alpha=0.8, label='Resampled curve')
    
    # Add markers for resampled points
    plt.scatter(resampled_indices, resampled_risk_values, 
                color='red', s=40, label='Resampled points', zorder=5)
    
    # Enhance the plot
    plt.title('Comparison of Original and Resampled failure phases')
    plt.xlabel('Time Step')
    plt.ylabel('Risk Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text annotation showing reduction
    reduction = 100 * (1 - len(resampled_episode)/len(original_episode))
    plt.annotate(f"Data reduction: {reduction:.1f}%\nOriginal: {len(original_episode)} points\nResampled: {len(resampled_episode)} points", 
                xy=(0.02, 0.96), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/new/comparison_{episode_name}.png')
    plt.close()

def read_config(file_path):
    """
    Read and parse a JSON configuration file.
    
    Args:
        file_path (str): Path to the JSON config file
        
    Returns:
        dict: The parsed configuration as a dictionary
    """
    try:
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' contains invalid JSON.")
        return None
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None
    
def combine_arrays(arrays, start_idx=0, end_idx=None, operation='sum'):
    """
    Combine values in a series of NumPy arrays position by position.
    
    Parameters:
    -----------
    arrays : list of numpy.ndarray or list
        List of NumPy arrays (or lists) to combine
    start_idx : int, optional
        Starting index (inclusive) of arrays to combine, default is 0
    end_idx : int, optional
        Ending index (inclusive) of arrays to combine, default is None (which means all arrays)
    operation : str, optional
        Operation to perform: 'sum', 'mean', 'max', 'min', 'prod', default is 'sum'
    
    Returns:
    --------
    numpy.ndarray
        Combined array with the result of the operation
    """
    # Validate input
    if not arrays:
        raise ValueError("Input list is empty")
    
    # Set default end_idx if not provided
    if end_idx is None:
        end_idx = len(arrays) - 1  # Inclusive of the last element
    
    # Validate indices
    if start_idx < 0 or start_idx >= len(arrays):
        raise ValueError(f"start_idx {start_idx} is out of bounds for array of length {len(arrays)}")
    if end_idx < 0 or end_idx >= len(arrays):
        raise ValueError(f"end_idx {end_idx} is out of bounds for array of length {len(arrays)}")
    if start_idx > end_idx:
        raise ValueError(f"start_idx {start_idx} must be less than or equal to end_idx {end_idx}")
    
    # Select subset of arrays - note the +1 to make end_idx inclusive
    selected_arrays = arrays[start_idx:end_idx+1]
    
    # Convert all elements to numpy arrays if they aren't already
    selected_arrays = [np.array(arr) for arr in selected_arrays]
    
    # Check if all arrays have the same shape
    shape = selected_arrays[0].shape
    for i, arr in enumerate(selected_arrays[1:], 1):
        if arr.shape != shape:
            raise ValueError(f"Array at index {i+start_idx} has shape {arr.shape}, different from {shape}")
    
    # Stack arrays along a new axis
    stacked = np.stack(selected_arrays)
    
    # Perform the requested operation along the first axis (the stacking axis)
    if operation == 'sum':
        return np.sum(stacked, axis=0)
    elif operation == 'mean':
        return np.mean(stacked, axis=0)
    elif operation == 'max':
        return np.max(stacked, axis=0)
    elif operation == 'min':
        return np.min(stacked, axis=0)
    elif operation == 'prod':
        return np.prod(stacked, axis=0)
    else:
        raise ValueError(f"Unsupported operation: {operation}")


def reshape_contours(contours):
    """
    Reshape contours to make them compatible with OpenCV functions.
    Handles different input shapes including nested contours.
    
    Args:
        contours: Contours in shape (1, N, 1, 2) or similar
        
    Returns:
        Reshaped contours as a list of OpenCV-compatible contours
    """
    reshaped_contours = []
    
    # Handle case where we have a single contour in shape (1, N, 1, 2)
    if isinstance(contours, np.ndarray) and contours.ndim == 4:
        for i in range(contours.shape[0]):
            # Extract each contour and reshape to (N, 1, 2)
            contour = contours[i].reshape(-1, 1, 2).astype(np.int32)
            reshaped_contours.append(contour)
    # Case where we already have a list of contours
    elif isinstance(contours, list):
        for contour in contours:
            if isinstance(contour, np.ndarray):
                # Make sure contour is in shape (N, 1, 2)
                if contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2:
                    reshaped_contours.append(contour.astype(np.int32))
                elif contour.ndim == 2 and contour.shape[1] == 2:
                    # Reshape (N, 2) to (N, 1, 2)
                    reshaped_contours.append(contour.reshape(-1, 1, 2).astype(np.int32))
                else:
                    raise ValueError(f"Unsupported contour shape: {contour.shape}")
    else:
        raise ValueError(f"Unsupported contours type: {type(contours)}")
    
    return reshaped_contours

def apply_transformation(contour, rotation, tx, ty):
    """
    Apply rotation and translation to a contour.
    
    Args:
        contour: Contour points array from OpenCV in shape (N, 1, 2)
        rotation: Rotation angle in radians
        tx, ty: Translation parameters
    
    Returns:
        Transformed contour in the same shape as input
    """
    # Extract points as (N, 2)
    points = contour.reshape(-1, 2).astype(np.float32)
    
    # Create rotation matrix
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    
    # Apply rotation and translation
    transformed_points = (R @ points.T).T + np.array([tx, ty])
    
    # Convert back to original contour format
    transformed_contour = transformed_points.reshape(contour.shape).astype(np.int32)
    
    return transformed_contour

def generate_sample_contours(shape1=(1, 56, 1, 2), shape2=(1, 53, 1, 2)):
    """Generate two sample contours with specified shapes for testing."""
    
    # Create a simple polygon for the first contour
    angles1 = np.linspace(0, 2*np.pi, shape1[1], endpoint=False)
    radius1 = 100 + 10 * np.sin(3 * angles1)  # Add some variation
    x1 = 200 + radius1 * np.cos(angles1)
    y1 = 200 + radius1 * np.sin(angles1)
    contour1_points = np.column_stack((x1, y1))
    
    # Reshape to match the requested shape
    contour1 = contour1_points.reshape(shape1).astype(np.int32)
    
    # Create a slightly different polygon for the second contour
    angles2 = np.linspace(0, 2*np.pi, shape2[1], endpoint=False)
    radius2 = 90 + 15 * np.sin(4 * angles2)  # Different variation
    x2 = 250 + radius2 * np.cos(angles2)  # Shifted center
    y2 = 220 + radius2 * np.sin(angles2)
    contour2_points = np.column_stack((x2, y2))
    
    # Reshape to match the requested shape
    contour2 = contour2_points.reshape(shape2).astype(np.int32)
    
    return contour1, contour2

def load_image_contours(image1_path, image2_path, target_shape1=(1, 56, 1, 2), target_shape2=(1, 53, 1, 2)):
    """
    Load contours from two images and reshape to target shapes.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        target_shape1: Target shape for first contour
        target_shape2: Target shape for second contour
        
    Returns:
        Two contours with specified shapes
    """
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Failed to load images: {image1_path} or {image2_path}")
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    raw_contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour from each image
    if len(raw_contours1) > 0 and len(raw_contours2) > 0:
        # Select largest contours
        contour1 = max(raw_contours1, key=cv2.contourArea)
        contour2 = max(raw_contours2, key=cv2.contourArea)
        
        # Resample contours to match target shapes
        contour1 = resample_contour(contour1, target_shape1[1])
        contour2 = resample_contour(contour2, target_shape2[1])
        
        # Reshape to match target shapes
        contour1 = contour1.reshape(target_shape1).astype(np.int32)
        contour2 = contour2.reshape(target_shape2).astype(np.int32)
        
        return contour1, contour2
    else:
        raise ValueError("No contours found in one or both images")

def resample_contour(contour, target_points):
    """
    Resample a contour to have exactly the specified number of points.
    
    Args:
        contour: OpenCV contour
        target_points: Desired number of points
        
    Returns:
        Resampled contour with target_points points
    """
    # Convert to (N, 2) array
    points = contour.reshape(-1, 2)
    
    # Calculate the perimeter
    perimeter = cv2.arcLength(contour, True)
    
    # Create a new array for resampled points
    resampled = np.zeros((target_points, 2), dtype=np.int32)
    
    # Distance between each new point
    step = perimeter / target_points
    
    # Initialize
    dist_traveled = 0
    new_idx = 0
    resampled[0] = points[0]
    
    # Loop through original points
    for i in range(1, len(points)):
        # Distance to next original point
        dist = np.linalg.norm(points[i] - points[i-1])
        
        while dist_traveled + dist >= step and new_idx < target_points - 1:
            # Interpolate to find next point
            alpha = (step - dist_traveled) / (dist + 1e-10)
            new_idx += 1
            resampled[new_idx] = points[i-1] + alpha * (points[i] - points[i-1])
            dist -= step - dist_traveled
            dist_traveled = 0
        
        dist_traveled += dist
    
    # Ensure we have the exact number of points (handle any floating-point issues)
    if new_idx < target_points - 1:
        resampled[new_idx+1:] = points[-1]
    
    return resampled.reshape(-1, 1, 2)

def visualize_contour_transformation(processed_c1, processed_c2, features, data_idx, output_path="./", episode_name="episode"):
    """
    Visualize original contours and the transformation using pre-processed contours and features.
    
    Args:
        processed_c1: First contour, already processed for use with process_consecutive_frames
        processed_c2: Second contour, already processed for use with process_consecutive_frames
        features: Transformation features from process_consecutive_frames (rotation, tx, ty)
        output_path: Path to save the visualization
    """
    # Extract rotation and translation parameters
    rotation, tx, ty = features[0], features[1], features[2]
    # print(f"Transformation parameters - Rotation: {rotation:.4f} rad ({rotation * 180 / np.pi:.2f}°), Translation: ({tx:.2f}, {ty:.2f})")
    
    # Reshape contours for visualization if needed
    reshaped_c1 = processed_c1
    reshaped_c2 = processed_c2
    
    # Ensure contours are in the right format for visualization
    if isinstance(processed_c1, np.ndarray) and processed_c1.ndim > 3:
        reshaped_c1 = reshape_contours(processed_c1)
    elif isinstance(processed_c1, list):
        reshaped_c1 = processed_c1
    
    if isinstance(processed_c2, np.ndarray) and processed_c2.ndim > 3:
        reshaped_c2 = reshape_contours(processed_c2)
    elif isinstance(processed_c2, list):
        reshaped_c2 = processed_c2
        
    # Apply transformation to the first contour
    transformed_contours = []
    if isinstance(reshaped_c1, list):
        for c in reshaped_c1:
            transformed_contour = apply_transformation(c, rotation, tx, ty)
            transformed_contours.append(transformed_contour)
    else:
        # Handle case where reshaped_c1 is a single contour
        transformed_contours.append(apply_transformation(reshaped_c1, rotation, tx, ty))
    
    # Create canvas for visualization
    canvas_size = (800, 600)
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    
    # Ensure all contours are in the right format for cv2.drawContours
    draw_c1 = reshaped_c1 if isinstance(reshaped_c1, list) else [reshaped_c1]
    draw_c2 = reshaped_c2 if isinstance(reshaped_c2, list) else [reshaped_c2]
    
    # Draw contours
    cv2.drawContours(canvas, draw_c1, -1, (255, 0, 0), 2)  # Original contour1 in blue
    cv2.drawContours(canvas, draw_c2, -1, (0, 255, 0), 2)  # Original contour2 in green
    cv2.drawContours(canvas, transformed_contours, -1, (0, 0, 255), 2)  # Transformed contour1 in red
    
    # Add legend
    legend_y = 30
    cv2.putText(canvas, "Original Contour 1", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(canvas, "Original Contour 2", (20, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "Transformed Contour 1", (20, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add transformation parameters
    rotation_deg = rotation * 180 / np.pi
    cv2.putText(canvas, f"Rotation: {rotation_deg:.2f} degrees", (20, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"Translation: ({tx:.2f}, {ty:.2f})", (20, legend_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save the visualization
    cv2.imwrite(f'{output_path}/new/transform_{episode_name}_{data_idx}.png', canvas)
    print(f"Visualization saved as {output_path}")
    
    # # Create figure and display/save the result
    # plt.figure(figsize=(10, 8))
    # plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # plt.title("Contour Transformation Visualization")
    # plt.axis('off')
    # plt.tight_layout()
    
    # # Save as PNG as well
    # output_png = os.path.splitext(output_path)[0] + "_plt.png"
    # plt.savefig(output_png)
    # plt.close()