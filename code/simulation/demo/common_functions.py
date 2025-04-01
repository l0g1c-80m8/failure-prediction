import numpy as np

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

def icp_2d(src_contour, dst_contour, max_iterations=16, tolerance=1e-6):
    """
    Perform 2D ICP algorithm between two contours.
    
    Args:
        src_contour: Source contour points from OpenCV findContours
        dst_contour: Destination contour points from OpenCV findContours
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for mean squared error
        
    Returns:
        H: 3x3 homogeneous transformation matrix
        error: Final mean squared error
    """
    # Convert contours to point arrays
    src_points = src_contour.reshape(-1, 2).astype(np.float32)
    dst_points = dst_contour.reshape(-1, 2).astype(np.float32)
    
    # Initialize transformation
    R_total = np.eye(2)
    t_total = np.zeros(2)
    
    prev_error = float('inf')
    
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
    
    # Apply log1p scaling to R_total and t_total
    scale_factor = 1e5
    R_total_scaled = np.sign(R_total) * np.log1p(np.abs(R_total) * scale_factor)
    t_total_scaled = np.sign(t_total) * np.log1p(np.abs(t_total) * scale_factor)
    
    # print("Original R_total:", R_total)
    # print("Scaled R_total:", R_total_scaled)
    # print("Original t_total:", t_total)
    # print("Scaled t_total:", t_total_scaled)
    
    # Create homogeneous transformation matrix with scaled values
    H = create_homogeneous_matrix(R_total_scaled, t_total_scaled)

    # print("H with scaled values:", H)
    
    return H, current_error

def process_consecutive_frames(contours1, contours2):
    """
    Process consecutive frames and calculate transformations for each contour pair.
    
    Args:
        contours1: List of contours from first frame
        contours2: List of contours from second frame
        
    Returns:
        List of (R, t, error) tuples for each matched contour pair
    """
    results = []
    
    # Match contours based on area similarity
    areas1 = [cv2.contourArea(cnt) for cnt in contours1]
    areas2 = [cv2.contourArea(cnt) for cnt in contours2]
    
    for i, cnt1 in enumerate(contours1):
        # Find best matching contour in second frame
        best_match = None
        best_area_diff = float('inf')
        
        for j, cnt2 in enumerate(contours2):
            area_diff = abs(areas1[i] - areas2[j])
            if area_diff < best_area_diff:
                best_area_diff = area_diff
                best_match = cnt2
        
        if best_match is not None:
            # Calculate ICP between matched contours
            H, error = icp_2d(cnt1, best_match)
            results.append((H, error))
    
    return results

def extract_transform_features(transforms):
    if not transforms:
        # Return zeros if no transforms
        return np.zeros(6)  # 4 for rotation matrix elements + 2 for translation
    
    # Take first transform if multiple are present
    H, error = transforms[0]
    
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

def process_camera_frame(frame):
    """
    Process camera frame to get mask and contours for non-zero pixel values.
    
    Args:
        frame: RGB image array (height, width, 3)
    
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
    min_contour_area = 100  # Adjust this threshold as needed
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Create visualization of the filtered image
    filtered_image = frame.copy()
    cv2.drawContours(filtered_image, contours, -1, (0, 255, 0), 2)
    
    # Draw bounding panel around detected objects
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(filtered_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    #     # Calculate and display centroid
    #     M = cv2.moments(contour)
    #     if M["m00"] != 0:
    #         cx = int(M["m10"] / M["m00"])
    #         cy = int(M["m01"] / M["m00"])
    #         cv2.circle(filtered_image, (cx, cy), 5, (0, 0, 255), -1)
    
    return mask, contours, filtered_image


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

def resample_data(episode):
    episode_resampled = []
    scale=15
    for item_idx in range(len(episode)):
        # print(episode[item_idx]['failure_phase_value'][0])
        if episode[item_idx]['failure_phase_value'][0] == 0.0 and item_idx%scale==0:
            episode_resampled.append(episode[item_idx])
        elif episode[item_idx]['failure_phase_value'][0] > 0.0 and episode[item_idx]['failure_phase_value'][0] < 1.0:
            episode_resampled.append(episode[item_idx])
        elif episode[item_idx]['failure_phase_value'][0] == 1.0 and item_idx%scale==0:
            episode_resampled.append(episode[item_idx])
    return episode_resampled
                    
def plot_raw_metrics(original_episode, episode_num, dataset_type, save_path):
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
    plt.savefig(f'{save_path}/comparison_episode{episode_num}_{dataset_type}.png')
    plt.close()

def plot_metrics(original_episode, resampled_episode, episode_num, dataset_type, save_path):
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
    
    # Map resampled points to their original indices
    resampled_indices = []
    for r_item in resampled_episode:
        # Find matching item in original episode
        for i, o_item in enumerate(original_episode):
            if np.array_equal(r_item['time_step'], o_item['time_step']) and r_item['failure_phase_value'][0] == o_item['failure_phase_value'][0]:
                resampled_indices.append(i)
                break
    
    # Extract resampled values
    resampled_risk_values = [item['failure_phase_value'][0] for item in resampled_episode]
    
    # Create figure and plot
    plt.figure(figsize=(12, 6))
    
    # Plot original data as a continuous line
    plt.plot(original_time_steps, original_risk_values, 'b-', 
            linewidth=2, alpha=0.6, label='Original data')
    
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
    plt.savefig(f'{save_path}/comparison_episode{episode_num}_{dataset_type}.png')
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