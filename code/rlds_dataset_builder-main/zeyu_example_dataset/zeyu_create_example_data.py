import numpy as np
import tqdm
import os
from scipy.interpolate import CubicSpline

N_TRAIN_EPISODES = 1
N_VAL_EPISODES = 1
EPISODE_LENGTH = 2  # Number of points in trajectory

# Thresholds for action calculation
DISPLACEMENT_THRESHOLD_LOW = 0.05
DISPLACEMENT_THRESHOLD_HIGH = 0.1
LINEAR_SPEED_THRESHOLD_LOW = 0.05
LINEAR_SPEED_THRESHOLD_HIGH = 0.1
ANGULAR_SPEED_THRESHOLD_LOW = 0.1
ANGULAR_SPEED_THRESHOLD_HIGH = 1.0

# Helper function to compute linear speed (Euclidean distance) and angular speed
def compute_speeds(prev_pos, curr_pos):
    # Linear speed (Euclidean distance between two points)
    linear_speed = np.linalg.norm(curr_pos - prev_pos)
    
    # Angular speed calculation
    angular_speed = np.arctan2(curr_pos[1] - prev_pos[1], curr_pos[0] - prev_pos[0])
    
    return linear_speed, angular_speed

# Function to calculate action value based on displacement, linear speed, and angular speed
def calculate_action(displacement, linear_speed, angular_speed):
    # Check if all values are in the lower range
    if np.linalg.norm(displacement) < DISPLACEMENT_THRESHOLD_LOW or \
       linear_speed < LINEAR_SPEED_THRESHOLD_LOW or \
       abs(angular_speed) < ANGULAR_SPEED_THRESHOLD_LOW:
        return 0  # Set action to 0
    
    # Check if any value exceeds the higher thresholds
    if np.linalg.norm(displacement) > DISPLACEMENT_THRESHOLD_HIGH or \
       linear_speed > LINEAR_SPEED_THRESHOLD_HIGH or \
       abs(angular_speed) > ANGULAR_SPEED_THRESHOLD_HIGH:
        return 1  # Set action to 1

    # If the values fall between thresholds, interpolate a reasonable action
    # Normalize between 0 and 1 based on distance from the lower to upper threshold
    displacement_factor = (np.linalg.norm(displacement) - DISPLACEMENT_THRESHOLD_LOW) / \
                          (DISPLACEMENT_THRESHOLD_HIGH - DISPLACEMENT_THRESHOLD_LOW)
    linear_speed_factor = (linear_speed - LINEAR_SPEED_THRESHOLD_LOW) / \
                          (LINEAR_SPEED_THRESHOLD_HIGH - LINEAR_SPEED_THRESHOLD_LOW)
    angular_speed_factor = (abs(angular_speed) - ANGULAR_SPEED_THRESHOLD_LOW) / \
                           (ANGULAR_SPEED_THRESHOLD_HIGH - ANGULAR_SPEED_THRESHOLD_LOW)

    # Combine the factors with an average, assuming equal importance
    action_value = (displacement_factor + linear_speed_factor + angular_speed_factor) / 3.0
    return action_value

def create_smooth_trajectory(num_points):
    # Generate some random control points within a 1x1x1 space
    control_points = np.random.rand(10, 3)  # 10 random control points
    
    # Generate a parameter for the control points (from 0 to 1)
    t_control = np.linspace(0, 1, len(control_points))
    
    # Generate a parameter for the trajectory points (from 0 to 1)
    t_trajectory = np.linspace(0, 1, num_points)
    
    # Use cubic splines to interpolate a smooth trajectory between the control points
    spline_x = CubicSpline(t_control, control_points[:, 0])
    spline_y = CubicSpline(t_control, control_points[:, 1])
    spline_z = CubicSpline(t_control, control_points[:, 2])
    
    # Evaluate the splines to get the smooth trajectory points
    trajectory = np.vstack([spline_x(t_trajectory), spline_y(t_trajectory), spline_z(t_trajectory)]).T
    
    return trajectory

def create_fake_episode(path):
    episode = []
    trajectory = create_smooth_trajectory(EPISODE_LENGTH)  # Get a smooth trajectory

    for step in range(EPISODE_LENGTH):
        if step == 0:
            prev_pos = trajectory[step]  # First point has no previous position
            curr_pos = trajectory[step]
            linear_speed, angular_speed = 0, 0  # No movement for the first point
        else:
            prev_pos = trajectory[step - 1]
            curr_pos = trajectory[step]
            linear_speed, angular_speed = compute_speeds(prev_pos, curr_pos)

        # Combine displacement, linear speed, and angular speed into a state array
        displacement = curr_pos - prev_pos  # Change in position
        state = np.hstack([displacement, [linear_speed, angular_speed]])

        # Pad state to 9 values (for compatibility)
        state_padded = np.pad(state, (0, 9 - len(state)), 'constant')

        # Calculate the action based on the thresholds and interpolation logic
        action_value = calculate_action(displacement, linear_speed, angular_speed)

        # Ensure action is stored as a (1,) tensor, not as a scalar
        action_value = np.asarray([action_value], dtype=np.float32)

        episode.append({
            'image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'state': np.asarray(state_padded, dtype=np.float32),  # Save the padded state
            'action': action_value,  # Ensure action is a tensor of shape (1,)
            'language_instruction': 'dummy instruction',
        })

    np.save(path, episode)


# Create fake episodes for train and validation
print("Generating train examples...")
os.makedirs('data/train', exist_ok=True)
for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    create_fake_episode(f'data/train/episode_{i}.npy')

print("Generating val examples...")
os.makedirs('data/val', exist_ok=True)
for i in tqdm.tqdm(range(N_VAL_EPISODES)):
    create_fake_episode(f'data/val/episode_{i}.npy')

print('Successfully created example data!')