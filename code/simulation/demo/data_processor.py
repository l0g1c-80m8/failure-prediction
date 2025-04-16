import os
import numpy as np
from common_functions import (linear_interpolation, resample_data, plot_metrics,
                              process_consecutive_frames, extract_transform_features)

def main(data_dir, interpolate_type = "linear"):
    episode_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    episodes = {}
    for file_idx, episode_file in enumerate(episode_files):
        episode_path = os.path.join(data_dir, episode_file)
        print("episode_path", episode_path)
        episode_folder_path = os.path.dirname(episode_path)
        os.makedirs(os.path.join(episode_folder_path, "new"), exist_ok=True)
        print("episode_folder_path", episode_folder_path)
        episodes[file_idx] = np.load(episode_path, allow_pickle=True)
        episode_failure_phase_start = -1
        episode_failure_phase_reach = -1
        
        top_camera_object_contours = []
        top_camera_panel_contours = []
        front_camera_object_contours = []
        front_camera_panel_contours = []
        window = 30
        
        # 1. Interpolate the data
        for data_idx in range(len(episodes[file_idx])):
            failure_phase_value = episodes[file_idx][data_idx]['failure_phase_value']  # Extract the scalar from the array
            episodes[file_idx][data_idx]["risk"] = failure_phase_value
            
            # print("file_idx", file_idx, "failure_phase_value", failure_phase_value)
            # Record first occurrence of value 0.5
            if failure_phase_value == 0.5 and episode_failure_phase_start == -1:
                episode_failure_phase_start = data_idx
                print(f"Episode {file_idx}: First occurrence of failure_phase_value=0.5 at index {data_idx}")
            # Record first occurrence of value 1.0
            if failure_phase_value == 1.0 and episode_failure_phase_reach == -1:
                episode_failure_phase_reach = data_idx
                print(f"Episode {file_idx}: First occurrence of failure_phase_value=1.0 at index {data_idx}")

            # 1.1 Calculate states
            top_object_contour = episodes[file_idx][data_idx]['object_top_contour']
            top_panel_contour = episodes[file_idx][data_idx]['gripper_top_contour']
            front_object_contour = episodes[file_idx][data_idx]['object_front_contour']
            front_panel_contour = episodes[file_idx][data_idx]['gripper_front_contour']
            end_effector_pos = episodes[file_idx][data_idx]['end_effector_pos']
            top_camera_object_contours.append(top_object_contour)
            top_camera_panel_contours.append(top_panel_contour)
            front_camera_object_contours.append(front_object_contour)
            front_camera_panel_contours.append(front_panel_contour)

            if len(top_camera_object_contours) >= window:
                try:
                    matrix = False
                    # print("top_camera_object_contours[-window] shape", top_camera_object_contours[-window].shape)
                    # print("top_object_contour shape", top_object_contour.shape)
                    if top_camera_object_contours[-window].shape[0] == 0 or top_object_contour.shape[0] == 0:
                        break
                    top_object_features = process_consecutive_frames(top_camera_object_contours[-window], top_object_contour, matrix=matrix)
                    # print("top_object_features", top_object_features)
                    top_panel_features = process_consecutive_frames(top_camera_panel_contours[-window], top_panel_contour, matrix=matrix)
                    # print("top_panel_features", top_panel_features)
                    front_object_features = process_consecutive_frames(front_camera_object_contours[-window], front_object_contour, matrix=matrix)
                    # print("front_object_features", front_object_features)
                    front_panel_features = process_consecutive_frames(front_camera_panel_contours[-window], front_panel_contour, matrix=matrix)
                    # print("front_panel_features", front_panel_features)

                    # Combine all features
                    combined_features = np.concatenate([
                        top_object_features,
                        top_panel_features,
                        front_object_features,
                        front_panel_features,
                        end_effector_pos
                    ])

                    # print("top_object_features", np.asarray(top_object_features).shape) # (4,) if matrix, otherwise (3,)
                    # print("top_panel_features", np.asarray(top_panel_features).shape) # (4,) if matrix, otherwise (3,)
                    # print("front_object_features", np.asarray(front_object_features).shape) # (4,) if matrix, otherwise (3,)
                    # print("front_panel_features", np.asarray(front_panel_features).shape) # (4,) if matrix, otherwise (3,)
                    # print("end_effector_pos", np.asarray(end_effector_pos).shape) # (3,)
                    # print("combined_features", combined_features.shape) # (19,) if matrix, otherwise (15,)
                    if (matrix and combined_features.shape[0]==19) or (not matrix and combined_features.shape[0]==15):
                        episodes[file_idx][data_idx]['state'] = np.asarray(combined_features, dtype=np.float32)
                    else:
                        raise ValueError(f"Error: combined_features shape {combined_features.shape} incorrect")
                    
                except Exception as e:
                    print(f"Error in contour processing for frame {data_idx}: {e}")

        # After processing the episode, report if any values weren't found
        if episode_failure_phase_start == -1 or episode_failure_phase_reach == -1:
            print(f"Episode {file_idx}: No occurrence of failure_phase_value=1.0")
        else:
            if interpolate_type == "linear":
                # Interpolate values from 0 to 1 for latest failure_time_step to first_failure_time_step
                interpolated_values = linear_interpolation(episode_failure_phase_reach, episode_failure_phase_start)
                # Update the "action" key for the dictionaries between i and k
                for idx, value in enumerate(interpolated_values, start=episode_failure_phase_start):
                    # print("value", value)
                    episodes[file_idx][idx]["risk"] = np.asarray([value], dtype=np.float32)
        
        # 2. Resample the data
        episode_resampled = resample_data(episodes[file_idx])
        dataset_type = "train" if "train" in episode_path else "val"
        # keep only the data after the first window
        episode_resampled_crop = episode_resampled[window:]

        # 4. Verify all frames have 'state' and 'risk' fields before saving
        missing_states = 0
        for frame_idx in range(len(episode_resampled_crop)):
            if 'state' not in episode_resampled_crop[frame_idx] or 'risk' not in episode_resampled_crop[frame_idx]:
                missing_states += 1
        
        if missing_states > 0:
            print(f"WARNING: Missing 'state' or 'risk' to {missing_states} frames in resampled episode {file_idx}")
        else:
            print(f"Generating {dataset_type} resampled examples...")
            plot_metrics(episodes[file_idx], episode_resampled, file_idx, dataset_type, episode_folder_path)
            # 5. Save the resampled data
            np.save(f"{episode_folder_path}/new/episode_{file_idx}.npy", episode_resampled_crop)


if __name__ == "__main__":
    data_dir = "demo/data/train_raw" # demo/data/test_data_0403/val_raw
    main(data_dir, interpolate_type = "linear")