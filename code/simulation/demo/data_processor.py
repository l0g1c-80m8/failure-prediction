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
            
            print("file_idx", file_idx, "failure_phase_value", failure_phase_value)
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
                top_object_transforms = process_consecutive_frames(top_camera_object_contours[-window], top_object_contour)
                # print("top_object_transforms", top_object_transforms)
                top_panel_transforms = process_consecutive_frames(top_camera_panel_contours[-window], top_panel_contour)
                # print("top_panel_transforms", top_panel_transforms)
                front_object_transforms = process_consecutive_frames(front_camera_object_contours[-window], front_object_contour)
                # print("front_object_transforms", front_object_transforms)
                front_panel_transforms = process_consecutive_frames(front_camera_panel_contours[-window], front_panel_contour)
                # print("front_panel_transforms", front_panel_transforms)

                try:
                    # Check for empty transforms
                    if len(top_object_transforms) == 0 and len(top_panel_transforms) == 0 and \
                    len(front_object_transforms) == 0 and len(front_panel_transforms) == 0:
                        raise Exception("All transforms are empty")
                except Exception as e:
                    print(f"Error in contour processing: {e}")
                    continue

                # Extract features from each transform set
                top_object_features = extract_transform_features(top_object_transforms)
                top_panel_features = extract_transform_features(top_panel_transforms)
                front_object_features = extract_transform_features(front_object_transforms)
                front_panel_features = extract_transform_features(front_panel_transforms)
                # print("top_object_transforms",top_object_transforms)
                # print("top_object_features",top_object_features)

                # Combine all features
                combined_features = np.concatenate([
                    top_object_features,
                    top_panel_features,
                    front_object_features,
                    front_panel_features,
                    end_effector_pos
                ])

                # print("top_object_features", np.asarray(top_object_features).shape) # (4,)
                # print("top_panel_features", np.asarray(top_panel_features).shape) # (4,)
                # print("front_object_features", np.asarray(front_object_features).shape) # (4,)
                # print("front_panel_features", np.asarray(front_panel_features).shape) # (4,)
                # print("end_effector_pos", np.asarray(end_effector_pos).shape) # (3,)
                # print("combined_features", combined_features.shape) # (19,)
                if combined_features.shape[0]!=19:
                    raise ValueError(f"Error: combined_features shape {combined_features.shape} != 19")
                
                episodes[file_idx][data_idx]['state'] = np.asarray(combined_features, dtype=np.float32)
            else:
                episodes[file_idx][data_idx]['state'] = np.ones(19, dtype=np.float32)

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
        # Plot after simulation
        dataset_type = "train" if "train" in episode_path else "val"
        plot_metrics(episodes[file_idx], episode_resampled, file_idx, dataset_type, episode_folder_path)

        # 3. Save the resampled data
        print(f"Generating {dataset_type} resampled examples...")
        np.save(f"{episode_folder_path}/new/episode_{file_idx}.npy", episode_resampled)


if __name__ == "__main__":
    data_dir = "demo/data/train_raw"
    main(data_dir, interpolate_type = "linear")