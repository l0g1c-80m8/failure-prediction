import numpy as np
import os
import glob

# Directory containing the .npy files
data_dir = "../videos/"  # You can change this to your folder path

# Essential contour keys that must be present
required_contours = [
    'object_top_contour', 
    'object_front_contour', 
    'gripper_top_contour', 
    'gripper_front_contour'
]

# Create the processed directory if it doesn't exist
processed_dir = os.path.join(data_dir, "processed")
os.makedirs(processed_dir, exist_ok=True)  # This will create the directory if it doesn't exist

# Find all .npy files in the directory
npy_files = glob.glob(os.path.join(data_dir, "*.npy"))

# Process each file
for file_path in npy_files:
    print(f"\nProcessing file: {file_path}")
    
    # Load the .npy file
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"File shape: {data.shape}")
        
        # Create a new list to hold valid items
        valid_items = []
        stop_copying = False
        
        # Process each item in the file
        for i, item in enumerate(data):
            # Check if we should stop copying items
            if stop_copying:
                print(f"Skipping item {i} and beyond due to missing contour(s)")
                break
                
            # Check if all required contours are present
            if isinstance(item, dict):
                # Check for missing contours
                missing_contours = [contour for contour in required_contours if contour not in item]
                
                if missing_contours:
                    print(f"\nItem {i}: Missing contour(s): {missing_contours}")
                    print("Stopping copy process at this item")
                    stop_copying = True
                    continue
                    
                # If we get here, all contours are present
                # Copy the 'risk' value to 'failure_phase_value' if it exists
                # if 'risk' in item:
                #     item['failure_phase_value'] = item['risk']  # Uncommented this line
                
                # Add the item to our valid items list
                valid_items.append(item)
                print(f"Item {i}: All contours present, added to new array")
            else:
                print(f"Item {i} is not a dictionary. Type: {type(item)}")
                print("Stopping copy process at this item")
                stop_copying = True
        
        # Convert the list to a numpy array
        if valid_items:
            new_data = np.array(valid_items, dtype=object)
            
            # Save the new array to a file
            base_name = os.path.basename(file_path)
            new_file_path = os.path.join(processed_dir, base_name)
            np.save(new_file_path, new_data)
            
            print(f"\nSaved {len(valid_items)} items to {new_file_path}")
            print(f"Original file had {len(data)} items")
        else:
            print("\nNo valid items found, no new file created")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("\nProcessing complete.")