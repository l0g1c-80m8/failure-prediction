import numpy as np

def read_npy_file(file_path):
    """
    Read a .npy file and return its contents as a NumPy array.
    
    Parameters:
    file_path (str): Path to the .npy file
    
    Returns:
    numpy.ndarray: The array stored in the .npy file
    """
    try:
        # Load the .npy file with allow_pickle=True
        data = np.load(file_path, allow_pickle=True)
        
        # Return the loaded data
        return data
    except Exception as e:
        print(f"Error reading .npy file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    file_path = "../simple_model/data/train/episode_0_stanford_bunny_raw.npy"
    
    # Read the .npy file
    data = read_npy_file(file_path)
    for i in range(len(data)):
        print(data[i]["state"])
    
    # if data is not None:
    #     print(f"Data loaded successfully!")
    #     print(f"Array shape: {data.shape}")
    #     print(f"Array data type: {data.dtype}")
    #     print(f"First few elements: {data.flatten()[:5]}")