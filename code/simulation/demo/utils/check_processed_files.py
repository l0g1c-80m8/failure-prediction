#!/usr/bin/env python3
import os
import sys
import re
import argparse

def extract_object_names(directory):
    """
    Extract object names from NPY files in the specified directory.
    Returns a set of unique object identifiers.
    """
    processed_objects = set()
    
    # Regular expression to extract object name from filename
    pattern = r'episode\d+_(.*?)_train_\d+_\d+_raw\.npy'
    
    # Walk through the directory
    if os.path.exists(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.npy'):
                    match = re.search(pattern, file)
                    if match:
                        object_name = match.group(1)
                        processed_objects.add(object_name)
    else:
        print(f"Directory: {directory} not found")
    return processed_objects

def main():
    parser = argparse.ArgumentParser(description='Extract object names from NPY files')
    parser.add_argument('--npy_dir', required=True, help='Directory containing NPY files')
    parser.add_argument('--output', required=True, help='Output file to store object names')
    
    args = parser.parse_args()
    
    # Extract object names
    processed_objects = extract_object_names(args.npy_dir)
    
    # Write to output file
    with open(args.output, 'w') as f:
        for obj in processed_objects:
            f.write(f"{obj}\n")
    
    print(f"Found {len(processed_objects)} processed objects, saved to {args.output}")

if __name__ == "__main__":
    main()