#!/usr/bin/env python3
import os
import re
import argparse
import shutil
import glob

def extract_object_name(filename):
    """
    Extract object name from an NPY filename.
    Example: "episode0_sem-WallClock-1e2ea05e566e315c35836c728d324152_train_20250427_173921_raw.npy"
    Returns: "sem-WallClock-1e2ea05e566e315c35836c728d324152"
    """
    pattern = r'episode\d+_(.*?)_train_\d+_\d+_raw\.(npy|png)$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def get_base_filename(filename):
    """
    Get the base filename without extension.
    Example: "episode0_sem-WallClock-1e2ea05e566e315c35836c728d324152_train_20250427_173921_raw.npy"
    Returns: "episode0_sem-WallClock-1e2ea05e566e315c35836c728d324152_train_20250427_173921_raw"
    """
    return os.path.splitext(filename)[0]

def load_processed_objects(processed_objects_file):
    """
    Load the list of processed object names from the file.
    """
    processed_objects = set()
    if os.path.exists(processed_objects_file):
        with open(processed_objects_file, 'r') as f:
            for line in f:
                obj_name = line.strip()
                if obj_name:
                    processed_objects.add(obj_name)
    return processed_objects

def cleanup_unprocessed_files(raw_data_dir, processed_objects_file, backup_dir=None, dry_run=True):
    """
    Delete files in raw_data_dir that aren't in the processed_objects list.
    Also removes corresponding PNG files with the same base name.
    
    Parameters:
    - raw_data_dir: Directory containing raw NPY files
    - processed_objects_file: File containing list of processed object names
    - backup_dir: Optional directory to move files to instead of deleting
    - dry_run: If True, only simulate deletion/moving without actually doing it
    """
    # Load the list of processed objects
    processed_objects = load_processed_objects(processed_objects_file)
    print(f"Loaded {len(processed_objects)} processed object names from {processed_objects_file}")
    
    # Create backup directory if needed
    if backup_dir and not dry_run:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Backup directory: {backup_dir}")
    
    # Count statistics
    total_files = 0
    to_remove = 0
    keep_files = 0
    unidentified = 0
    
    # Create sets to track files already processed
    processed_base_filenames = set()
    
    # First pass: identify NPY files to remove/keep
    npy_files_to_remove = []
    base_filenames_to_remove = set()
    
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if filename.endswith('.npy'):
                total_files += 1
                file_path = os.path.join(root, filename)
                object_name = extract_object_name(filename)
                base_filename = get_base_filename(filename)
                
                if object_name is None:
                    print(f"Warning: Could not extract object name from {filename}")
                    unidentified += 1
                    continue
                
                if object_name in processed_objects:
                    # This file is processed, keep it
                    keep_files += 1
                    processed_base_filenames.add(base_filename)
                    print(f"Keeping: {filename}")
                else:
                    # This file is not processed, mark for removal
                    to_remove += 1
                    npy_files_to_remove.append(file_path)
                    base_filenames_to_remove.add(base_filename)
                    if dry_run:
                        print(f"Would remove NPY: {filename}")
    
    # Second pass: find and remove all related files (PNG, etc.)
    related_files_to_remove = []
    
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if not filename.endswith('.npy'):  # Look for non-NPY files
                file_path = os.path.join(root, filename)
                base_filename = get_base_filename(filename)
                
                # If the base filename is in our removal list
                if base_filename in base_filenames_to_remove:
                    related_files_to_remove.append(file_path)
                    if dry_run:
                        print(f"Would remove related file: {filename}")
    
    # Actually perform the deletion/move if not a dry run
    if not dry_run:
        for file_path in npy_files_to_remove + related_files_to_remove:
            filename = os.path.basename(file_path)
            if backup_dir:
                # Move to backup instead of deleting
                backup_path = os.path.join(backup_dir, filename)
                print(f"Moving to backup: {filename}")
                shutil.move(file_path, backup_path)
            else:
                # Delete the file
                print(f"Deleting: {filename}")
                os.remove(file_path)
    
    # Print summary
    print("\nSummary:")
    print(f"Total NPY files scanned: {total_files}")
    print(f"Files to keep: {keep_files}")
    print(f"NPY files to remove: {len(npy_files_to_remove)}")
    print(f"Related files to remove: {len(related_files_to_remove)}")
    print(f"Total files to remove: {len(npy_files_to_remove) + len(related_files_to_remove)}")
    print(f"Unidentified files: {unidentified}")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually deleted or moved.")
        print("To actually perform the deletion, run with --no-dry-run")

def main():
    parser = argparse.ArgumentParser(description='Clean up unprocessed NPY files and related files')
    parser.add_argument('--raw_data_dir', required=True, help='Directory containing raw NPY files')
    parser.add_argument('--processed_objects_file', required=True, help='File containing list of processed object names')
    parser.add_argument('--backup_dir', help='Move files to this directory instead of deleting them')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually delete/move files instead of just simulating')
    
    args = parser.parse_args()
    
    # Run the cleanup with or without dry run
    cleanup_unprocessed_files(
        args.raw_data_dir, 
        args.processed_objects_file, 
        args.backup_dir, 
        not args.no_dry_run
    )

if __name__ == "__main__":
    main()