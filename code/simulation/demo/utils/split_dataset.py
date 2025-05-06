#!/usr/bin/env python3
import os
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm

def split_dataset(source_dir, target_dir, train_ratio=0.8, seed=42):
    """
    Split dataset files from source_dir into train and validation sets
    and copy them to the respective subdirectories in target_dir.
    
    Args:
        source_dir: Directory containing the dataset files
        target_dir: Target directory where train and val subdirectories will be created
        train_ratio: Ratio of files to use for training (default: 0.8)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create target directories if they don't exist
    train_dir = os.path.join(target_dir, "train")
    val_dir = os.path.join(target_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # List all NPY files in the source directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    
    if not files:
        print(f"No NPY files found in {source_dir}")
        return
    
    # Shuffle the files
    random.shuffle(files)
    
    # Calculate the split index
    split_idx = int(len(files) * train_ratio)
    
    # Split the files
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    print(f"Total files: {len(files)}")
    print(f"Training files: {len(train_files)} ({len(train_files)/len(files)*100:.1f}%)")
    print(f"Validation files: {len(val_files)} ({len(val_files)/len(files)*100:.1f}%)")
    
    # Copy training files
    print("Copying training files...")
    for file in tqdm(train_files):
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(train_dir, file)
        shutil.copy2(src_path, dst_path)
    
    # Copy validation files
    print("Copying validation files...")
    for file in tqdm(val_files):
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(val_dir, file)
        shutil.copy2(src_path, dst_path)
    
    print("Dataset split complete!")
    print(f"Training files saved to: {train_dir}")
    print(f"Validation files saved to: {val_dir}")
    
    # Create metadata file with information about the split
    metadata_path = os.path.join(target_dir, "split_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Dataset split performed on: {os.path.basename(source_dir)}\n")
        f.write(f"Total files: {len(files)}\n")
        f.write(f"Training files: {len(train_files)} ({len(train_files)/len(files)*100:.1f}%)\n")
        f.write(f"Validation files: {len(val_files)} ({len(val_files)/len(files)*100:.1f}%)\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Train ratio: {train_ratio}\n\n")
        
        # List training files
        f.write("Training files:\n")
        for file in sorted(train_files):
            f.write(f"- {file}\n")
        
        # List validation files
        f.write("\nValidation files:\n")
        for file in sorted(val_files):
            f.write(f"- {file}\n")
    
    print(f"Split metadata saved to: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into training and validation sets')
    parser.add_argument('--source', default='demo/data/train_raw/new/',
                        help='Source directory containing dataset files')
    parser.add_argument('--target', default='../../code/simple_model/data',
                        help='Target directory for train and val subdirectories')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of files to use for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.source):
        print(f"Error: Source directory {args.source} does not exist")
        return
    
    # Perform the split
    split_dataset(args.source, args.target, args.train_ratio, args.seed)

if __name__ == "__main__":
    main()