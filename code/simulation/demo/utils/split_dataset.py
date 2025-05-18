#!/usr/bin/env python3
import os
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm

def split_dataset(source_dir, target_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset files from source_dir into train, validation, and test sets
    and copy them to the respective subdirectories in target_dir.
    
    Args:
        source_dir: Directory containing the dataset files
        target_dir: Target directory where train, val, and test subdirectories will be created
        train_ratio: Ratio of files to use for training (default: 0.8)
        val_ratio: Ratio of files to use for validation (default: 0.1)
        test_ratio: Ratio of files to use for testing (default: 0.1)
        seed: Random seed for reproducibility
    """
    # Ensure ratios sum to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        print(f"Warning: Ratios do not sum to 1.0 (sum: {train_ratio + val_ratio + test_ratio})")
        print(f"Normalizing ratios...")
        total = train_ratio + val_ratio + test_ratio
        train_ratio = train_ratio / total
        val_ratio = val_ratio / total
        test_ratio = test_ratio / total
        print(f"Normalized ratios - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create target directories if they don't exist
    train_dir = os.path.join(target_dir, "train")
    val_dir = os.path.join(target_dir, "val")
    test_dir = os.path.join(target_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # List all NPY files in the source directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    
    if not files:
        print(f"No NPY files found in {source_dir}")
        return
    
    # Shuffle the files
    random.shuffle(files)
    
    # Calculate the split indices
    n_files = len(files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    # Split the files
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    print(f"Total files: {n_files}")
    print(f"Training files: {len(train_files)} ({len(train_files)/n_files*100:.1f}%)")
    print(f"Validation files: {len(val_files)} ({len(val_files)/n_files*100:.1f}%)")
    print(f"Test files: {len(test_files)} ({len(test_files)/n_files*100:.1f}%)")
    
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

    # Copy test files
    print("Copying test files...")
    for file in tqdm(test_files):
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(test_dir, file)
        shutil.copy2(src_path, dst_path)
    
    print("Dataset split complete!")
    print(f"Training files saved to: {train_dir}")
    print(f"Validation files saved to: {val_dir}")
    print(f"Test files saved to: {test_dir}")
    
    # Create metadata file with information about the split
    metadata_path = os.path.join(target_dir, "split_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Dataset split performed on: {os.path.basename(source_dir)}\n")
        f.write(f"Total files: {n_files}\n")
        f.write(f"Training files: {len(train_files)} ({len(train_files)/n_files*100:.1f}%)\n")
        f.write(f"Validation files: {len(val_files)} ({len(val_files)/n_files*100:.1f}%)\n")
        f.write(f"Test files: {len(test_files)} ({len(test_files)/n_files*100:.1f}%)\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Train ratio: {train_ratio}\n")
        f.write(f"Validation ratio: {val_ratio}\n")
        f.write(f"Test ratio: {test_ratio}\n\n")
        
        # List training files
        f.write("Training files:\n")
        for file in sorted(train_files):
            f.write(f"- {file}\n")
        
        # List validation files
        f.write("\nValidation files:\n")
        for file in sorted(val_files):
            f.write(f"- {file}\n")

        # List test files
        f.write("\nTest files:\n")
        for file in sorted(test_files):
            f.write(f"- {file}\n")
    
    print(f"Split metadata saved to: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into training, validation, and test sets')
    parser.add_argument('--source', default='demo/data/train_raw/new/',
                        help='Source directory containing dataset files')
    parser.add_argument('--target', default='../../code/simple_model/data',
                        help='Target directory for train, val, and test subdirectories')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of files to use for training (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of files to use for validation (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Ratio of files to use for testing (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.source):
        print(f"Error: Source directory {args.source} does not exist")
        return
    
    # Perform the split
    split_dataset(args.source, args.target, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

if __name__ == "__main__":
    main()