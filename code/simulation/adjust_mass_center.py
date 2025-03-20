#!/usr/bin/env python3
"""
OBJ Center of Mass Normalizer

This script reads an OBJ file, calculates its center of mass (assuming uniform density),
and creates a new OBJ file with vertices translated so the center of mass is at the origin.

The script preserves all the original file information including:
- Comments and other metadata
- Face definitions including texture/normal coordinates
- Original vertex order

Usage:
    python obj_center_of_mass.py input.obj [output.obj]

If output.obj is not specified, it will create a file with "_centered" appended to the input filename.
"""

import sys
import os
import numpy as np
from argparse import ArgumentParser


def parse_obj(file_path):
    """Parse an OBJ file and extract vertices, faces, and other content."""
    vertices = []
    faces = []
    other_lines = []
    vertex_lines = []  # Store full vertex lines for format preservation
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse vertex line
            if line.startswith('v '):
                vertex_lines.append(line)
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            
            # Parse face line
            elif line.startswith('f '):
                parts = line.split()
                # OBJ format uses 1-indexed vertices
                # Handle potential texture/normal coordinates (e.g., "1/1/1")
                face = []
                for p in parts[1:]:
                    face_indices = p.split('/')
                    face.append(int(face_indices[0]))
                faces.append((face, line))  # Store full line for format preservation
            
            # Keep other lines as they are (comments, materials, etc.)
            else:
                other_lines.append(line)
    
    return np.array(vertices), faces, other_lines, vertex_lines


def calculate_center_of_mass(vertices):
    """Calculate the center of mass for a set of vertices (assuming uniform density)."""
    if len(vertices) == 0:
        return np.array([0.0, 0.0, 0.0])
    return np.mean(vertices, axis=0)


def translate_vertices(vertices, translation_vector):
    """Translate vertices by a given vector."""
    return vertices - translation_vector


def write_obj(file_path, vertices, faces, other_lines=None, original_file_path=None, center_of_mass=None):
    """Write vertices and faces to a new OBJ file."""
    with open(file_path, 'w') as f:
        # Write header comments
        f.write("# OBJ file centered by obj_center_of_mass.py\n")
        if original_file_path:
            f.write(f"# Original file: {os.path.basename(original_file_path)}\n")
        if center_of_mass is not None:
            f.write(f"# Original center of mass: {center_of_mass[0]:.6f} {center_of_mass[1]:.6f} {center_of_mass[2]:.6f}\n")
        f.write(f"# Vertices have been translated to place center of mass at origin\n")
        f.write(f"# vertex count = {len(vertices)}\n")
        f.write(f"# face count = {len(faces)}\n")
        
        # Write custom metadata lines
        if other_lines:
            for line in other_lines:
                if line.startswith('#'):
                    f.write(f"{line}\n")
        
        # Write vertices with precision
        for i, v in enumerate(vertices):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (preserving original format including texture/normal coordinates)
        for face_info, original_line in faces:
            f.write(original_line + "\n")


def process_obj_file(input_path, output_path=None, verbose=True):
    """Process an OBJ file to center its mass at the origin."""
    # Determine output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_centered{ext}"
    
    if verbose:
        print(f"Processing: {input_path}")
        print(f"Output will be saved to: {output_path}")
    
    # Parse the OBJ file
    vertices, faces, other_lines, vertex_lines = parse_obj(input_path)
    
    if verbose:
        print(f"Found {len(vertices)} vertices and {len(faces)} faces")
    
    # Calculate center of mass
    center_of_mass = calculate_center_of_mass(vertices)
    
    if verbose:
        print(f"Original center of mass: {center_of_mass}")
    
    # Translate vertices to place center of mass at origin
    centered_vertices = translate_vertices(vertices, center_of_mass)
    
    # Verify the new center of mass is at origin
    new_center = calculate_center_of_mass(centered_vertices)
    
    if verbose:
        print(f"New center of mass: {new_center} (should be close to [0, 0, 0])")
        
        # Print some sample vertices before and after centering
        print("\nSample vertices before and after centering:")
        for i in range(min(3, len(vertices))):
            print(f"  Original vertex {i+1}: {vertices[i]}")
            print(f"  Centered vertex {i+1}: {centered_vertices[i]}")
    
    # Write the new OBJ file
    write_obj(output_path, centered_vertices, faces, other_lines, input_path, center_of_mass)
    
    if verbose:
        print(f"\nCentered OBJ file saved to: {output_path}")
    
    return center_of_mass, output_path


def main():
    """Main function to handle command line arguments."""
    # parser = ArgumentParser(description="Center an OBJ file by its center of mass")
    # parser.add_argument("input", help="Input OBJ file", default="model/universal_robots_ur5e/assets/cube.obj")
    # parser.add_argument("output", nargs="?", help="Output OBJ file (optional)", default="model/universal_robots_ur5e/assets/cube_new.obj")
    # parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    
    # args = parser.parse_args()
    input = "model/universal_robots_ur5e/assets/stanford_bunny.obj"
    output = "model/universal_robots_ur5e/assets/stanford_bunny_new.obj"
    quiet = True
    
    try:
        # center_of_mass, output_path = process_obj_file(args.input, args.output, not args.quiet)
        center_of_mass, output_path = process_obj_file(input, output, quiet)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()