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


def calculate_volume_weighted_center_of_mass(vertices, faces):
    """
    Calculate the center of mass using volume-weighted tetrahedra.
    This accounts for hollow regions in the mesh.
    
    Args:
        vertices: np.array of vertex coordinates
        faces: list of face definitions (using vertex indices)
    
    Returns:
        np.array: Center of mass coordinates [x, y, z]
    """
    import numpy as np
    
    # Extract the vertex indices from the face data structure
    triangles = []
    for face_indices, _ in faces:
        if len(face_indices) == 3:
            # Already a triangle
            triangles.append([face_indices[0]-1, face_indices[1]-1, face_indices[2]-1])  # Convert to 0-indexed
        elif len(face_indices) > 3:
            # Triangulate the face (simple fan triangulation)
            for i in range(1, len(face_indices)-1):
                triangles.append([face_indices[0]-1, face_indices[i]-1, face_indices[i+1]-1])  # Convert to 0-indexed
    
    # Choose an arbitrary interior point (e.g., the average of all vertices)
    interior_point = np.mean(vertices, axis=0)
    
    # Initialize variables for weighted sum calculation
    total_volume = 0.0
    weighted_position = np.zeros(3)
    
    # Process each triangle
    for triangle in triangles:
        # Get vertices of the triangle
        v1 = vertices[triangle[0]]
        v2 = vertices[triangle[1]]
        v3 = vertices[triangle[2]]
        
        # Form a tetrahedron with the interior point
        tetra_volume = calculate_tetrahedron_volume(interior_point, v1, v2, v3)
        
        # Calculate centroid of the tetrahedron
        tetra_centroid = (interior_point + v1 + v2 + v3) / 4.0
        
        # Add weighted contribution
        weighted_position += tetra_volume * tetra_centroid
        total_volume += tetra_volume
    
    # Guard against division by zero
    if total_volume < 1e-10:
        return np.mean(vertices, axis=0)  # Fall back to simple averaging
    
    # Return the volume-weighted center of mass
    return weighted_position / total_volume

def calculate_tetrahedron_volume(p1, p2, p3, p4):
    """
    Calculate the volume of a tetrahedron defined by four points.
    
    Args:
        p1, p2, p3, p4: np.arrays with coordinates [x, y, z]
    
    Returns:
        float: Volume of the tetrahedron
    """
    import numpy as np
    
    # Calculate vectors from p1 to other points
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1
    
    # Calculate volume using triple product
    volume = abs(np.dot(np.cross(v1, v2), v3)) / 6.0
    return volume

# def translate_vertices(vertices, translation_vector):
#     """Translate vertices by a given vector."""
#     return vertices - translation_vector


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
    """Process an OBJ file to center its mass at the origin using volume-weighted calculation."""
    
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
    
    # Calculate center of mass using the improved method
    center_of_mass = calculate_volume_weighted_center_of_mass(vertices, faces)
    
    # For comparison, also calculate using the simple method
    simple_center = np.mean(vertices, axis=0)
    
    if verbose:
        print(f"Original center of mass (simple average): {simple_center}")
        print(f"Original center of mass (volume-weighted): {center_of_mass}")
    
    # Translate vertices to place center of mass at origin
    centered_vertices = vertices - center_of_mass
    
    # Verify the new center of mass is at origin
    new_center = calculate_volume_weighted_center_of_mass(centered_vertices, faces)
    
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
    input = "model/universal_robots_ur5e/assets/teamug_old.obj"
    output = "model/universal_robots_ur5e/assets/teamug.obj"
    quiet = True
    
    try:
        # center_of_mass, output_path = process_obj_file(args.input, args.output, not args.quiet)
        center_of_mass, output_path = process_obj_file(input, output, quiet)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()