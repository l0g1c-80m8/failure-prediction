#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as ET
import re
import json
import argparse
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='Update MuJoCo XML file with obj meshes')
    parser.add_argument('xml_file', help='Path to the XML file to modify')
    parser.add_argument('--obj_name', required=True,
                        help='Name of the obj file (without .obj extension)')
    parser.add_argument('--obj_dir', required=True,
                        help='Dir of the obj file (without .obj extension)')
    parser.add_argument('--prop_file', default='object_properties.json',
                        help='Path to the object_properties.json file (default: object_properties.json)')
    parser.add_argument('--output', default=None,
                        help='Output file path (default: overwrite input file)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Get the obj file name
    obj_name = args.obj_name
    obj_dir = args.obj_dir

    sub_folder = obj_dir.split("/")[-1]
    
    try:
        # Read the object properties file
        try:
            with open(args.prop_file, 'r') as f:
                object_properties = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Object properties file {args.prop_file} not found. Only updating current_object.")
            object_properties = {}

        # If the object exists in the properties file
        if obj_name in object_properties:
            obj_size_scale = object_properties[obj_name]['size_variation']
            print(f"Object {obj_name} already exists in config, use existing size scale")
        else:
            # Object not found in properties file, add default properties
            obj_size_scale = [1.0, 1.7]
            default_properties = {
                    "base_mass": 0.5,
                    "mass_variation": [0.8, 1.2],
                    "base_size": 1,
                    "size_variation": [0.05, 0.1],
                    "friction_variation": [0.25, 0.35],
                    "x_offset": [-0.04, 0.04],
                    "y_offset": [-0.04, 0.04],
                    "z_offset": [0.05, 0.06],
                    "qpos": [[0, 0, 1, 1]]
                }
            object_properties[obj_name] = default_properties
            print(f"Object {obj_name} not found in properties file, added default properties")

            # Save the updated properties back to the file
            try:
                with open(args.prop_file, 'w') as f:
                    json.dump(object_properties, f, indent=4)
                print(f"Updated properties file {args.prop_file} with default properties for {obj_name}")
            except Exception as e:
                print(f"Warning: Could not update properties file: {e}")

        obj_size_variation = random.uniform(obj_size_scale[0], obj_size_scale[1])  # n% variation


        # Parse the XML file
        tree = ET.parse(args.xml_file)
        root = tree.getroot()
        
        # Find the asset element
        asset = root.find('.//asset')
        if asset is None:
            print(f"Error: Could not find <asset> element in XML file for {obj_name}.")
            sys.exit(1)
        
        # Check if material already exists
        material_name = f"{obj_name}_material"
        existing_material = asset.find(f"./material[@name='{material_name}']")
        if existing_material is None:
            # Create material element
            material = ET.SubElement(asset, 'material')
            material.set('class', 'ur5e')
            material.set('name', material_name)
            material.set('rgba', '0.49 0.0 0.0 1')
            print(f"Added material: {material_name}")
        
        # Check if mesh already exists
        existing_mesh = asset.find(f"./mesh[@file='{sub_folder}/{obj_name}.obj']")
        if existing_mesh is None:
            # Create mesh element
            mesh = ET.SubElement(asset, 'mesh')
            mesh.set('file', f'{sub_folder}/{obj_name}.obj')
            mesh.set('scale', f'{obj_size_variation} {obj_size_variation} {obj_size_variation}')
            print(f"Added mesh: {sub_folder}/{obj_name}.obj")
        
        # Find and update the object_collision geom
        object_collision = root.find(".//geom[@name='object_collision']")
        if object_collision is not None:
            object_collision.set('mesh', obj_name)
            print(f"Updated object_collision mesh to: {obj_name}")
        else:
            print(f"Warning: Could not find geom with name='object_collision' for {obj_name}")
        
        # Save the modified XML
        if args.output:
            output_file = args.output
        else:
            # Create a tmp filename by adding "_tmp" before the extension
            base_name, extension = os.path.splitext(args.xml_file)
            output_file = f"{base_name}_tmp{extension}"
        
        # Convert the XML to string, preserving formatting as much as possible
        xml_str = ET.tostring(root, encoding='unicode')
        
        # Pretty print the XML with proper indentation
        try:
            import xml.dom.minidom
            xml_pretty = xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ")
            
            # Remove extra blank lines that minidom sometimes adds
            xml_pretty = "\n".join([line for line in xml_pretty.split("\n") if line.strip()])
            
            with open(output_file, 'w') as f:
                f.write(xml_pretty)
        except:
            # Fallback if minidom fails
            print(f"Warning: Pretty printing failed for {obj_name}, saving raw XML")
            with open(output_file, 'w') as f:
                f.write(xml_str)
        
        print(f"Updated XML saved to: {output_file} with {obj_name}")
        
    except ET.ParseError as e:
        print(f"Error parsing XML file for {obj_name}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred processing {obj_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()