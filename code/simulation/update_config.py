#!/usr/bin/env python3
import os
import sys
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Update simulation config JSON with object properties')
    parser.add_argument('config_file', help='Path to the simulation_config.json file to modify')
    parser.add_argument('--obj_name', required=True,
                        help='Name of the object to set as current object')
    parser.add_argument('--prop_file', default='object_properties.json',
                        help='Path to the object_properties.json file (default: object_properties.json)')
    parser.add_argument('--xml_file', default=None,
                        help='Model file path (default: create temporary file with _tmp suffix)')
    parser.add_argument('--output', default=None,
                        help='Output file path (default: create temporary file with _tmp suffix)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Get the object name
    obj_name = args.obj_name
    
    try:
        # Read the current simulation config file
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
        
        # Read the object properties file
        try:
            with open(args.prop_file, 'r') as f:
                object_properties = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Object properties file {args.prop_file} not found. Only updating current_object.")
            object_properties = {}
        
        # Update the xml file
        config_data['simulation_related']['xml_path'] = args.xml_file

        # Update the current object in the config
        if 'object_related' in config_data:
            config_data['object_related']['current_object'] = obj_name
            print(f"Updated current_object to: {obj_name}")
            
            # If the object exists in the properties file but not in the config, add it
            if obj_name in object_properties and obj_name not in config_data['object_related']:
                config_data['object_related'][obj_name] = object_properties[obj_name]
                print(f"Added properties for {obj_name} to the config")
            elif obj_name in object_properties:
                # Object exists in both - no need to update as it's already there
                print(f"Object {obj_name} already exists in config, keeping existing properties")
            else:
                # Object not found in properties file, add default properties
                default_properties = {
                    "base_mass": 0.5,
                    "mass_variation": [0.8, 1.2],
                    "base_size": 1,
                    "size_variation": [1.0, 1.0],
                    "friction_variation": [0.25, 0.35],
                    "x_offset": [-0.04, 0.04],
                    "y_offset": [-0.04, 0.04],
                    "z_offset": [0.05, 0.06],
                    "qpos": [[0, 0, 1, 1]]
                }
                config_data['object_related'][obj_name] = default_properties
                print(f"Object {obj_name} not found in properties file, added default properties")
        else:
            print("Error: 'object_related' section not found in config file")
            sys.exit(1)
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            # Create a tmp filename by adding "_tmp" before the extension
            base_name, extension = os.path.splitext(args.config_file)
            output_file = f"{base_name}_tmp{extension}"
        
        # Write the updated config to file
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"Updated config saved to: {output_file}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()