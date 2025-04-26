#!/bin/bash

# Directory containing the obj files
OBJ_DIR="model/universal_robots_ur5e/assets/get_a_grip_mesh"

# Path to the XML file
XML_FILE="model/universal_robots_ur5e/test_scene_complete.xml"

# Path to the config file
CONFIG_FILE="demo/simulation_config.json"

# Path to the object property file
OBJ_PROP_FILE="demo/object_properties.json"

# Check if directory exists
if [ ! -d "$OBJ_DIR" ]; then
  echo "Error: Directory $OBJ_DIR does not exist"
  exit 1
fi

# Check if XML file exists
if [ ! -f "$XML_FILE" ]; then
  echo "Error: XML file $XML_FILE does not exist"
  exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file $CONFIG_FILE does not exist"
  exit 1
fi

# List all .obj files and process them one by one
echo "OBJ files found in $OBJ_DIR:"
find "$OBJ_DIR" -name "*.obj" -type f | while read file; do
  basename=$(basename "$file" .obj)
  echo "Processing: $basename"

  # Call the Python script to update XML file
  python3 update_xml.py "$XML_FILE" --obj_dir "$OBJ_DIR" --obj_name "$basename" --prop_file "$OBJ_PROP_FILE"

  # Call the Python script to update config file
  python3 update_config.py "$CONFIG_FILE" --obj_name "$basename" --prop_file "$OBJ_PROP_FILE" --xml_file "${XML_FILE%.*}_tmp.${XML_FILE##*.}"

  python3 demo/historical_backtracking_bin.py --config "${CONFIG_FILE%.*}_tmp.${CONFIG_FILE##*.}"
  echo "-----------------------------------"
done

echo "All objects have been processed."

# Make sure all files have proper permissions
chmod -R 755 model/ 2>/dev/null
chmod 644 "$CONFIG_FILE" 2>/dev/null
echo "File permissions updated."