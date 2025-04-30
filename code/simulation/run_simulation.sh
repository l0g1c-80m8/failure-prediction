#!/bin/bash

# Directory containing the obj files
OBJ_DIR="model/universal_robots_ur5e/assets/get_a_grip_mesh"

# Path to the XML file
XML_FILE="model/universal_robots_ur5e/test_scene_complete.xml"

# Path to the config file
CONFIG_FILE="demo/simulation_config.json"

# Path to the object property file
OBJ_PROP_FILE="demo/object_properties.json"

# NPY directory containing processed objects
NPY_DIR="demo/data/train_raw/new/"

# Raw data directory to clean up
RAW_DATA_DIR="demo/data/train_raw/"

# File to store processed object names
PROCESSED_OBJECTS_FILE="demo/processed_objects.txt"

# Backup directory for removed files (optional)
BACKUP_DIR="demo/data/train_raw/backup"

# Log file for the simulation run
LOG_FILE="demo/simulation_run_$(date +%Y%m%d_%H%M%S).log"

# Check if directory exists
if [ ! -d "$OBJ_DIR" ]; then
  echo "Error: Directory $OBJ_DIR does not exist"
  exit 1
fi

if [ ! -d "$RAW_DATA_DIR" ]; then
  echo "Error: Directory $RAW_DATA_DIR does not exist"
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

# Make sure the NPY directories exist
mkdir -p "$NEW_NPY_DIR"
mkdir -p "$BACKUP_DIR"

# Run the script to extract processed object names
echo "Extracting processed object names from NPY files..."
python3 demo/utils/check_processed_files.py --npy_dir "$NPY_DIR" --output "$PROCESSED_OBJECTS_FILE"

# Ask user if they want to clean up unprocessed files
echo ""
echo "Do you want to clean up unprocessed files in $RAW_DATA_DIR? (y/n)"
read -r clean_choice

if [[ "$clean_choice" == "y" || "$clean_choice" == "Y" ]]; then
  # Ask if this should be a dry run
  echo "Do you want to do a dry run first (no actual deletion)? (y/n)"
  read -r dry_run_choice
  
  if [[ "$dry_run_choice" == "y" || "$dry_run_choice" == "Y" ]]; then
    # Do a dry run first
    echo "Performing dry run cleanup..."
    python3 demo/utils/cleanup_unprocessed_files.py --raw_data_dir "$RAW_DATA_DIR" \
      --processed_objects_file "$PROCESSED_OBJECTS_FILE" \
      --backup_dir "$BACKUP_DIR"
    
    # Ask if they want to proceed with actual deletion
    echo ""
    echo "Do you want to proceed with actual cleanup? (y/n)"
    read -r proceed_choice
    
    if [[ "$proceed_choice" == "y" || "$proceed_choice" == "Y" ]]; then
      echo "Performing actual cleanup..."
      python3 demo/utils/cleanup_unprocessed_files.py --raw_data_dir "$RAW_DATA_DIR" \
        --processed_objects_file "$PROCESSED_OBJECTS_FILE" \
        --backup_dir "$BACKUP_DIR" \
        --no-dry-run
    else
      echo "Skipping actual cleanup."
    fi
  else
    # Do the actual cleanup directly
    echo "Performing cleanup..."
    python3 demo/utils/cleanup_unprocessed_files.py --raw_data_dir "$RAW_DATA_DIR" \
      --processed_objects_file "$PROCESSED_OBJECTS_FILE" \
      --backup_dir "$BACKUP_DIR" \
      --no-dry-run
  fi
fi

# Ask if user wants to continue with the simulation
echo ""
echo "Do you want to continue with the simulation? (y/n)"
read -r sim_choice

if [[ "$sim_choice" != "y" && "$sim_choice" != "Y" ]]; then
  echo "Exiting without running the simulation."
  exit 0
fi

# Count total objects to process
TOTAL_OBJECTS=$(find "$OBJ_DIR" -name "*.obj" -type f | wc -l)
echo "Total objects to process: $TOTAL_OBJECTS"

# Initialize counters
PROCESSED=0
SKIPPED=0
FAILED=0

# Start logging
echo "=== Simulation started at $(date) ===" > "$LOG_FILE"
echo "OBJ directory: $OBJ_DIR" >> "$LOG_FILE"
echo "NPY directory: $NEW_NPY_DIR" >> "$LOG_FILE"
echo "Total objects: $TOTAL_OBJECTS" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"


# List all .obj files and process them one by one
echo "OBJ files found in $OBJ_DIR:"
find "$OBJ_DIR" -name "*.obj" -type f | while read file; do
  basename=$(basename "$file" .obj)
  echo "Processing: $basename"

  # Check if object has already been processed
  if grep -q "^$basename$" "$PROCESSED_OBJECTS_FILE"; then
    echo "Object $basename has already been processed. Skipping..."
    echo "-----------------------------------"
    continue
  fi
  # if [[ "$basename" == "core-pillow-4c617e5be0596ee2685998681d42efb8"* ]]; then

  #   # Call the Python script to update XML file
  #   python3 update_xml.py "$XML_FILE" --obj_dir "$OBJ_DIR" --obj_name "$basename" --prop_file "$OBJ_PROP_FILE"

  #   # Call the Python script to update config file
  #   python3 update_config.py "$CONFIG_FILE" --obj_name "$basename" --prop_file "$OBJ_PROP_FILE" --xml_file "${XML_FILE%.*}_tmp.${XML_FILE##*.}"

  #   python3 demo/historical_backtracking_bin.py --config "${CONFIG_FILE%.*}_tmp.${CONFIG_FILE##*.}"
  #   break
  # fi

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