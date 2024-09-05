#!/bin/bash

# list of directories to search
directories=(
    "/code/simulation"
    "/code/octo"
)

# function to remove environments
remove_envs() {
    local dir="$1"
    echo "Searching in $dir"

    # find and remove directories ending with _env
    find "$dir" -type d -name "*_env" -print -exec rm -rf {} +
}

# main script
echo "Starting environment removal process..."

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        remove_envs "$dir"
    else
        echo "Directory not found: $dir"
    fi
done

echo "Environment removal process completed."
