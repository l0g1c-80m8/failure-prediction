#!/bin/bash

# Define variables
IMAGE_NAME="vsc-failure-prediction"
IMAGE_TAG="latest"
DOCKERFILE_PATH="Dockerfile"
BUILD_CONTEXT="./"

set -x

# Build the Docker image
docker build \
  -f "$DOCKERFILE_PATH" \
  -t "$IMAGE_NAME:$IMAGE_TAG" \
  "$BUILD_CONTEXT"

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Docker build succeeded"
else
  echo "Docker build failed"
  exit 1
fi