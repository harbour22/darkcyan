#!/bin/bash

# Setup RPi dependencies if needed

if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "Running on Raspberry Pi. Executing additional script..."
    ./build_scripts/raspberry_pi_deps.sh
else
    echo "Not a Raspberry Pi. Skipping additional script."
fi

# Directory setup
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="${SCRIPT_DIR}/.."
SCRATCH_DIR="${PROJECT_ROOT}/scratch"
OPENCV_DIR="${SCRATCH_DIR}/opencv"
OPENCV_CONTRIB_DIR="${SCRATCH_DIR}/opencv_contrib"
BUILD_DIR="${SCRATCH_DIR}/build"
OPENCV_TAG="tags/4.10.0"

# Function to clone or update repository
clone_or_update() {
    local repo_dir="$1"
    local repo_url="$2"
    
    if [ -d "$repo_dir" ]; then
        echo "Updating $repo_dir..."
        git -C "$repo_dir" pull
    else
        echo "Cloning $repo_dir..."
        git clone "$repo_url" "$repo_dir"
    fi
}

# Clone or update OpenCV repositories
echo "Setting up OpenCV repositories..."
clone_or_update "$OPENCV_DIR" "https://github.com/opencv/opencv.git"
clone_or_update "$OPENCV_CONTRIB_DIR" "https://github.com/opencv/opencv_contrib.git"

git -C "$OPENCV_DIR" checkout "$OPENCV_TAG"
git -C "$OPENCV_CONTRIB_DIR" checkout "$OPENCV_TAG"

# Clean build directory if it exists
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create scratch directory if it doesn't exist
mkdir -p "$SCRATCH_DIR"

# Create build directory
mkdir -p "$BUILD_DIR"

echo "Configuring build with CMake..."

export BUILD_DIR
export OPENCV_DIR
export OPENCV_CONTRIB_DIR

python $SCRIPT_DIR/configure_opencv.py $OPENCV_DIR $OPENCV_CONTRIB_DIR --build-dir=$BUILD_DIR

# Build and install
echo "Building OpenCV..."
make -C "$BUILD_DIR" -j$(($(nproc) - 1)) install

# cleaning (frees 320 MB)
make -C "$BUILD_DIR" clean
sudo apt-get update
