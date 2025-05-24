#!/bin/bash

# Setup RPi dependencies if needed

if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "Running on Raspberry Pi. Executing additional script..."
    ./raspberry_pi_deps.sh
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

# Function to clone or update repository
clone_or_update() {
    local repo_dir="$1"
    local repo_url="$2"
    
    if [ -d "$repo_dir" ]; then
        echo "Updating $repo_dir..."
        git -C --depth=1 "$repo_dir" pull
    else
        echo "Cloning $repo_dir..."
        git clone --depth=1 "$repo_url" "$repo_dir"
    fi
}

# Clone or update OpenCV repositories
echo "Setting up OpenCV repositories..."
clone_or_update "$OPENCV_DIR" "https://github.com/opencv/opencv.git"
clone_or_update "$OPENCV_CONTRIB_DIR" "https://github.com/opencv/opencv_contrib.git"

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
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=~/local/opencv \
    -D OPENCV_EXTRA_MODULES_PATH="$OPENCV_CONTRIB_DIR/modules" \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENMP=ON \
    -D WITH_OPENCL=OFF \
    -D WITH_TBB=ON \
    -D BUILD_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_VTK=OFF \
    -D WITH_PROTOBUF=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D WITH_GSTREAMER=ON \
    -D WITH_OPENGL=ON \
    -D BUILD_EXAMPLES=OFF \
    -B "$BUILD_DIR" \
    -S "$OPENCV_DIR"

# Build and install
echo "Building OpenCV..."
make -C "$BUILD_DIR" -j$(($(nproc) - 1)) install
