import sys
import sysconfig
import subprocess
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Configure OpenCV build with custom Python environment.")
    parser.add_argument("opencv_dir", type=Path, help="Path to OpenCV source directory")
    parser.add_argument("opencv_contrib_dir", type=Path, help="Path to opencv_contrib source directory")
    parser.add_argument("--build-dir", type=Path, default=Path.cwd() / "build",
                        help="Path to build directory (default: ./build)")
    args = parser.parse_args()

    args.build_dir.mkdir(parents=True, exist_ok=True)

    python_exec = sys.executable
    include_dir = sysconfig.get_paths()['include']
    packages_dir = sysconfig.get_paths()['purelib']

    print(f"Using Python executable: {python_exec}")
    print(f"Python include dir: {include_dir}")
    print(f"Python packages dir: {packages_dir}")

    cmake_command = [
        "cmake",
        "-D", "CMAKE_BUILD_TYPE=RELEASE",
        "-D", f"CMAKE_INSTALL_PREFIX={str(Path.home() / 'local/opencv')}",
        "-D", f"OPENCV_EXTRA_MODULES_PATH={args.opencv_contrib_dir / 'modules'}",
        "-D", "INSTALL_PYTHON_EXAMPLES=OFF",
        "-D", "INSTALL_C_EXAMPLES=OFF",
        "-D", "BUILD_opencv_python2=OFF",
        "-D", "BUILD_opencv_python3=ON",
        "-D", "OPENCV_ENABLE_NONFREE=ON",
        "-D", "WITH_QT=OFF",
        "-D", "WITH_OPENMP=ON",
        "-D", "WITH_OPENCL=OFF",
        "-D", "WITH_TBB=ON",
        "-D", "BUILD_TBB=ON",
        "-D", "WITH_V4L=ON",
        "-D", "WITH_LIBV4L=ON",
        "-D", "WITH_VTK=OFF",
        "-D", "WITH_PROTOBUF=ON",
        "-D", f"PYTHON3_EXECUTABLE={python_exec}",
        "-D", f"PYTHON3_INCLUDE_DIR={include_dir}",
        "-D", f"PYTHON3_PACKAGES_PATH={packages_dir}",
        "-D", "WITH_GSTREAMER=ON",
        "-D", "WITH_OPENGL=ON",
        "-D", "BUILD_EXAMPLES=OFF",
        "-B", str(args.build_dir),
        "-S", str(args.opencv_dir)
    ]

    print("\nRunning CMake command:\n" + " \\\n    ".join(cmake_command))
    subprocess.run(cmake_command, check=True)

if __name__ == "__main__":
    main()
