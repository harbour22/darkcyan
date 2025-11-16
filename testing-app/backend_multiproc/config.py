from pathlib import Path

# Video + YOLO configuration shared by supervisor and worker processes.

YOLO_MODEL_PATH = Path(
    "/Users/chris/Documents/developer/darkcyan_data/engines/det/yolov8_4.15_large-det.mlpackage"
)

YOLO_INPUT_WIDTH = 640
YOLO_MIN_CONF = 0.3
YOLO_NUM_WORKERS = 2  # processes handled by supervisor
DISPLAY_MAX_WIDTH = 1024

# Source map – each entry becomes its own worker process.
VIDEO_SOURCES = {
    "cam1": "/Users/chris/Documents/developer/darkcyan_data/test_data/video/Reolink4kFront-20230910-191600.mp4",
    "cam2": "/Users/chris/Documents/developer/darkcyan_data/test_data/video/Reolink4kFront-20230910-191600.mp4",
}

# JPEG encode quality for WebSocket delivery.
JPEG_QUALITY = 85
