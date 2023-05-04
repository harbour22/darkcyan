from enum import Enum
from pathlib import Path

DEFAULT_CONFIG_DIR = Path.home() / ".darkcyan"

DataType = Enum("DataType", ["det", "cls"])
DataTag = Enum("DataTag", ["main", "scratch", "temp"])
YoloBaseModels = Enum("YoloBaseModels", ["xlarge", "large", "medium", "small", "nano"])

YOLOBATCHSIZEMAP = {
    DataType.cls: {
        YoloBaseModels.xlarge: 96,
        YoloBaseModels.large: 128,
        YoloBaseModels.medium: 128,
        YoloBaseModels.small: 128,
        YoloBaseModels.nano: 256,
    }
}

YOLOMODELMAP = {
    DataType.cls: {
        YoloBaseModels.xlarge: "yolov8x-cls.pt",
        YoloBaseModels.large: "yolov8l-cls.pt",
        YoloBaseModels.medium: "yolov8m-cls.pt",
        YoloBaseModels.small: "yolov8s-cls.pt",
        YoloBaseModels.nano: "yolov8n-cls.pt",
    }
}

DEFAULT_DATA_PREFIX = "limetree"
DEFAULT_DET_SRC_NAME = "images_det_src"
DEFAULT_CLS_SRC_NAME = "images_cls_src"
DEFAULT_CLASSES_TXT = "classes.txt"
GOOGLEDRIVE_DATA_ROOT = "/content/drive/MyDrive"
DEFAULT_GOOGLEDRIVE_YOLO_DATA_DIR = "yolo/training_data"
DEFAULT_GOOGLEDRIVE_YOLO_CONFIG_DIR = "yolo/runtime_config"
DEFAULT_GOOGLEDRIVE_YOLO_TRAINING_OUTPUT_DIR = "yolo/training_output"
DEFAULT_GOOGLEDRIVE_YOLO_ENGINE_DIR = "yolo/engines"
DEFAULT_GOOGLEDRIVE_SCOPE = ["https://www.googleapis.com/auth/drive"]

DEFAULT_YOLO_TRAINING_CONFIG = f"{DEFAULT_DATA_PREFIX}_yolo_training_config.json"
