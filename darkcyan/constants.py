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
    },
    DataType.det: {
        YoloBaseModels.xlarge: 16,
        YoloBaseModels.large: 24,
        YoloBaseModels.medium: 32,
        YoloBaseModels.small: 64,
        YoloBaseModels.nano: 128,
    },
}

YOLOMODELMAP = {
    "v8": {
        DataType.cls: {
            YoloBaseModels.xlarge: "yolov8x-cls.pt",
            YoloBaseModels.large: "yolov8l-cls.pt",
            YoloBaseModels.medium: "yolov8m-cls.pt",
            YoloBaseModels.small: "yolov8s-cls.pt",
            YoloBaseModels.nano: "yolov8n-cls.pt",
        },
        DataType.det: {
            YoloBaseModels.xlarge: "yolov8x.pt",
            YoloBaseModels.large: "yolov8l.pt",
            YoloBaseModels.medium: "yolov8m.pt",
            YoloBaseModels.small: "yolov8s.pt",
            YoloBaseModels.nano: "yolov8n.pt",
        },
    },
    "v9": {
        DataType.det: {
            YoloBaseModels.xlarge: "yolov9e.pt",
            YoloBaseModels.large: "yolov9c.pt",
            YoloBaseModels.medium: "yolov9m.pt",
            YoloBaseModels.small: "yolov9s.pt",
            YoloBaseModels.nano: "yolov9t.pt",
        }
    },
    "v10": {
        DataType.det: {
            YoloBaseModels.xlarge: "yolov10n.pt",
            YoloBaseModels.large: "yolov10s.pt",
            YoloBaseModels.medium: "yolov10m.pt",
            YoloBaseModels.small: "yolov10l.pt",
            YoloBaseModels.nano: "yolov10x.pt",
        }
    }    
}

DEFAULT_DATA_PREFIX = "limetree"
DEFAULT_DET_SRC_NAME = "images_det_src"
DEFAULT_CLS_SRC_NAME = "images_cls_src"
DEFAULT_CLASSES_TXT = "classes.txt"
GOOGLEDRIVE_CONTENT_ROOT = "/content"
GOOGLEDRIVE_SRC_TRAINING_DATA_ROOT = f"{GOOGLEDRIVE_CONTENT_ROOT}/drive/MyDrive"
DEFAULT_TRAINING_YOLO_DATA_DIR = "yolo/training_data"
DEFAULT_TRAINING_YOLO_CONFIG_DIR = "yolo/runtime_config"
DEFAULT_TRAINING_YOLO_OUTPUT_DIR = "yolo/training_output"
DEFAULT_TRAINING_OUTPUT_YOLO_ENGINE_DIR = "yolo/engines"


DEFAULT_GOOGLEDRIVE_SCOPE = ["https://www.googleapis.com/auth/drive"]

DEFAULT_YOLO_TRAINING_CONFIG = f"{DEFAULT_DATA_PREFIX}_yolo_training_config.json"
DEFAULT_DET_TRAINING_YAML = "darkcyan.yaml"

RUNTIME_CONFIG_FILE_PREFIX = 'magenta'
DEFAULT_RUNTIME_CONFIG_FILE = f'{RUNTIME_CONFIG_FILE_PREFIX}-defaults.yaml'

ENVIRONMENTS = ['prod', 'test']
