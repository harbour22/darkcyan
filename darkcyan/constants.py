from pathlib import Path
from enum import Enum


DEFAULT_CONFIG_DIR = Path.home() / '.darkcyan'

DataType = Enum('DataType', ['det','cls'])
DataTag = Enum('DataTag',['main', 'scratch'])

DEFAULT_DATA_SUFFIX = 'limetree'
DEFAULT_DET_SRC_NAME = 'images_det_src'
DEFAULT_CLS_SRC_NAME = 'images_cls_src'
DEFAULT_CLASSES_TXT = 'classes.txt'

DEFAULT_GOOGLEDRIVE_YOLO_DIR = 'yolo/training_data'
DEFAULT_GOOGLEDRIVE_SCOPE = ['https://www.googleapis.com/auth/drive']
