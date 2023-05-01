from pathlib import Path
from enum import Enum


DEFAULT_CONFIG_DIR = Path.home() / '.darkcyan'

DataType = Enum('DataType', ['det','cls'])
DataTag = Enum('DataTag',['main', 'scratch'])

DEFAULT_DATA_SUFFIX = 'limetree'
DEFAULT_DET_SRC_NAME = 'images_det_src'
DEFAULT_CLASSES_TXT = 'classes.txt'