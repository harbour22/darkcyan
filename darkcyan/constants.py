from pathlib import Path
from enum import Enum


DEFAULT_CONFIG_DIR = Path.home() / '.darkcyan'

DataType = Enum('DataType', ['det','cls'])
DataTag = Enum('DataTag',['main', 'scratch'])

DEFAULT_DATA_SUFFIX = 'limetree'
