import getpass
import json
from datetime import datetime
from pathlib import Path

from rich import print

from .config import Config
from .constants import (
    DEFAULT_YOLO_TRAINING_CONFIG,
    YOLOBATCHSIZEMAP,
    DataType,
    YoloBaseModels,
)


def create_config_file(version, type=DataType.det, basemodel=YoloBaseModels.nano):
    print(f"Creating config file for {type.name} {version} {basemodel.name}")
    config = {}
    config["version"] = version
    config["type"] = type.name
    config["basemodel"] = basemodel.name
    config["epochs"] = Config.get_value("training_epochs")
    config["imgsz"] = 224 if type == DataType.cls else 640
    config["batchsize"] = YOLOBATCHSIZEMAP[type][basemodel]
    config["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["created_by"] = getpass.getuser()
    print(config)

    temp_dir = Path(Config.get_value("temp_dir"))
    config_file = temp_dir / DEFAULT_YOLO_TRAINING_CONFIG

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    return config_file


def main():
    create_config_file("4.1lb", DataType.cls, YoloBaseModels.nano)


if __name__ == "__main__":
    main()
