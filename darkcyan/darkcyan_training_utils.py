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
    config["training_data"] = get_training_zip_name(version, type, True)
    config["basemodel"] = basemodel.name
    config["epochs"] = Config.get_value("training_epochs")
    config["imgsz"] = 224 if type == DataType.cls else 640
    config["batchsize"] = YOLOBATCHSIZEMAP[type][basemodel]
    config["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["created_by"] = getpass.getuser()
    print(config)

    temp_dir = Path(Config.get_value("temp_dir"))
    config_file = temp_dir / DEFAULT_YOLO_TRAINING_CONFIG

    save_config(config, config_file)

    return config_file


def save_config(config, dest_file):
    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def get_training_zip_name(version, type=DataType.det, append_extension=False):
    return f"{Config.get_value('data_prefix')}_v{version}_{type.name}{'.zip' if append_extension else ''}"


def get_training_data_src_directory(training_version, type):
    temp_dir = Path(Config.get_value("temp_dir"))
    return (
        temp_dir
        / f"{Config.get_value('data_prefix')}_v{training_version}_{type.name}_train"
    )


def main():
    create_config_file("4.1lb", DataType.cls, YoloBaseModels.nano)


if __name__ == "__main__":
    main()
