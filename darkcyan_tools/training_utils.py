import getpass
import json
import shutil
from datetime import datetime
from pathlib import Path

from blessed import Terminal
from rich import print
from rich.progress import Progress

from darkcyan.config import Config
from darkcyan.constants import (
    DEFAULT_YOLO_TRAINING_CONFIG,
    YOLOBATCHSIZEMAP,
    DataType,
    YoloBaseModels,
)

from .local_data_utils import init_directories

term = Terminal()


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
    config["config_creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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


def create_training_zipfile(version, type):
    """Create a zip file for the classification dataset, expect version to have lb suffix for letterbox"""
    init_directories()
    temp_dir = Path(Config.get_value("temp_dir"))

    input_directory = temp_dir / get_training_data_src_directory(version, type)
    if not input_directory.exists():
        print(term.red(f"No data found for {version}, expected it {input_directory}"))
        return
    final_zip_filename = temp_dir / get_training_zip_name(version, type, True)
    if final_zip_filename.exists():
        print(
            term.red(f"Zip file already exists for {version} in {final_zip_filename}")
        )
        return final_zip_filename

    with Progress(transient=False) as progress:
        task1 = progress.add_task(
            f"[blue]Creating zip file for {version} from {input_directory}", total=None
        )
        zip_filename = temp_dir / get_training_zip_name(version, type)
        zipfile = shutil.make_archive(
            zip_filename, "zip", root_dir=input_directory, base_dir="."
        )
        progress.update(task1, completed=1)
        return zipfile


def main():
    create_config_file("4.1lb", DataType.cls, YoloBaseModels.nano)


if __name__ == "__main__":
    main()