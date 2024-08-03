import os
import sys
import time

import getpass
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch

from blessed import Terminal
from rich import print
from rich.progress import Progress

from ultralytics import YOLO

from darkcyan.config import Config
from .local_data_utils import init_directories

from darkcyan.constants import DEFAULT_TRAINING_YOLO_DATA_DIR, \
                               DEFAULT_TRAINING_YOLO_CONFIG_DIR, \
                               DEFAULT_TRAINING_YOLO_OUTPUT_DIR, \
                               DEFAULT_YOLO_TRAINING_CONFIG, \
                               DEFAULT_TRAINING_OUTPUT_YOLO_ENGINE_DIR, \
                               YOLOMODELMAP, \
                               YoloVersion, \
                               YOLOBATCHSIZEMAP, \
                               YoloBaseModels, \
                               DataType, \
                               DEFAULT_DET_TRAINING_YAML


def get_platform():
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'Darwin',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    
    return platforms[sys.platform]


platform = get_platform()
term = Terminal()


def train():
    """Train the model"""
    print("Training the model")
    if os.getenv("COLAB_RELEASE_TAG"):
        print(f"Running in Colab on {platform}")
        in_colab = True
    else:
        in_colab = False
        print(f"Running on {platform}")
    training_data_root = Path(Config.get_value("training_data_root"))
    temp_dir_root = Path(Config.get_value("temp_dir"))

    ## Load config
    config_file = training_data_root / \
        DEFAULT_TRAINING_YOLO_CONFIG_DIR / \
        DEFAULT_YOLO_TRAINING_CONFIG

    print(f'Loading config from {config_file}')

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(json.dumps(config, indent=4))

    data_type = DataType[config['type']]
    model_size = YoloBaseModels[config['basemodel']]
    version = config['version']
    batch = config['batchsize']
    epochs = config['epochs']
    imgsz = config['imgsz']
    yolo_version = YoloVersion[config.get('yolov', 'v8')]

    project_path = Path(training_data_root) / \
                DEFAULT_TRAINING_YOLO_OUTPUT_DIR / \
                config['type'] / f"darkcyan_{config['version']}" / \
                model_size.name

    data = data_path = temp_dir_root / f"{data_type.name}_{version}_training_data"
    

    if data_type == DataType.det:
        data = data / DEFAULT_DET_TRAINING_YAML

    if(data_path.exists()):
        print(f"Training data found at {data_path}")
    else:
        zip_filename = Path(training_data_root) / DEFAULT_TRAINING_YOLO_DATA_DIR / config['training_data']
        print(f"Unzipping training data {zip_filename} to {data_path}")
        shutil.unpack_archive(zip_filename,data_path)

    last_run = project_path / 'train' / 'weights' / 'last.pt'
    if last_run.exists():
        base_model = last_run
        resume=True
    else:
        base_model = YOLOMODELMAP[data_type][yolo_version][model_size]
        resume=False

    print(base_model, last_run, resume)

    model = YOLO(base_model) # pass any model type
    mps_available = torch.backends.mps.is_available()

    if not mps_available:
        print("MPS not available")
    else:
        print("MPS is available")


    start_time = time.time()
    if(mps_available):
        model.train(epochs=epochs, resume=resume, project=project_path.as_posix(), batch=batch, data=data.as_posix(), imgsz=imgsz, exist_ok = True, device='mps')
    else:
        model.train(epochs=epochs, resume=resume, project=project_path.as_posix(), batch=batch, data=data.as_posix(), imgsz=imgsz, exist_ok = True)
    end_time = time.time()     


    engine_file_name = f"yolov8_{config['version']}_{config['basemodel']}-{config['type']}.pt"
    config_file_name = f"yolov8_{config['version']}_{config['basemodel']}-{config['type']}.json"
    training_output = project_path / 'train' / 'weights' / 'best.pt'
    engine_dir = Path(training_data_root) / DEFAULT_TRAINING_OUTPUT_YOLO_ENGINE_DIR
    engine_output = engine_dir / engine_file_name
    config_output = Path(training_data_root) / DEFAULT_TRAINING_OUTPUT_YOLO_ENGINE_DIR / config_file_name

    if(not Path(engine_dir).exists()):
        engine_dir.mkdir(parents=True)

    shutil.copy(training_output, engine_output)

    config['output_engine'] = engine_file_name
    config['colab_version'] = os.environ['COLAB_RELEASE_TAG']
    config['elapsed_training_time_mins'] = f'{(end_time - start_time)/60:.2f}'
    config["training_end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_config(config, config_output)

def create_config_file(version, type=DataType.det, basemodel=YoloBaseModels.nano, yoloVersion=YoloBaseModels.medium):
    print(f"Creating config file for {type.name} {version} {basemodel.name}")
    config = {}
    config["version"] = version
    config["type"] = type.name
    config["yolov"] = yoloVersion
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
    train()
    #create_config_file("4.1lb", DataType.cls, YoloBaseModels.nano)


if __name__ == "__main__":
    main()
