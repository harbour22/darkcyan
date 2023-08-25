import getpass
import json
import os
from datetime import datetime
from pathlib import Path

from rich import print

from .constants import DEFAULT_CONFIG_DIR, DataTag, GOOGLEDRIVE_SRC_TRAINING_DATA_ROOT

DEFAULT_CONFIG = {
    f"scratch_dir": (
        Path.home() / "developer" / "darkcyan_data" / "scratch"
    ).as_posix(),
    "temp_dir": (Path.home() / "developer" / "darkcyan_data" / "temp").as_posix(),
    "local_data_repository": (
        Path.home() / "developer" / "darkcyan_data" / "main"
    ).as_posix(),
    "data_prefix": "limetree",
    "labelImg_cmd": "labelImg",
    "cls_test_ratio": 0.2,
    "det_test_ratio": 0.2,
    "training_epochs": 350,
    "training_data_root": GOOGLEDRIVE_SRC_TRAINING_DATA_ROOT if os.getenv("COLAB_RELEASE_TAG") else "."
}


class Config:
    _config = None

    def config():
        if Config._config is None:
            Config._config = Config.get_config()
        return Config._config

    def get_value(key):
        if key not in Config.config():
            print(f"Config key {key} not found, attempting defaults")
            if key not in DEFAULT_CONFIG:
                raise KeyError(f"Config key {key} not found in defaults")
            else:
                Config.config()[key] = DEFAULT_CONFIG[key]
                Config.save_config()
        return Config.config()[key]

    @staticmethod
    def init_config():
        config_filename = DEFAULT_CONFIG_DIR / "config.json"
        if not config_filename.exists():
            print(f"Config file {config_filename} does not exist, creating")
            os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
            with open(config_filename, "w", encoding="utf-8") as f:
                data = {
                    "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "created_by": getpass.getuser(),
                    "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def save_config():
        config_filename = DEFAULT_CONFIG_DIR / "config.json"
        with open(config_filename, "w", encoding="utf-8") as f:
            data = Config.config()
            data["last_modified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def get_config():
        """Get the configuration for the application"""
        Config.init_config()
        config_filename = DEFAULT_CONFIG_DIR / "config.json"

        with open(config_filename) as config_file:
            app_config = json.load(config_file)

        return app_config


if __name__ == "__main__":
    print(Config.config())
    print(Config.get_value("scratch_dir"))
