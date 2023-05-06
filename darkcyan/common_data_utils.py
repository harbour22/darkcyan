import shutil
from pathlib import Path

from blessed import Terminal
from rich.progress import Progress

from .config import Config
from .darkcyan_training_utils import (
    get_training_data_src_directory,
    get_training_zip_name,
)
from .local_data_utils import get_local_scratch_directory_for_version, init_directories

term = Terminal()


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
