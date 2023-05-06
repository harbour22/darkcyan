import os
import shutil
from pathlib import Path

from blessed import Terminal
from rich import print
from rich.progress import Progress
from rich.tree import Tree

from .config import Config
from .constants import DataTag, DataType

term = Terminal()
local_data_repository = Path(Config.get_value("local_data_repository"))
scratch_dir = Path(Config.get_value("scratch_dir"))
data_prefix = Config.get_value("data_prefix")
temp_dir = Path(Config.get_value("temp_dir"))


def init_directories():
    if not local_data_repository.exists():
        print(f"Local main repository {local_data_repository} doesn't exist, creating")
        os.makedirs(local_data_repository)
    if not scratch_dir.exists():
        print(f"Local scratch repository {scratch_dir} doesn't exist, creating")
        os.makedirs(scratch_dir)
    if not temp_dir.exists():
        print(f"Local temp directory {scratch_dir} doesn't exist, creating")
        os.makedirs(temp_dir)


def prepare_working_directory(base_version, new_version, type=DataType.det):
    """Prepare the working directory for the given version"""
    init_directories()
    working_dir = get_local_scratch_directory_for_version(new_version, type)
    with Progress(transient=True) as progress:
        task1 = progress.add_task(
            f"[blue]Unpacking {base_version} zipfile for {type.name}...", total=None
        )
        shutil.unpack_archive(
            get_local_zipfile_for_version(base_version, type), working_dir
        )
        progress.update(task1, completed=1)


def create_main_from_scratch(version, type=DataType.det):
    """Prepare the working directory for the given version"""
    init_directories()
    working_dir = get_local_scratch_directory_for_version(version, type)
    data_filename = f"{data_prefix}_v{version}_{type.name}"
    target_filename = get_local_zipfile_for_version(version, type, False)

    with Progress(transient=True) as progress:
        task1 = progress.add_task("[blue]Creating zipfile...", total=None)
        shutil.make_archive(
            target_filename, "zip", root_dir=working_dir, base_dir=".", dry_run=False
        )
        progress.update(task1, completed=1)


def remove_scratch_version(scratch_version, type=DataType.det):
    init_directories()
    working_dir = scratch_dir / f"{data_prefix}_v{scratch_version}_{type.name}"
    if working_dir.exists():
        with Progress(transient=True) as progress:
            task1 = progress.add_task("[blue]Removing scratch version..", total=None)
            shutil.rmtree(working_dir)
            progress.update(task1, completed=1)

    else:
        print(f"Scratch version {working_dir} doesn't exist, skipping")


def clear_temp_directory():
    shutil.rmtree(temp_dir)
    init_directories()


def get_source_data_name(version, type=DataType.det, include_suffix=True):
    if include_suffix:
        return Path(f"{data_prefix}_v{version}_{type.name}.zip")
    else:
        return Path(f"{data_prefix}_v{version}_{type.name}")


def get_local_zipfile_for_version(
    version, type=DataType.det, include_suffix=True, tag=DataTag.main
):
    if tag == DataTag.main:
        parent = local_data_repository
    elif tag == DataTag.scratch:
        parent = scratch_dir
    else:
        parent = temp_dir
    source_data_filename = get_source_data_name(version, type, include_suffix)
    source_data_file = parent / source_data_filename
    return source_data_file


def get_local_scratch_directory_for_version(version, type):
    source_data_filename = get_source_data_name(version, type, False)
    scratch_data_directory = scratch_dir / source_data_filename
    return scratch_data_directory


def get_available_data(type=DataType.det, version=DataTag.main):
    """Query the local data cache and return available data"""
    init_directories()
    if version == DataTag.main:
        repo_to_query = local_data_repository
    elif version == DataTag.scratch:
        repo_to_query = scratch_dir
    else:
        (version == DataTag.temp)
        repo_to_query = temp_dir
    data = []
    for file in repo_to_query.glob(f"{data_prefix}*{type.name}*"):
        if file.suffix == ".zip" or version == DataTag.scratch:
            data.append(repo_to_query / file)

    return data


def get_available_data_versions(type=DataType.det, tag=DataTag.main):
    init_directories()
    available_data = get_available_data(type, tag)
    versions = []
    for file in available_data:
        versions.append((file.name.split("_")[1])[1:])
    versions.sort()
    return versions


def display_available_data():
    """Query the local data cache and display available data"""
    tree_root = Tree(term.black_on_cyan("Available Data"))

    for tag in DataTag:
        if tag == DataTag.main:
            type_root = tree_root.add(
                term.darkcyan(f"Master Data ") + term.cyan(f"({local_data_repository})")
            )
        elif tag == DataTag.scratch:
            type_root = tree_root.add(
                term.darkcyan(f"Draft Data ") + term.cyan(f"({scratch_dir})")
            )
        else:
            type_root = tree_root.add(
                term.darkcyan(f"Local Training/Temp Data ") + term.cyan(f"({temp_dir})")
            )

        for datatype in DataType:
            version_root = type_root.add(datatype.name)
            versions = get_available_data_versions(datatype, tag)
            for version in versions:
                version_root.add(term.green(f"{version}"))

    print(tree_root)


if __name__ == "__main__":
    display_available_data()
