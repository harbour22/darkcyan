import difflib
import shutil
import sys
from pathlib import Path

from blessed import Terminal
from rich.progress import Progress

from darkcyan.classify_data_utilities import (
    create_or_get_classification_zipfile,
    create_yolo_classification_dataset,
)
from darkcyan.config import Config
from darkcyan.constants import (
    DEFAULT_CLASSES_TXT,
    DEFAULT_DET_SRC_NAME,
    DEFAULT_GOOGLEDRIVE_YOLO_CONFIG_DIR,
    DEFAULT_GOOGLEDRIVE_YOLO_DATA_DIR,
    DataTag,
    DataType,
    YoloBaseModels,
)
from darkcyan.darkcyan_training_utils import create_config_file
from darkcyan.google_drive_utils import (
    delete_file,
    get_directory_id_from_path,
    get_file_id,
    upload_file,
)
from darkcyan.local_data_utils import (
    clear_temp_directory,
    create_main_from_scratch,
    display_available_data,
    get_available_data_versions,
    get_local_scratch_directory_for_version,
    get_local_zipfile_for_version,
    prepare_working_directory,
    remove_scratch_version,
)

term = Terminal()


def ask_for_dataset_type():
    print()
    print(
        term.white(
            f"What type of dataset? (1: {DataType.det.name}, 2: {DataType.cls.name})"
        )
    )

    inp = ""
    with term.cbreak():
        inp = term.inkey()
    print(term.darkcyan(f"{inp}"))

    datatype = DataType.__call__(int(inp))
    return datatype


def ask_for_data_version(datatype, tag):
    versions = get_available_data_versions(datatype, tag)
    if len(versions) == 0:
        print(term.red(f"No data found, any key to continue"))
        return None

    print(term.white(f"Choose the {tag.name} version: "))
    for choice, version in enumerate(versions, start=1):
        print(term.white(f"{choice}: {version}"))

    with term.cbreak():
        version = versions[int(term.inkey()) - 1]
        print(term.darkcyan(f"{version}"))
        return version


def upload_to_google_drive():
    datatype = ask_for_dataset_type()
    version = ask_for_data_version(datatype, DataTag.temp)

    file_to_upload = get_local_zipfile_for_version(version, datatype, tag=DataTag.temp)
    if not file_to_upload.exists():
        print(term.red(f"{file_to_upload} not found to upload"))
        return

    parent_dir = get_directory_id_from_path(DEFAULT_GOOGLEDRIVE_YOLO_DATA_DIR)

    ## Delete existing files from google drive
    existing_files = get_file_id(file_to_upload.name, parent_dir)
    if existing_files:
        print(
            term.red(
                f"Found existing {len(existing_files)} file(s) with name {file_to_upload.name} - overwrite? (y/N)"
            )
        )
        with term.cbreak():
            overwrite = term.inkey()
            print(term.darkcyan(f"{overwrite}"))
            if overwrite == "y":
                for file in existing_files:
                    delete_file(file["id"])
                    print(term.white(f"Deleted {file}"))
            else:
                return

    upload_file(file_to_upload, parent_dir, mimetype="application/zip")


def run_labelimg():
    data_version = ask_for_data_version(DataType.det, DataTag.scratch)

    data_directory = get_local_scratch_directory_for_version(data_version, DataType.det)

    parent_directory = data_directory / DEFAULT_DET_SRC_NAME

    image_dirs = []
    for file in Path(parent_directory).iterdir():
        if file.is_dir():
            image_dirs.append(file)

    if len(image_dirs) == 0:
        print(term.red(f"No images found"))
        return

    print(term.white(f"Choose the image directory: "))
    for choice, image_dir in enumerate(image_dirs, start=1):
        print(term.white(f"{choice}: {image_dir.name}"))
    inp = ""
    with term.cbreak():
        inp = term.inkey()
        print(term.darkcyan(f"{inp}"))

    image_dir = image_dirs[int(inp) - 1]
    print(image_dir)

    if not (image_dir / DEFAULT_CLASSES_TXT).exists():
        print(
            term.red(
                f"No classes.txt found in {image_dir}, copying from {parent_directory}"
            )
        )
        shutil.copy(
            parent_directory / DEFAULT_CLASSES_TXT, image_dir / DEFAULT_CLASSES_TXT
        )

    import subprocess
    import sys

    labelImg_cmd = Config.get_value("labelImg_cmd")
    labelImg_args = f"{image_dir.as_posix()} {(parent_directory / DEFAULT_CLASSES_TXT).as_posix()} {image_dir.as_posix()}"
    with Progress(transient=True) as progress:
        task1 = progress.add_task(
            f"[blue]Running imageLbl for version {data_version}", total=None
        )

        pipe = subprocess.Popen(
            f"{labelImg_cmd} {labelImg_args}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output = pipe.communicate()[0].decode("utf-8")
        output_stderr = pipe.communicate()[1].decode("utf-8")
        progress.update(task1, completed=1)

    print()
    with open(parent_directory / DEFAULT_CLASSES_TXT, "r") as parent_classes:
        with open(image_dir / DEFAULT_CLASSES_TXT, "r") as child_classes:
            diff = difflib.unified_diff(
                parent_classes.readlines(),
                child_classes.readlines(),
                fromfile="Parent Classes.txt",
                tofile="Edited Classes.txt",
            )
            edits = False
            for line in diff:
                sys.stdout.write(line)
                edits = True
            if edits:
                print(
                    term.red(f"Changes detected, do you want to copy to parent? (y/n)")
                )
                with term.cbreak():
                    key = term.inkey()
                    print(term.darkcyan(f"{key}"))

                    if key == "y":
                        shutil.copy(
                            image_dir / DEFAULT_CLASSES_TXT,
                            parent_directory / DEFAULT_CLASSES_TXT,
                        )
                        print(
                            term.white(
                                f"Copied {image_dir / DEFAULT_CLASSES_TXT} to {parent_directory / DEFAULT_CLASSES_TXT}"
                            )
                        )


def author_new_dataset():
    datatype = ask_for_dataset_type()

    versions = get_available_data_versions(datatype)
    if len(versions) == 0:
        print(term.red(f"No data found, any key to continue"))
        with term.cbreak():
            term.inkey()
            return

    base_version = ask_for_data_version(datatype, DataTag.main)
    if base_version is None:
        return

    major_version = int(base_version.split(".")[0])
    minor_version = int(base_version.split(".")[1])

    print(term.white(f"Choose major or minor version: "))
    print(
        term.white(
            f"1: Major Version Increment ({base_version} to {major_version+1}.0)"
        )
    )
    print(
        term.white(
            f"2: Minor Version Increment ({base_version} to {major_version}.{minor_version+1})"
        )
    )
    with term.cbreak():
        choice = term.inkey()
    if choice == "1":
        working_version = f"{major_version+1}.0"
    elif choice == "2":
        working_version = f"{major_version}.{minor_version+1}"
    else:
        print(term.red(f"Illogical choice {choice}, any key to continue"))
        with term.cbreak():
            term.inkey()
        return

    if working_version in versions:
        print(
            term.red(f"Version v{working_version} already exists, any key to continue")
        )
        with term.cbreak():
            term.inkey()
        return

    scratch_versions = get_available_data_versions(datatype, DataTag.scratch)

    if working_version in scratch_versions:
        print(
            term.red(
                f"Version v{working_version} already exists in scratch - overwrite? (y/N)"
            )
        )
        with term.cbreak():
            overwrite = term.inkey()
            print(term.darkcyan(f"{overwrite}"))
            if overwrite != "y":
                return
            remove_scratch_version(working_version, datatype)

    print()
    print(term.darkcyan(f"Creating new dataset v{working_version}"))
    print()
    """ Create the local working directory """
    print(term.black_on_darkcyan(("Creating the local working copy")))
    print()
    prepare_working_directory(base_version, working_version, datatype)
    print(term.black_on_darkcyan(("Done.")))
    print()
    display_available_data()
    print()


def create_main_dataset_from_scratch():
    datatype = ask_for_dataset_type()
    scratch_version = ask_for_data_version(datatype, DataTag.scratch)
    if scratch_version is None:
        return

    zipfile = get_local_zipfile_for_version(scratch_version, datatype)
    if zipfile.exists():
        print(term.red(f"Found existing {zipfile} - overwrite? (y/N)"))
        with term.cbreak():
            overwrite = term.inkey()
            print(term.darkcyan(f"{overwrite}"))
            if overwrite != "y":
                return
            print(term.red(f"Overwriting {zipfile}"))

    create_main_from_scratch(scratch_version, datatype)
    print()
    display_available_data()
    print()


def remove_working_copy_of_data():
    datatype = ask_for_dataset_type()
    scratch_version = ask_for_data_version(datatype, DataTag.scratch)
    if scratch_version is None:
        return

    source_data_name = get_local_scratch_directory_for_version(
        scratch_version, datatype
    )
    if source_data_name.exists():
        remove_scratch_version(scratch_version, datatype)
        print(term.white(f"Removed {source_data_name}"))
    else:
        print(term.red(f"No existing {source_data_name} found."))

    print()
    display_available_data()
    print()


def prepare_data_for_training():
    datatype = ask_for_dataset_type()
    if datatype == DataType.det:
        # unsupported
        print(term.red(f"{DataType.det.name} unsupported for now, any key to continue"))
        return

    version = ask_for_data_version(datatype, DataTag.scratch)

    if datatype == DataType.cls:
        print(term.white(f"Use letterbox images? (y/n)"))
        letterbox = True
        with term.cbreak():
            letterbox = term.inkey()
            print(term.white(f"{letterbox}"))
            if letterbox != "y":
                letterbox = False
        create_yolo_classification_dataset(version, letterbox)
        create_or_get_classification_zipfile(version, letterbox)


def remove_and_recreate_temp_directory():
    with Progress(transient=True) as progress:
        task1 = progress.add_task(
            f"[blue]Clearing and recreating the temp directory", total=None
        )
        clear_temp_directory()
        progress.update(task1, completed=1)
        print(term.darkcyan(f"Cleared and recreating the temp directory"))


def create_colab_training_config():
    datatype = ask_for_dataset_type()
    version = ask_for_data_version(datatype, DataTag.temp)

    print(term.white(f"Select modelsize to train"))
    for choice, model in enumerate(YoloBaseModels, start=1):
        print(term.white(f"{choice}: {model.name}"))

    with term.cbreak():
        choice = term.inkey()
        if choice not in [str(i) for i in range(1, len(YoloBaseModels) + 1)]:
            print(term.red(f"Illogical choice {choice}"))
            return

        basemodel = YoloBaseModels(int(choice))

        config_file = create_config_file(version, datatype, basemodel)
        google_parent_dir = get_directory_id_from_path(
            DEFAULT_GOOGLEDRIVE_YOLO_CONFIG_DIR
        )
        ## Delete existing files from google drive
        existing_files = get_file_id(config_file.name, google_parent_dir)
        if existing_files:
            print(
                term.red(
                    f"Found existing {len(existing_files)} file(s) with name {config_file.name} - overwrite? (y/N)"
                )
            )
            with term.cbreak():
                overwrite = term.inkey()
                print(term.darkcyan(f"{overwrite}"))
                if overwrite == "y":
                    for file in existing_files:
                        delete_file(file["id"])
                        print(term.white(f"Deleted {file}"))
                else:
                    return

        upload_file(config_file, google_parent_dir, "application/json")
        print(term.white(f"Uploaded {config_file} to {google_parent_dir}"))


def print_command_menu():
    print()

    for choice, description, func in command_options:
        if not choice:
            print(term.yellow(f"{description}"))
        else:
            print(term.white(f"{choice}: {description}"))


def quit():
    print(term.white(f"Bye!"))
    sys.exit(0)


command_options = [
    ("", "=== Data Generation and Management ===", None),
    ("1", "Display available data", display_available_data),
    ("2", "Create local working copy of data", author_new_dataset),
    ("3", "Launch labelImg for detection data", run_labelimg),
    ("", "", None),
    ("", "=== Training Data Manipulation ===", None),
    (
        "4",
        "Build new master dataset from local working copy",
        create_main_dataset_from_scratch,
    ),
    ("5", "Remove local working copy", remove_working_copy_of_data),
    (
        "6",
        "Prepare data for training from local working copy",
        prepare_data_for_training,
    ),
    ("", "", None),
    ("", "=== Cloud Upload and Training ===", None),
    ("7", "Upload data to google drive", upload_to_google_drive),
    (
        "8",
        "Create colab training command and upload to google drive",
        create_colab_training_config,
    ),
    ("", "", None),
    ("", "=== Utilities ===", None),
    ("9", "Clear temp directory", remove_and_recreate_temp_directory),
    ("m", "Show full menu", print_command_menu),
    ("q", "Quit", quit),
]


def run():
    print(term.clear())
    print(term.black_on_darkcyan(("DarkCyan Tools")))
    print_command_menu()
    while True:
        print()
        print(term.white(f"Select an option: {term.blue(f'(m for menu, q to quit)')}"))
        inp = ""
        with term.cbreak():
            inp = term.inkey()
            print(term.white(f"{inp}"))

            command = [command for command in command_options if command[0] == inp]
            if len(command) == 0:
                print(term.red(f"Illogical choice {inp}"))
                continue

            command = command[
                0
            ]  ## In case we have multiple commands mapped to the same key

            print(term.white(f"Running " + term.blue(f"{command[1]}")))
            command[2]()


if __name__ == "__main__":
    run()
