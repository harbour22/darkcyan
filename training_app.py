from darkcyan_tools.classify_data_utilities import (
    create_yolo_classification_dataset,
)
from darkcyan_tools.training_utils import create_training_zipfile
from darkcyan.config import Config
from darkcyan.constants import (
    DEFAULT_CLASSES_TXT,
    DEFAULT_DET_SRC_NAME,
    DEFAULT_TRAINING_YOLO_CONFIG_DIR,
    DEFAULT_TRAINING_YOLO_DATA_DIR,
    YOLOMODELMAP,
    DataTag,
    DataType,
    YoloVersion,
    YoloBaseModels,
)
from darkcyan_tools.training_utils import create_config_file
from darkcyan_tools.detection_data_utilities \
    import create_yolo_detection_dataset
from darkcyan_tools.google_drive_utils import (
    delete_file,
    get_directory_id_from_path,
    get_file_id,
    upload_file,
)
from darkcyan_tools.local_data_utils import (
    clear_temp_directory,
    create_main_from_scratch,
    display_available_data,
    get_available_data_versions,
    get_local_scratch_directory_for_version,
    get_local_zipfile_for_version,
    prepare_working_directory,
    remove_scratch_version,
)

import difflib
import shutil
import sys
from pathlib import Path
from collections.abc import Iterable


from blessed import Terminal
from rich.progress import Progress

import functools

echo = functools.partial(print, end='', flush=True)
term = Terminal()


def ask_for_dataset_type():
    print()
    print(
        term.magenta(
            f"What type of dataset? (1: {DataType.det.name}, 2: {DataType.cls.name})"
        )
    )

    inp = ""
    with term.cbreak():
        inp = term.inkey()
    print(term.darkcyan(f"{inp}"))

    datatype = DataType.__call__(int(inp))
    return datatype


def ask_for_yolo_version(dataType):
    print()
    print(term.magenta(f"Select Yolo Version for {dataType.name} training"))
    for choice, yoloVersion in enumerate(YOLOMODELMAP[dataType], start=1):
        print(term.magenta(f"{choice}: {yoloVersion.name}"))

    with term.cbreak():
        choice = term.inkey()
        if choice not in [str(i) for i in range(1, len(YoloBaseModels) + 1)]:
            print(term.red(f"Illogical choice {choice}"))
            return
    # never going to work...
    yoloVersion = list(YOLOMODELMAP[dataType])[int(choice)-1]

    return yoloVersion


def ask_for_data_version(datatype, tag):
    versions = get_available_data_versions(datatype, tag)
    if len(versions) == 0:
        print(term.red(f"No data found, any key to continue"))
        return None

    print(term.magenta(f"Choose the {tag.name} version: "))
    for choice, version in enumerate(versions, start=1):
        print(term.magenta(f"{choice}: {version}"))

    with term.cbreak():
        version = versions[int(term.inkey()) - 1]
        print(term.darkcyan(f"{version}"))
        return version
    
def ask_for_yolo_model():
    print(term.magenta(f"Select model size to train"))
    for choice, model in enumerate(YoloBaseModels, start=1):
        print(term.magenta(f"{choice}: {model.name}"))

    with term.cbreak():
        choice = term.inkey()
        if choice not in [str(i) for i in range(1, len(YoloBaseModels) + 1)]:
            print(term.red(f"Illogical choice {choice}"))
            return None
    
        return YoloBaseModels(int(choice))
    


def upload_to_google_drive(datatype = None, version = None):
    if(datatype == None):
        datatype = ask_for_dataset_type()
    if(version == None):
        version = ask_for_data_version(datatype, DataTag.temp)

    file_to_upload = get_local_zipfile_for_version(version, datatype,
                                                   tag=DataTag.temp)
    if not file_to_upload.exists():
        print(term.red(f"{file_to_upload} not found to upload"))
        return
    parent_dir = get_directory_id_from_path(DEFAULT_TRAINING_YOLO_DATA_DIR)

    ## Delete existing files from google drive
    existing_files = get_file_id(file_to_upload.name, parent_dir)
    if existing_files:
        print(
            term.red(
                f"Found existing {len(existing_files)} file(s) with name \
                    {file_to_upload.name} - overwrite? (y/N)"
            )
        )
        with term.cbreak():
            overwrite = term.inkey()
            print(term.darkcyan(f"{overwrite}"))
            if overwrite == "y":
                for file in existing_files:
                    delete_file(file["id"])
                    print(term.magenta(f"Deleted {file}"))
            else:
                return

    upload_file(file_to_upload, parent_dir, mimetype="application/zip")


def get_integer_input():
    with term.cbreak():
        val = ''
        
        while True:
            inp = term.inkey()
            if(inp.code == term.KEY_ENTER):
                if(len(val) == 0):
                    print(term.red(f"Please enter an integer"))
                    continue
                else:
                    break
            if(inp.code == term.KEY_BACKSPACE):
                val = val[:-1]
                echo(u'\b \b')
                continue
            if(inp.isdigit()):
                val += inp
                echo(term.darkcyan(inp))
    return int(val)

def trim_detection_dataset():
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

    print(term.magenta(f"Choose the image directory: "))
    for choice, image_dir in enumerate(image_dirs, start=1):
        print(term.magenta(f"{choice}: {image_dir.name}"))
    inp = ""
    with term.cbreak():
        inp = term.inkey()
        print(term.darkcyan(f"{inp}"))

    image_dir = image_dirs[int(inp) - 1]
    
    ext = '.jpg'
    n_files = len([p for p in image_dir.iterdir() if p.suffix==f'{ext}'])
    print(term.magenta(f"Found {n_files} {ext} files"))

    print(term.magenta(f"How many images to keep?"))

    image_cnt = get_integer_input()

    counter = keep = 0
    keep_every_nth = int(n_files / image_cnt)
    print(keep_every_nth)

    for image_file in [p for p in image_dir.iterdir() if p.suffix==f'{ext}']:
        counter += 1
        yolo_file = image_file.with_suffix(".txt")

        if(counter % keep_every_nth == 0):            
            keep += 1
        else:            
            image_file.unlink()
            yolo_file.unlink()
            continue

    print(term.magenta(f"Kept {keep} images"))

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

    print(term.magenta(f"Choose the image directory: "))
    for choice, image_dir in enumerate(image_dirs, start=1):
        print(term.magenta(f"{choice}: {image_dir.name}"))
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
        print(labelImg_cmd, labelImg_args)

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
                            term.magenta(
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

    print(term.magenta(f"Choose major or minor version: "))
    print(
        term.magenta(
            f"1: Major Version Increment ({base_version} to {major_version+1}.0)"
        )
    )
    print(
        term.magenta(
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


def create_main_dataset_from_scratch(datatype = None, scratch_version = None):
    if(datatype == None):
        datatype = ask_for_dataset_type()
    if(scratch_version == None):
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
        print(term.magenta(f"Removed {source_data_name}"))
    else:
        print(term.red(f"No existing {source_data_name} found."))

    print()
    display_available_data()
    print()


def prepare_data_for_training(datatype = None, src_version = None, training_version = None):
    if(datatype==None):
        datatype = ask_for_dataset_type()
    if(src_version==None):
        src_version = training_version = ask_for_data_version(datatype, DataTag.scratch)
    if(training_version==None):
        training_version = ask_for_data_version(datatype, DataTag.scratch)        
    if datatype == DataType.det:
        create_yolo_detection_dataset(src_version, training_version, 980)
        create_training_zipfile(training_version, DataType.det)

    if datatype == DataType.cls:
        print(term.magenta(f"Create/use letterbox version? (y/n)"))
        use_letterbox = False
        with term.cbreak():
            letterbox = term.inkey()
            print(term.magenta(f"{letterbox}"))
            if letterbox == "y":
                use_letterbox = True
                training_version = f"{src_version}lb"

        create_yolo_classification_dataset(src_version, training_version, use_letterbox)
        create_training_zipfile(training_version, DataType.cls)


def remove_and_recreate_temp_directory():
    with Progress(transient=True) as progress:
        task1 = progress.add_task(
            f"[blue]Clearing and recreating the temp directory", total=None
        )
        clear_temp_directory()
        progress.update(task1, completed=1)
        print(term.darkcyan(f"Cleared and recreating the temp directory"))


def create_colab_training_config(datatype = None, version = None, yoloVersion = None, basemodel = None):

    if(datatype==None):
        datatype = ask_for_dataset_type()
    if(version == None):
        version = ask_for_data_version(datatype, DataTag.temp)
    if(yoloVersion == None):
        yoloVersion = ask_for_yolo_version(datatype)
    if(basemodel == None):
        basemodel = ask_for_yolo_model()

    if(None in (datatype, yoloVersion, basemodel)):
        print(term.magenta(f"One or more parameters missing ({datatype}, {yoloVersion}, {basemodel})"))
        return

    config_file = create_config_file(version, datatype, basemodel, yoloVersion)
    google_parent_dir = get_directory_id_from_path(
        DEFAULT_TRAINING_YOLO_CONFIG_DIR
    )
    ## Delete existing files from google drive
    existing_files = get_file_id(config_file.name, google_parent_dir)
    if existing_files:
        print(
            term.red(
                f"Found existing {len(existing_files)} file(s) with name {config_file.name} - deleting?"
            )
        )
        
        for file in existing_files:
            delete_file(file["id"])
            print(term.magenta(f"Deleted {file}"))

    upload_file(config_file, google_parent_dir, "application/json")
    print(term.magenta(f"Uploaded {config_file} to {google_parent_dir}"))

def run_build_chain():
    
        datatype = ask_for_dataset_type()
        if(datatype == None):
            print(term.magenta(f"Unable to run build chain"))
            return
        version = ask_for_data_version(datatype, DataTag.scratch)
        if(version == None):
            print(term.magenta(f"Unable to run build chain"))
            return
        yoloVersion = ask_for_yolo_version(datatype)
        if(yoloVersion == None):
            print(term.magenta(f"Unable to run build chain"))
            return
        basemodel = ask_for_yolo_model()
        if(basemodel == None):
            print(term.magenta(f"Unable to run build chain"))
            return

        prepare_data_for_training(datatype, version, version)
        upload_to_google_drive(datatype, version)
        create_colab_training_config(datatype, version, yoloVersion, basemodel)
        create_main_dataset_from_scratch(datatype, version)


def print_command_menu():
    print()

    for choice, description, func in command_options:
        if not choice:
            print(term.yellow(f"{description}"))
        else:
            print(term.blue(f"{choice}: {description}"))


def quit():
    print(term.magenta(f"Bye!"))
    sys.exit(0)


command_options = [
    ("", "=== Data Generation and Management ===", None),
    ("1", "Display available data", display_available_data),
    ("2", "Create local working copy of data", author_new_dataset),
    ("3", "Trim detection dataset", trim_detection_dataset),
    ("4", "Launch labelImg for detection data", run_labelimg),
    ("", "", None),

    ("", "=== Cloud Upload and Training ===", None),
    ("t", "Run training command chain (prepare, upload, command, build)", 
     run_build_chain),
    (
        "5",
        "Prepare data for training from local working copy",
        prepare_data_for_training,
    ),    
    ("6", "Upload data to google drive (on auth err rm token.json)", upload_to_google_drive),
    (
        "7",
        "Create colab training command and upload to google drive",
        create_colab_training_config,
    ),
    ("", "", None),
    ("", "=== Training Data Manipulation ===", None),
    (
        "8",
        "Build new master dataset from local working copy",
        create_main_dataset_from_scratch,
    ),
    ("9", "Remove local working copy", remove_working_copy_of_data),
    
    ("", "", None),    
    ("", "=== Utilities ===", None),
    ("c", "Clear temp directory", remove_and_recreate_temp_directory),
    ("m", "Show full menu", print_command_menu),
    ("q", "Quit", quit),
]

def run_functions(funcs):
    if isinstance(funcs, Iterable) and not isinstance(funcs, (str, bytes)):
        for f in funcs:
            if callable(f):
                f()
            else:
                print(term.red(f"Item {f} is not callable."))
    elif callable(funcs):
        funcs()
    else:
        print(term.red("Input is neither a function nor an iterable of functions."))


def run():
    print(term.clear())
    print(term.darkcyan(("DarkCyan Tools")))
    print_command_menu()
    while True:
        print()
        print(term.magenta(f"Select an option: {term.blue(f'(m for menu, q to quit)')}"))
        inp = ""
        with term.cbreak():
            inp = term.inkey()
            print(term.magenta(f"{inp}"))

            command = [command for command in command_options if command[0] == inp]
            if len(command) == 0:
                print(term.red(f"Illogical choice {inp}"))
                continue

            command = command[
                0
            ]  ## In case we have multiple commands mapped to the same key

            print(term.magenta(f"Running " + term.blue(f"{command[1]}")))
            run_functions(command[2])

if __name__ == "__main__":
    run()
