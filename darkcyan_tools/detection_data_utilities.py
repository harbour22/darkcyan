import math
import random
import shutil
from pathlib import Path

import PIL
import yaml
from blessed import Terminal
from PIL import Image, ImageOps
from rich.progress import Progress

from darkcyan.config import Config
from darkcyan.constants import DEFAULT_DET_SRC_NAME, DEFAULT_DET_TRAINING_YAML, DataType
from .training_utils import (
    create_training_zipfile,
    get_training_data_src_directory,
)
from .local_data_utils import get_local_scratch_directory_for_version

term = Terminal()


def prep_directories(directories):
    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)


def build_output_structure(image_src, label_src, output_directory):
    if output_directory.exists():
        print(term.red(f"Using existing output directory {output_directory}"))
        return
    images_dir = output_directory / "images"
    train_dir_img = images_dir / "train"
    test_dir_img = images_dir / "test"
    prep_directories([images_dir, train_dir_img, test_dir_img])

    labels_dir = output_directory / "labels"
    train_dir_lbl = labels_dir / "train"
    test_dir_lbl = labels_dir / "test"
    prep_directories([labels_dir, train_dir_lbl, test_dir_lbl])

    # For each image src, flatten and copy to output, along with the labels
    image_src_dirs = list(image_src.glob(f"[!.]*"))
    for image_src in image_src_dirs:
        if image_src.is_dir():
            image_list = list(image_src.glob(f"*.[jpg|jpeg|png]*"))
            num_images = len(image_list)

            num_test_images = math.ceil(Config.get_value("det_test_ratio") * num_images)
            with Progress() as progress:
                test_images_task = progress.add_task(
                    f"[darkcyan]Processing test images for {image_src.name}...",
                    total=num_test_images,
                )

                for i in range(num_test_images):
                    progress.advance(test_images_task)
                    # Pick a random image and copy
                    idx = random.randint(0, len(image_list) - 1)
                    shutil.copy(image_list[idx], test_dir_img)

                    ## grab the txt file and create (if missing) or copy
                    label_file = (
                        label_src / image_src.name / f"{image_list[idx].stem}.txt"
                    )
                    if label_file.exists():
                        shutil.copy(label_file, test_dir_lbl)
                    else:
                        label_file.touch()
                    image_list.remove(image_list[idx])
            with Progress() as progress:
                train_images_task = progress.add_task(
                    f"[darkcyan]Processing train images for {image_src.name}...",
                    total=num_test_images,
                )
                for filename in image_list:
                    progress.advance(train_images_task)
                    shutil.copy(filename, train_dir_img)
                    ## grab the txt file and create (if missing) or copy
                    label_file = label_src / image_src.name / f"{filename.stem}.txt"
                    if label_file.exists():
                        shutil.copy(label_file, train_dir_lbl)
                    else:
                        label_file.touch()


def create_config(classes, outputdir):
    with open(classes, "r") as f:
        classes_txt = f.read()
        classes_list = list(filter(None, classes_txt.split("\n")))

    yolo_config = {
        "path": "path to directory on colab machine",
        "train": f"/content/det_training_data/images/train",
        "val": f"/content/det_training_data/images/test",
        "nc": len(classes_list),
        "names": classes_list,
    }

    with open(outputdir / DEFAULT_DET_TRAINING_YAML, "w") as f:
        documents = yaml.dump(yolo_config, f, default_flow_style=None)


def create_training_images(src_data_dir, temp_working_dir, target_width):
    """The source directory contains a subdirectory for each image source, iterate through each and create the
    resized training images in the temp_working_dir
    """

    if temp_working_dir.exists():
        print(term.red(f"Temp working directory {temp_working_dir} already exists, "))
        return
    temp_working_dir.mkdir(parents=True)

    src_dirs = list(src_data_dir.glob(f"[!.]*"))
    for src_dir in src_dirs:
        if src_dir.is_dir():
            image_list = list(src_dir.glob(f"*.[jpg|jpeg|png]*"))

            (temp_working_dir / src_dir.name).mkdir(parents=True, exist_ok=True)
            with Progress() as progress:
                resize_task = progress.add_task(
                    f"[darkcyan]Processing {src_dir.name}...", total=len(image_list)
                )
                for img_file in image_list:
                    progress.advance(resize_task)
                    img = Image.open(img_file)
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")
                    w, h = img.size
                    ratio = w / target_width
                    target_height = int(h / ratio)
                    img = img.resize((target_width, target_height), PIL.Image.LANCZOS)
                    img.save(temp_working_dir / src_dir.name / img_file.name)


def create_yolo_detection_dataset(src_version, training_version, resize_width=980):
    # num_test_images = math.ceil(Config.get_value("det_test_ratio") * num_images)
    src_data_dir = (
        get_local_scratch_directory_for_version(src_version, DataType.det)
        / DEFAULT_DET_SRC_NAME
    )

    if not src_data_dir.exists():
        print(term.red(f"No data found for {src_version}"))
        return

    # Prepare image data
    temp_working_dir = temp_dir = (
        Path(Config.get_value("temp_dir"))
        / f"{Path(Config.get_value('data_prefix'))}_v{training_version}_det_{resize_width}"
        / DEFAULT_DET_SRC_NAME
    )
    if not temp_working_dir.exists():
        create_training_images(src_data_dir, temp_working_dir, resize_width)
    else:
        print(
            term.red(f"Using existing generated training data in: {temp_working_dir}")
        )

    # Prepare training data
    output_directory = get_training_data_src_directory(training_version, DataType.det)

    build_output_structure(temp_working_dir, src_data_dir, output_directory)
    create_config(src_data_dir / "classes.txt", output_directory)


def main():
    training_data_version = "4.1.1"
    create_yolo_detection_dataset("4.1", training_data_version, 980)
    create_training_zipfile(training_data_version, DataType.det)


if __name__ == "__main__":
    main()
