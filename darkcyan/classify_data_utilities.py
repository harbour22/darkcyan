import math
import random
import shutil


from blessed import Terminal
from rich.progress import Progress

from constants import DataType
from constants import DataTag
from constants import DEFAULT_DET_SRC_NAME
from constants import DEFAULT_CLS_SRC_NAME
from constants import DEFAULT_CLASSES_TXT
from local_data_utility import init_directories
from local_data_utility import get_available_data_versions
from local_data_utility import prepare_working_directory
from local_data_utility import remove_scratch_version
from local_data_utility import create_main_from_scratch
from local_data_utility import get_local_zipfile_for_version
from local_data_utility import get_local_scratch_directory_for_version

from config import Config

from pathlib import Path

import PIL
from PIL import Image, ImageOps

from rich.progress import Progress

term = Terminal()

def generate_letterbox_images(version):
    init_directories()
    temp_dir = Path(Config.get_value('temp_dir'))
    data_directory = get_local_scratch_directory_for_version(version, DataType.cls)
    if(not data_directory.exists()):
        print(term.red(f'No data found for {version}'))
        return
    letterbox_dir = temp_dir / 'images_cls_letterbox_src' / DEFAULT_CLS_SRC_NAME
    if(letterbox_dir.exists()):
        print(term.red(f'Letterbox images already exist for {version}'))
        return letterbox_dir
    
    print(f'Generating letterbox images from {letterbox_dir} for {version} from {data_directory}')

    if(not letterbox_dir.exists()):
        letterbox_dir.mkdir(parents=True)  

    classifiers_dirs = list(data_directory.glob(f'*/[!.]*'))

    with Progress() as progress:

        for img_class_dir in classifiers_dirs:
            image_list = list(img_class_dir.glob(f'*.[jpg|jpeg|png]*'))

            target_dir = letterbox_dir / img_class_dir.name

            if(not target_dir.exists()):
                target_dir.mkdir(parents=True)                            

            letterbox_task = progress.add_task(f"[darkcyan]Processing {img_class_dir.name}...", total=len(image_list))

            for image_path in image_list:
                progress.advance(letterbox_task)
                
                img = Image.open(image_path)
                w, h = img.size
                if(w<h):
                    img = img.resize([int(w*(224/h)), 224], PIL.Image.Resampling.LANCZOS)    
                    w, h = img.size
                    img = ImageOps.expand(img,border=(int((224-w)/2), 0),fill='black')
                else:
                    img = img.resize([224, int(h*(224/w))], PIL.Image.Resampling.LANCZOS)    
                    w, h = img.size
                    img = ImageOps.expand(img,border=(0, int((224-h)/2)),fill='black')

                img = img.resize([224, 224], PIL.Image.Resampling.LANCZOS)
                img.save(target_dir / image_path.name)
            progress.update(letterbox_task, visible=False)
    return letterbox_dir
    

def create_yolo_classification_dataset(version, letterbox=False):
    init_directories()
    if(letterbox):        
        data_directory = generate_letterbox_images(version)
    else:
        data_directory = get_local_scratch_directory_for_version(version, DataType.cls)
    if(not data_directory.exists()):
        print(term.red(f'No data found for {version}'))
        return
    temp_dir = Path(Config.get_value('temp_dir'))

    output_directory = temp_dir / f"{Config.get_value('data_suffix')}_v{version}_classify"

    print(f'Creating YOLO classification dataset from {data_directory} using {temp_dir} in {output_directory}')

    train_dir_img = output_directory / 'train'
    test_dir_img = output_directory / 'test'
    if(not train_dir_img.exists()):
        train_dir_img.mkdir(parents=True)
    if(not test_dir_img.exists()):
        test_dir_img.mkdir(parents=True)

    classifiers_dirs = list(data_directory.glob(f'*/[!.]*'))

    with Progress() as progress:

        for img_class_dir in classifiers_dirs:
            image_list = list(img_class_dir.glob(f'*.[jpg|jpeg|png]*'))

            test_image_dir = test_dir_img / img_class_dir.name
            train_image_dir = train_dir_img / img_class_dir.name

            num_images = len(image_list)
            num_test_images = math.ceil(Config.get_value('test_ratio')*num_images)

            if(num_test_images<=2):
                print(f'Insufficient test images for {img_class_dir.name}, skipping')
                continue

            if not test_image_dir.exists():
                test_image_dir.mkdir(parents=True)

            test_task = progress.add_task(f"[darkcyan]Processing {img_class_dir.name} test images...", total=num_test_images)

            for i in range(num_test_images):
                progress.advance(test_task)                
                idx = random.randint(0, len(image_list)-1)
                filename = image_list[idx]
                shutil.copy(filename, test_image_dir / filename.name)
                image_list.remove(image_list[idx])
            progress.update(test_task, visible=False)
            

            if not train_image_dir.exists():
                train_image_dir.mkdir(parents=True)
                
            train_task = progress.add_task(f"[darkcyan]Processing {img_class_dir.name} train images...", total=len(image_list))

            for filename in image_list:
                progress.advance(train_task)                
                shutil.copy(filename, train_image_dir / filename.name)
            progress.update(train_task, visible=False)

def create_classification_zipfile(version):
    init_directories()
    temp_dir = Path(Config.get_value('temp_dir'))

    output_directory = temp_dir / f"{Config.get_value('data_suffix')}_v{version}_classify"
    if(not output_directory.exists()):
        print(term.red(f'No data found for {version}'))
        return
    
    with Progress(transient=True) as progress:
        task1 = progress.add_task(f"[blue]Creating zip file for {version} from {output_directory}", total=None)
        zip_filename = temp_dir / f"{Config.get_value('data_suffix')}_v{version}_classify"
        zipfile = shutil.make_archive(zip_filename, 'zip', root_dir=output_directory, base_dir='.')
        progress.update(task1, completed=1)
        return zipfile
   

def main():
    create_yolo_classification_dataset('4.1', False)
    #create_classification_zipfile('4.1')


if(__name__ == '__main__'):
    main()