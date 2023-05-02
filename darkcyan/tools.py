from blessed import Terminal
from rich.progress import Progress

from constants import DataType
from constants import DataTag
from constants import DEFAULT_DET_SRC_NAME
from constants import DEFAULT_CLASSES_TXT
from local_data_utility import display_available_data
from local_data_utility import get_available_data_versions
from local_data_utility import prepare_working_directory
from local_data_utility import remove_scratch_version
from local_data_utility import create_main_from_scratch
from local_data_utility import get_local_zipfile_for_version
from local_data_utility import get_local_scratch_directory_for_version

from config import Config

from pathlib import Path
import shutil
import difflib



command_options =  ['Display available data',
                    'Create local working copy of data',
                    'Launch labelImg',
                    'Build master from local working copy',
                    'Remove local working copy of data',
                    'Prepare data for training']

term = Terminal()

def ask_for_dataset_type():
    print()
    print(term.white(f'What type of dataset? (1: {DataType.det.name}, 2: {DataType.cls.name})'))
    
    inp=''
    with term.cbreak():
        inp = term.inkey()
    print(term.darkcyan(f'{inp}'))

    datatype = DataType.__call__(int(inp))
    return datatype

def ask_for_data_version(datatype, tag):

    versions = get_available_data_versions(datatype, tag)
    if(len(versions) == 0):
        print(term.red(f'No data found, any key to continue'))
        return None
    
    print(term.white(f'Choose the {tag.name} version: '))    
    for choice, version in enumerate(versions, start=1):
        print(term.white(f'{choice}: {version}'))
    
    with term.cbreak():
        version = versions[int(term.inkey())-1]
        print(term.darkcyan(f'{version}'))
        return version

def run_labelimg():

    data_version = ask_for_data_version(DataType.det, DataTag.scratch)

    data_directory = get_local_scratch_directory_for_version(data_version, DataType.det)

    parent_directory = data_directory / DEFAULT_DET_SRC_NAME
    
    image_dirs = []
    for file in Path(parent_directory).iterdir():
        if file.is_dir():
            image_dirs.append(file)

    if(len(image_dirs) == 0):
        print(term.red(f'No images found'))
        return
    
    print(term.white(f'Choose the image directory: '))
    for choice, image_dir in enumerate(image_dirs, start=1):
        print(term.white(f'{choice}: {image_dir.name}'))
    inp = ''
    with term.cbreak():
        inp = term.inkey()
        print(term.darkcyan(f'{inp}'))

    image_dir = image_dirs[int(inp)-1]
    print(image_dir)

    if not (image_dir / DEFAULT_CLASSES_TXT).exists():
        print(term.red(f'No classes.txt found in {image_dir}, copying from {parent_directory}'))
        shutil.copy(parent_directory / DEFAULT_CLASSES_TXT, 
                    image_dir / DEFAULT_CLASSES_TXT)

    import subprocess
    import sys

    labelImg_cmd = Config.get_value('labelImg_cmd')
    labelImg_args = f'{image_dir.as_posix()} {(parent_directory / DEFAULT_CLASSES_TXT).as_posix()} {image_dir.as_posix()}'
    with Progress(transient=True) as progress:
        task1 = progress.add_task(f"[blue]Running imageLbl for version {data_version}", total=None)
        
        pipe = subprocess.Popen(f'{labelImg_cmd} {labelImg_args}', 
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
        output = pipe.communicate()[0].decode('utf-8')
        output_stderr = pipe.communicate()[1].decode('utf-8')
        progress.update(task1, completed=1) 

    print()
    with open(parent_directory / DEFAULT_CLASSES_TXT, 'r') as parent_classes:
        with open(image_dir / DEFAULT_CLASSES_TXT, 'r') as child_classes:
            diff = difflib.unified_diff(
                parent_classes.readlines(),
                child_classes.readlines(),
                fromfile='Parent Classes.txt',
                tofile='Edited Classes.txt',
            )
            edits = False
            for line in diff:
                sys.stdout.write(line)
                edits = True
            if(edits):
                print(term.red(f'Changes detected, do you want to copy to parent? (y/n)'))
                with term.cbreak():
                    key = term.inkey()
                    print(term.darkcyan(f'{key}'))

                    if(key == 'y'):
                        shutil.copy(image_dir / DEFAULT_CLASSES_TXT, 
                                    parent_directory / DEFAULT_CLASSES_TXT)
                        print(term.white(f'Copied {image_dir / DEFAULT_CLASSES_TXT} to {parent_directory / DEFAULT_CLASSES_TXT}'))

def author_new_dataset():
    
    datatype = ask_for_dataset_type()

    versions = get_available_data_versions(datatype)
    if(len(versions)==0):
        print(term.red(f'No data found, any key to continue'))
        with term.cbreak():
            term.inkey()
            return    

    base_version = ask_for_data_version(datatype, DataTag.main)
    if(base_version is None):
        return

    major_version = int(base_version.split('.')[0])
    minor_version = int(base_version.split('.')[1])
    
    print(term.white(f'Choose major or minor version: '))
    print(term.white(f'1: Major Version Increment ({base_version} to {major_version+1}.0)'))
    print(term.white(f'2: Minor Version Increment ({base_version} to {major_version}.{minor_version+1})'))
    with term.cbreak():
        choice = term.inkey()
    if(choice == '1'):
        working_version = f'{major_version+1}.0'
    elif(choice == '2'):
        working_version = f'{major_version}.{minor_version+1}'
    else:
        print(term.red(f'Illogical choice {choice}, any key to continue'))
        with term.cbreak():
            term.inkey()
        return

    if(working_version in versions):
        print(term.red(f'Version v{working_version} already exists, any key to continue'))
        with term.cbreak():
            term.inkey()
        return
    
    scratch_versions = get_available_data_versions(datatype, DataTag.scratch)

    if(working_version in scratch_versions):
        print(term.red(f'Version v{working_version} already exists in scratch - overwrite? (y/N)'))
        with term.cbreak():
            overwrite = term.inkey()
            if(overwrite != 'y'):
                return
            remove_scratch_version(working_version, datatype)            


    print()
    print(term.darkcyan(f'Creating new dataset v{working_version}'))
    print()
    """ Create the local working directory """
    print(term.black_on_darkcyan(('Creating the local working copy')))
    print()
    prepare_working_directory(base_version, working_version, datatype)
    print(term.black_on_darkcyan(('Done.')))
    print()
    display_available_data()
    print()


def create_main_dataset_from_scratch():
    datatype = ask_for_dataset_type()
    scratch_version = ask_for_data_version(datatype, DataTag.scratch)
    if(scratch_version is None):
        return

    zipfile = get_local_zipfile_for_version(scratch_version, datatype)
    if(zipfile.exists()):
        print(term.red(f'Found existing {zipfile} - overwrite? (y/N)'))
        with term.cbreak():
            overwrite = term.inkey()
            if(overwrite != 'y'):
                return
            print(term.red(f'Overwriting {zipfile}'))

    create_main_from_scratch(scratch_version, datatype)
    print() 
    display_available_data()
    print()

def remove_working_copy_of_data():
    datatype = ask_for_dataset_type()
    scratch_version = ask_for_data_version(datatype, DataTag.scratch)
    if(scratch_version is None):
        return

    source_data_name = get_local_scratch_directory_for_version(scratch_version, datatype)
    if(source_data_name.exists()):
        remove_scratch_version(scratch_version, datatype)
        print(term.white(f'Removed {source_data_name}'))
    else:
        print(term.red(f'No existing {source_data_name} found.'))        
    
    print()
    display_available_data()
    print()     

if(__name__== '__main__'):
    print(term.clear())    
    print(term.black_on_darkcyan(('DarkCyan Tools')))

    while True:
        
        print()
        for choice, description in enumerate(command_options, start=1):
            print(term.white(f'{choice}: {description}'))
        print(term.white(f'q: Quit'))
        
        print()
        inp=''
        with term.cbreak():
            inp = term.inkey()
        match inp:
            case '1':
                print()
                display_available_data()
                print()

            case '2':
                print(term.white(f'Running '+term.blue(f'{command_options[int(inp)-1]}')))
                author_new_dataset()
            case '3':
                print(term.white(f'Running '+term.blue(f'{command_options[int(inp)-1]}')))
                run_labelimg()
            case '4':
                print(term.white(f'Running '+term.blue(f'{command_options[int(inp)-1]}')))
                create_main_dataset_from_scratch()
            case '5':
                print(term.white(f'Running '+term.blue(f'{command_options[int(inp)-1]}')))
                remove_working_copy_of_data()

            case 'q':
                print(term.darkcyan('Goodbye'))

                break
            case _:
                print(term.red('Invalid option'))
            
