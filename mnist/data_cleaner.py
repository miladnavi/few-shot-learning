## Data Cleaner

##### This script approach to provide clean datasets for few-shot-learning 

import os 
import tarfile
import glob
import shutil
import random

def unpack_zip_file(source_path, destination_path, real_dir_name, custom_dir_name):
    """Unpaking data-set zip file 

    Parameters
    ----------
    source_path: str
        The path of source where zip file is in it
    
    destination_path: str
        The path of file where zip file should be unpacked

    real_dir_name: str
        The name of unpacked dir of data-set
    
    custom_dir_name: str
        The custom name of data set dir
    """

    tf = tarfile.open(source_path)
    tf.extractall(destination_path)
    os.rename(destination_path + real_dir_name, destination_path + custom_dir_name)

def few_shot_dataset(source_path, number_of_instances): 
    training_path = source_path + '/training'
    testing_path = source_path + '/testing'

    list_of_classes_dir = glob.glob(training_path + '/*', recursive=True)
    list_of_classes_dir_test = glob.glob(testing_path + '/*', recursive=True)

    os.mkdir(source_path + '/train')
    os.mkdir(source_path + '/test')

    for class_dir in list_of_classes_dir:
        class_dir_name = class_dir.split('/')
        class_dir_name = class_dir_name[-1]
        list_of_instances = glob.glob(class_dir + '/*', recursive=True)
        os.mkdir(source_path + '/train/' + class_dir_name)
        for i in range(number_of_instances):
            file_path = random.choice(list_of_instances)
            file_name = file_path.split('/')
            file_name = file_name[-1]
            copy_file_train = 'cp ' + file_path + ' ' + source_path + '/train/' + class_dir_name + '/' + file_name
            os.system(copy_file_train)

    for class_dir in list_of_classes_dir_test:
        class_dir_name_test = class_dir.split('/')
        class_dir_name_test = class_dir_name_test[-1]
        list_of_instances_test = glob.glob(class_dir + '/*', recursive=True)
        try:
            os.mkdir(source_path + '/test/' + class_dir_name_test)
        except:
            print("Folder " + class_dir_name_test + " exist!")

        for element in list_of_instances_test:
            file_path = element
            file_name = file_path.split('/')
            file_name = file_name[-1]
            copy_file_test = 'cp ' + file_path + ' ' + source_path + '/test/' + class_dir_name_test + '/' + file_name
            os.system(copy_file_test)

    
    shutil.rmtree(training_path)
    shutil.rmtree(testing_path)
