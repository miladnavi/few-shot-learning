import argparse, os
import os 
import tarfile
import glob
import shutil
import random
import torch
import torchvision.datasets
import os
import Augmentor


def random_erasing_augmentor(source_path, destination_path, classes_dir, output_dir, dataset_kind, sample_number):
    source_path = source_path + dataset_kind
    if dataset_kind == '/train':
        for class_dir in classes_dir:
            p = Augmentor.Pipeline(source_path + class_dir)
            p.random_erasing(probability=1, rectangle_area=0.2)
            p.sample(sample_number)

    for class_dir in classes_dir:
        if dataset_kind == '/train':
            source_dir = source_path + class_dir + output_dir
        elif dataset_kind == '/test':
            source_dir = source_path + class_dir + '/'

        destination_dir = destination_path + dataset_kind + class_dir
        try:
            os.mkdir(destination_path + dataset_kind)
        except:
            print("Dir exists")
        try:
            os.mkdir(destination_dir)
        except:
            print("Dir exists")
        
        files = os.listdir(source_dir)
        for f in files:
            shutil.move(source_dir + f, destination_dir)

    os.rmdir(source_dir)

# Clean Augmented Dataset
def clean_augmented_dataset():
    try:
        shutil.rmtree('./Augmented_Dataset/train')
    except:
        print('No such file or director: ./Augmented_Dataset/train')
    try:
        shutil.rmtree('./Augmented_Dataset/test')
    except:
        print('No such file or director: ./Augmented_Dataset/test')

    if os.path.isdir('./Augmented_Dataset') is False:
        os.mkdir('./Augmented_Dataset')

"""parsing and configuration"""
def parse_args():
    desc = "Generate Few Shot Dataset"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist', 'cifar'],
                        help='Generate Few Shot for Dataset')
    parser.add_argument('--number_of_sample', type=int, default=100, help='The number of the samples of each class')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --number_of_sample
    try:
        assert args.number_of_sample >= 1
    except:
        print('number of samples must be larger than or equal to one')

    try:
        assert args.dataset in ['mnist', 'fashion_mnist', 'cifar']
    except:
        print('Dataset should be one of the following: [mnist, fashion_mnist, cifar]')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    few_shot_source_path = './Few_Shot_Dataset/'
    augmented_destination_path = './Augmented_Dataset'
    output_dir = '/output/'
    dataset_kind_train = '/train'
    dataset_kind_test = '/test'
    classes_dir = ['/0', '/1', '/2', '/3', '/4', '/5', '/6', '/7', '/8', '/9']

    if args.dataset == 'mnist':
        few_shot_source_path = few_shot_source_path + 'MNIST'
    elif args.dataset == 'fashion_mnist':
        few_shot_source_path = few_shot_source_path + 'FashionMNIST'
    elif args.dataset == 'cifar':
        few_shot_source_path = few_shot_source_path + 'CIFAR'
    else:
        raise Exception("[!] There is no option for " + args.dataset)

    clean_augmented_dataset()

    #Random Erasing Generator
    random_erasing_augmentor(few_shot_source_path, augmented_destination_path, classes_dir, output_dir, dataset_kind_train, args.number_of_sample)
    random_erasing_augmentor(few_shot_source_path, augmented_destination_path, classes_dir, output_dir, dataset_kind_test, args.number_of_sample)


if __name__ == '__main__':
    main()


few_shot_source_path = './Few_Shot_Dataset/CIFAR'
augmented_destination_path = './Augmented_Dataset'
output_dir = '/output/'
dataset_kind_train = '/train'
dataset_kind_test = '/test'
