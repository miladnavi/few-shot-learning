# python main.py --dataset mnist --gan_type ACGAN --epoch 50 --batch_size 100
import argparse, os


import os 
import tarfile
import glob
import shutil
import random
import torch
import torchvision.datasets
import os



def few_shot_dataset_mnist(number_of_sample): 
    source_path_unzip = './Dataset/MNIST.tar.gz'
    destination_path = './Few_Shot_Dataset'
    real_dir_name = '/mnist_png'
    custom_dir_name = '/MNIST'
    source_path = './Few_Shot_Dataset/MNIST'

    #Unpaking data-set zip file 
    tf = tarfile.open(source_path_unzip)
    tf.extractall(destination_path)
    os.rename(destination_path + real_dir_name, destination_path + custom_dir_name)

    #Few Shot Dataset
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
        for i in range(number_of_sample):
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


def few_shot_dataset_fashion_mnist(number_of_sample): 

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./Dataset', train=True, download=True)

    test_dataset = torchvision.datasets.FashionMNIST(
        root='./Dataset', train=False, download=True)
    try:
        os.mkdir('./Few_Shot_Dataset')
    except:
        print("Dir exists")
    try:
        os.mkdir('./Few_Shot_Dataset/FashionMNIST')
    except:
        print("Dir exists")
    try:
        os.mkdir('./Few_Shot_Dataset/FashionMNIST/train')
    except:
        print("Dir exists")
    try:
        os.mkdir('./Few_Shot_Dataset/FashionMNIST/test')
    except:
        print("Dir exists")

    classes = train_dataset.classes
    classes_dic = {}
    for i, (el) in enumerate(classes):
        classes_dic.update({el: 0})
        try:
            os.mkdir('./Few_Shot_Dataset/FashionMNIST/train/' + str(i))
        except:
            print("Dir exists")
        try:
            os.mkdir('./Few_Shot_Dataset/FashionMNIST/test/' + str(i))
        except:
            print("Dir exists")

    for i, (image, label) in enumerate(train_dataset):
        if set(classes_dic.values()) == set([number_of_sample]):
            break
        else:
            if classes_dic[classes[label]] < number_of_sample:
                classes_dic[classes[label]] += 1
                image.save('./Few_Shot_Dataset/FashionMNIST/train/' + str(label) + '/' + classes[label].replace('/', '-') + '-' + str(i) +'.png')
  
    for i, (image, label) in enumerate(test_dataset):
        image.save('./Few_Shot_Dataset/FashionMNIST/test/' + str(label) + '/' + classes[label].replace('/', '-')+ '-' + str(i) +'.png')
        
                
def few_shot_dataset_cifar(number_of_sample):
    train_dataset = torchvision.datasets.CIFAR10(
        root='./Dataset', train=True, download=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./Dataset', train=False, download=True)
    try:
        os.mkdir('./Few_Shot_Dataset')
    except:
        print("Dir exists")
    try:
        os.mkdir('./Few_Shot_Dataset/CIFAR')
    except:
        print("Dir exists")
    try:
        os.mkdir('./Few_Shot_Dataset/CIFAR/train')
    except:
        print("Dir exists")
    try:
        os.mkdir('./Few_Shot_Dataset/CIFAR/test')
    except:
        print("Dir exists")

    classes = train_dataset.classes
    classes_dic = {}
    for i, (el) in enumerate(classes):
        classes_dic.update({el: 0})
        try:
            os.mkdir('./Few_Shot_Dataset/CIFAR/train/' + str(i))
        except:
            print("Dir exists")
        try:
            os.mkdir('./Few_Shot_Dataset/CIFAR/test/' + str(i))
        except:
            print("Dir exists")

    for i, (image, label) in enumerate(train_dataset):
        if set(classes_dic.values()) == set([number_of_sample]):
            break
        else:
            if classes_dic[classes[label]] < number_of_sample:
                classes_dic[classes[label]] += 1
                image.save('./Few_Shot_Dataset/CIFAR/train/' + str(label) + '/' + classes[label] + '-' + str(i) +'.png')
  
    for i, (image, label) in enumerate(test_dataset):
        image.save('./Few_Shot_Dataset/CIFAR/test/' + str(label) + '/' + classes[label] + '-' + str(i) +'.png')


"""parsing and configuration"""
def parse_args():
    desc = "Generate Few Shot Dataset"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['mnist', 'fashion_mnist', 'cifar'],
                        help='Generate Few Shot for Dataset')
    parser.add_argument('--number_of_sample', type=int, default=10, choices=[1,5,10,20,30] ,help='The number of the samples of each class')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --number_of_sample
    try:
        assert args.epoch >= 1
    except:
        print('number of smaples must be larger than or equal to one')

    # --batch_size
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

        # declare instance for GAN
    if args.dataset == 'MNIST':
        few_shot_dataset_mnist(args.number_of_sample)
    elif args.dataset == 'Fashion_MNIST':
        few_shot_dataset_fashion_mnist(args.number_of_sample)
    elif args.dataset == 'CIFAR':
        few_shot_dataset_cifar(args.number_of_sample)
    else:
        raise Exception("[!] There is no option for " + args.dataset)

if __name__ == '__main__':
    main()
