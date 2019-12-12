# %%
# Data Cleaner

# This script approach to provide clean datasets for few-shot-learning
import torch
import torchvision.datasets
import os


def few_shot_dataset(number_of_sample):
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

    classes_dic = {}
    for i, (el) in enumerate(classes):
        classes_dic.update({el: 0})
  
    for i, (image, label) in enumerate(test_dataset):
        if set(classes_dic.values()) == set([10]):
            break
        else:
            if classes_dic[classes[label]] < 10:
                classes_dic[classes[label]] += 1
                image.save('./Few_Shot_Dataset/CIFAR/test/' + str(label) + '/' + classes[label] + '-' + str(i) +'.png')
