# %%
# Data Cleaner

# This script approach to provide clean datasets for few-shot-learning
import torch
import torchvision.datasets
import os


def few_shot_dataset(number_of_sample):
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
        
                
