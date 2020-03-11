import os
import gzip
import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


# transforms to apply to the data
trans = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28), interpolation=5),
        transforms.ToTensor()])


def load_mnist(dataset):
    if dataset == 'MNIST' or dataset == 'CIFAR' or dataset == 'FashionMNIST':
        # Load data
        train_path = '../Few_Shot_Dataset/' + dataset + '/train'
        test_path = '../Few_Shot_Dataset/' + dataset + '/test'
        print('Train Path (%s )' %(train_path))
        print('Test Path (%s )' %(test_path))
    else:
        # Load data
        train_path = '../Augmented_Dataset/train'
        test_path = '../Augmented_Dataset/test'
        print('Train Path (%s )' %(train_path))
        print('Test Path (%s )' %(test_path))

    train_loader_all = datasets.ImageFolder(
        root=train_path, transform=trans)
    test_loader = datasets.ImageFolder(
        root=test_path, transform=trans)

    train_data = train_loader_all[0][0]
    train_target = []
    for i, (image, label) in enumerate(train_loader_all):
        train_data = torch.cat((train_data, image), 0)
        train_target.append(label)

    train_target = torch.tensor(train_target)

    test_data = test_loader[0][0]
    test_target = []
    for i, (image, label) in enumerate(test_loader):
        test_data = torch.cat((test_data, image), 0)
        test_target.append(label)

    test_target = torch.tensor(test_target)

    # train_data = train_loader_all.dataset.train_data
    train_data.unsqueeze_(1)

    # train_target = train_loader_all.dataset.train_labels
    train_target.unsqueeze_(1)

    # test_data = test_loader.dataset.test_data
    test_data.unsqueeze_(1)

    # test_target = test_loader.dataset.test_labels
    test_target.unsqueeze_(1)

    X_train = np.asarray(train_data).astype("float32")
    y_train = np.asarray(train_target).astype(np.int)

    X_test = np.asarray(test_data).astype("float32")
    y_test = np.asarray(test_target).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    y_train_vec = np.zeros((len(y_train), 10), dtype=np.float)
    for i, label in enumerate(y_train):
        y_train_vec[i, y_train[i]] = 1

    y_test_vec = np.zeros((len(y_test), 10), dtype=np.float)
    for i, label in enumerate(y_test):
        y_test_vec[i, y_test[i]] = 1

    X_train = X_train / 255.
    X_test = X_test / 255.

    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    y_train_vec = torch.from_numpy(y_train_vec).type(torch.FloatTensor)
    y_test_vec = torch.from_numpy(y_test_vec).type(torch.FloatTensor)

    return X_train, y_train_vec, X_test, y_test_vec


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    imageio.imwrite(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError(
            'in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)


def loss_plot(hist, path='Train_hist.png', model_name=''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']
    y3 = hist['C_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.plot(x, y3, label='C_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
