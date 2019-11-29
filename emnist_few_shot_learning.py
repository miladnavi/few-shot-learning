#%%
# Training MNIST with 10 instances for each calsses (few-shot-learning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np

# %%
# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

DATA_PATH = 'Data'
MODEL_STORE_PATH = 'Model'
# %%
# transforms to apply to the data
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=False, transform=trans)

for i in range(len(train_dataset)):
    print('size of image {} lable {}' .format(train_dataset[i][0].size(), train_dataset[i][1]))
    if i > 2: break

# %%
import matplotlib.pyplot as plt
for i in range(len(train_dataset)):
    torchimage = train_dataset[i][0]
    npimage = torchimage.permute(1,2,0)
    plt.imshow(npimage.squeeze())
    print('Label: {}' .format(train_dataset[i][1]))
    plt.show()
    if i > 10: break
# %%
classes_result = np.array([], dtype=int)
for i, (el)in enumerate(train_dataset):
    train_dataset_few_shot_learning = train_dataset[i]
    classes_result = np.append(classes_result, [train_dataset_few_shot_learning[1]])
classes_result = np.unique(classes_result)
classes_result
result  = np.array([classes_result, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
result_transpose = result.transpose()
# %%
print(type(train_dataset))
# %%
