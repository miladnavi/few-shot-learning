# %%
import glob
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import cnn
import Augmentor


#%%
# Uupack the dataset zip
from data_cleaner import unpack_zip_file, few_shot_dataset
unpack_zip_file('./Dataset/MNIST.tar.gz', './Few_Shot_Dataset', '/mnist_png', '/MNIST')

# Create few-shot dataset
few_shot_dataset('./Few_Shot_Dataset/MNIST', 10)

#%%
from data_augmentor import label_preserving_trasnformation

classes_dir = ['/0', '/1', '/2', '/3', '/4', '/5', '/6', '/7', '/8', '/9']

source_path = './Few_Shot_Dataset/MNIST'
destination_path = './Augmented_Dataset/train'
output_dir = '/output/'
dataset_kind = '/train'
sample_number = 100

def label_preserving_trasnformation(source_path, destination_path, classes_dir, output_dir, dataset_kind, sample_number):
    source_path = source_path + dataset_kind
    for class_dir in classes_dir:
        p = Augmentor.Pipeline(source_path + class_dir)
        p.crop_random(probability=1, percentage_area=0.8)
        p.resize(probability=1.0, width=28, height=28)
        p.sample(sample_number)
        p.flip_left_right(probability=1.0)
        p.sample(sample_number)

    for class_dir in classes_dir:
        source_dir = source_path + class_dir + output_dir
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

label_preserving_trasnformation('./Few_Shot_Dataset/MNIST', './Augmented_Dataset', classes_dir, '/output/', '/train', 500)

label_preserving_trasnformation('./Few_Shot_Dataset/MNIST', './Augmented_Dataset', classes_dir, '/output/', '/test', 50)

source_path = './Few_Shot_Datasets'
destination_path = './Augmented_Dataset'
classes_dir = {'/cats', '/dogs'}
output_dir = '/output/'
# %%
# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 10
learning_rate = 0.001

DATA_PATH = 'Data'
MODEL_STORE_PATH = 'Model'

# %%
# transforms to apply to the data
trans = transforms.Compose(
    [   transforms.Grayscale(num_output_channels= 1),
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.ImageFolder(
    root='./Augmented_Dataset/train', transform=trans)

test_dataset = torchvision.datasets.ImageFolder(
    root='./Dataset/testing', transform=trans)

# %%
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=10, shuffle=False)

# %%
model = cnn.ConvNet()


# %%
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# %%
# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                      (correct / total) * 100))

# %%
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    correct1 = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        mean = torch.mean(outputs.data, 1)
        transpose = torch.transpose(outputs.data, 0, 1)
        sum_of_tensor = torch.sum(transpose, 1)
        label_of_prediction = torch.argmax(sum_of_tensor, 0).item()
        if label_of_prediction == labels.unique().data[0]:
            correct1 += 1
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        (correct / total) * 100))
    print(correct1/10000)
# %%
# Save the plot
p = figure(width=850, y_range=(0, 1))
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list)
       * 100, y_range_name='Accuracy', color='red')
show(p)

# %%

# Get a list of all the file paths that ends with .txt from in specified directory


fileList = glob.glob(dest_dir + '/*', recursive=True)
for filePath in fileList:
    shutil.rmtree(filePath)


# %%
a = torch.randn(2, 3)
print(a)
a = torch.transpose(a, 0,1)
print(a)
b = torch.sum(a ,1)
print(b)
# %%
