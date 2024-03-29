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
from bokeh.io import show, output_notebook
from bokeh.models import LinearAxis, Range1d
import numpy as np
import Augmentor
import cifar_cnn
import matplotlib.pyplot as plt


# %%
# Hyperparameters
num_epochs = 20
num_classes = 10
train_batch_size = 100
test_batch_size = 10
learning_rate = 0.001
classes=('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# Training onGPU when it is available otherwise CPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
from data_cleaner import few_shot_dataset
few_shot_sample_number = 10

# Create few-shot dataset
#few_shot_dataset(few_shot_sample_number)

#%%
classes_dir = ['/0', '/1', '/2', '/3', '/4', '/5', '/6', '/7', '/8', '/9']

few_shot_source_path = './Few_Shot_Dataset/CIFAR'
augmented_destination_path = './Augmented_Dataset'
output_dir = '/output/'
dataset_kind_train = '/train'
dataset_kind_test = '/test'
augment_sample_train_number = 50
augment_sample_test_number = 5000

def image_translation(source_path, destination_path, classes_dir, output_dir, dataset_kind, sample_number):
    source_path = source_path + dataset_kind
    for class_dir in classes_dir:
        p = Augmentor.Pipeline(source_path + class_dir)
        p.crop_random(probability=1, percentage_area=0.875)
        p.resize(probability=1.0, width=32, height=32)
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

# Clean Augmented Dataset
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

# Training Dataset
image_translation(
    few_shot_source_path, augmented_destination_path, classes_dir, output_dir, dataset_kind_train, augment_sample_train_number)

# Testting Dataset
image_translation(
    few_shot_source_path, augmented_destination_path, classes_dir, output_dir, dataset_kind_test, augment_sample_test_number)


# %%
# transforms to apply to the data
trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# MNIST dataset
train_dataset = torchvision.datasets.ImageFolder(
    root='./Augmented_Dataset/train', transform=trans)

test_dataset = torchvision.datasets.ImageFolder(
    root='./Augmented_Dataset/test', transform=trans)

# %%
# Data size
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print('Tarining dataset size: {}' .format(train_dataset_size))
print('Testing dataset size: {}' .format(test_dataset_size))

# %%
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=test_batch_size, shuffle=False)


# %%
model = cifar_cnn.ConvNet().to(device)


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
        images = images.to(device)
        labels = labels.to(device)
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
    confusion_matrix = np.zeros([10,10], int)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #x_unique = predicted.unique(sorted=True)
        #x_unique_count = torch.stack([(predicted==x_u).sum() for x_u in x_unique])
        transpose = torch.transpose(outputs.data, 0, 1)
        sum_of_tensor = torch.sum(transpose, 1)
        label_of_prediction = torch.argmax(sum_of_tensor, 0).item()
        confusion_matrix[labels.unique().data[0], label_of_prediction] += 1 
        if label_of_prediction == labels.unique().data[0]:
            correct1 += 1
        correct += (predicted == labels).sum().item()
    #print('Test Accuracy of the model without avraging on softmax layer on the {} test images: {} %'.format( test_dataset_size, (correct / total) * 100))    
    print('Test Accuracy of the model on the {} test images: {:.4f} %'.format(test_dataset_size, (correct1/test_dataset_size) * 1000))
    
# %%
# Save the plot   
if os.path.isdir('./Accuracy_Heatmap') is False:
    os.mkdir('./Accuracy_Heatmap')
if os.path.isdir('./Accuracy_Heatmap/CIFAR') is False:
    os.mkdir('./Accuracy_Heatmap/CIFAR')
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
for (i, j), z in np.ndenumerate(confusion_matrix):
    ax.text(j, i, format((z/1000), '.2%'), ha='center', va='center')
# plt.ylabel('Actual Lable')
plt.yticks(range(10), classes)
plt.xlabel('Predicted Lable')
plt.xticks(range(10), classes)
plt.savefig('./Accuracy_Heatmap/CIFAR/cifar_image_translation.png')

# %%
# Save the plot
p = figure(width=850, y_range=(0, 1))
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list)
       * 100, y_range_name='Accuracy', color='red')
show(p)