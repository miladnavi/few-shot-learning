# %%
from data_cleaner import few_shot_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import fashion_mnist_cnn
import matplotlib.pyplot as plt
import numpy as np
import os


# %%
# Hyperparameters
num_epochs = 10
num_classes = 10
train_batch_size = 5
test_batch_size = 10
learning_rate = 0.001
classes = ('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
           
DATA_PATH = 'Data'
MODEL_STORE_PATH = 'Model'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
few_shot_sample_number = 1
# Create few-shot dataset
#few_shot_dataset(few_shot_sample_number)

# %%
# transforms to apply to the data
trans = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

# FASHIONMNIST dataset
train_dataset = torchvision.datasets.ImageFolder(
    root='./Few_Shot_Dataset/FashionMNIST/train', transform=trans)

test_dataset = torchvision.datasets.ImageFolder(
    root='./Few_Shot_Dataset/FashionMNIST/test', transform=trans)



# %%
# Data size
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print('Tarining dataset size: {}' .format(train_dataset_size))
print('Testing dataset size: {}' .format(test_dataset_size))

# %%
# Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=test_batch_size, shuffle=False)

# %%
# Load CNN
model = fashion_mnist_cnn.ConvNet().to(device)


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
    confusion_matrix = np.zeros([10,10], int)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i, l in enumerate(labels):
            confusion_matrix[l.item(), predicted[i].item()] += 1 
    print('Test Accuracy of the model on the {} test images: {} %'.format(test_dataset_size,
                                                                          (correct / total) * 100))


# %%
# Save the plot
if os.path.isdir('./Accuracy_Heatmap') is False:
    os.mkdir('./Accuracy_Heatmap')
if os.path.isdir('./Accuracy_Heatmap/FashionMNIST') is False:
    os.mkdir('./Accuracy_Heatmap/FashionMNIST')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
for (i, j), z in np.ndenumerate(confusion_matrix):
    ax.text(j, i, format((z/1000), '.2%'), ha='center', va='center')
plt.ylabel('Actual Lable')
plt.yticks(range(10), classes)
plt.xlabel('Predicted Lable')
plt.xticks(range(10), classes)
plt.savefig('./Accuracy_Heatmap/FashionMNIST/fashion_mnist_without_augmentation.png')

# %%
# Save the model
#torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

# %%
# Save the plot
p = figure(width=850, y_range=(0, 1))
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list)
       * 100, y_range_name='Accuracy', color='red')
show(p)
