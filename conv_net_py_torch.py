# %%
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
batch_size = 5
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


# %%
each_instances_numnber = np.zeros((10,), dtype=int)
labels = np.arange(10)

lables_instances_number_map = np.array([labels,each_instances_numnber]).transpose()
lables_instances_number_map

lables_instances_number_map[:, 1]

# %%
my_x = [np.array(train_dataset[0][0].numpy())]
my_y = [np.array([train_dataset[0][1]])]
for i, (el) in enumerate(train_dataset):
    if lables_instances_number_map[train_dataset[i][1]][1] < 10:
        my_x = np.append(my_x, [np.array(train_dataset[i][0].numpy())], axis=0)
        my_y = np.append(my_y, [np.array([train_dataset[i][1]])], axis=0)
        lables_instances_number_map[train_dataset[i][1]
                     ][1] = lables_instances_number_map[train_dataset[i][1]][1] + 1
    if len(np.unique(lables_instances_number_map[:, 1])) == 1:
        if np.unique(lables_instances_number_map[:, 1]) == [10]:
            break

tensor_x = torch.stack([torch.Tensor(i) for i in my_x])

# transform to torch tensors
tensor_y = torch.stack([torch.Tensor(i) for i in my_y]).long()

train_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

# %%
# Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)
train_loader

# %%
# Train Data Info
number_of_each_classes_instance = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
    '6': 0,
    '7': 0,
    '8': 0,
    '9': 0,
}
for i, (images, labels) in enumerate(train_loader):
    for i, (label) in enumerate(labels):
        number_of_each_classes_instance[str(
            label.item())] = number_of_each_classes_instance[str(label.item())] + 1


number_of_each_classes_instance
# %%
# Convolutional neural network (two convolutional layers)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# %%
model = ConvNet()


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
        loss = criterion(outputs, labels.squeeze(1))
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.squeeze(1)).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


# %%
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        (correct / total) * 100))

# %%
# Save the model
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
# %%
# Save the plot
p = figure(width=850, y_range=(0, 1))
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list)
       * 100, y_range_name='Accuracy', color='red')
show(p)



