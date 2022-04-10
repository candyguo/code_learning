import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import datetime
from sklearn import preprocessing
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pickle

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

train_dataset = datasets.MNIST('./mnist_data', train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./mnist_data', train=False, transform=transforms.ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# class MnistModel(nn.Module):
#     def __init__(self):
#         super(MnistModel, self).__init__()
#         self.linear1 = nn.Linear(784, 128)
#         self.linear2 = nn.Linear(128, 256)
#         self.linear3 = nn.Linear(256, 10)

#     def forward(self, x):
#         out = F.relu(self.linear1(x))
#         out = F.relu(self.linear2(out))
#         out = self.linear3(out)
#         return out

# train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
# valid_ds = TensorDataset(x_valid, y_valid)
# valid_dl = DataLoader(valid_ds, batch_size=16)

# model = MnistModel()

class MnistCnnModel(nn.Module):
    def __init__(self):
        super(MnistCnnModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), # -> 28 * 28 * 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # -> 14 * 14 * 16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2), # -> 14 * 14 * 32
            nn.ReLU(),
            nn.MaxPool2d(2) # -> 7 * 7 * 32
        )
        self.linear1 = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # flatten image to (batch size, 32 * 7 * 7)
        return self.linear1(x)    

model = MnistCnnModel()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = []
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            loss = F.cross_entropy(model(x), y)
            test_loss.append(loss.cpu().item())
        print("current epoch: {}, loss: {}".format(epoch, np.mean(test_loss)))

# use test data to visualize test
for test_data_x, test_data_y in test_dataset:
    test_data_x_sample = test_data_x.unsqueeze(0)
    test_data_x_gpu = test_data_x_sample.to(device)
    predict_y = torch.argmax(model(test_data_x_gpu))

    plt.imshow(test_data_x.squeeze(0).numpy())
    plt.title("label: {}, predict: {}".format(test_data_y, predict_y.cpu()))
    plt.show()
