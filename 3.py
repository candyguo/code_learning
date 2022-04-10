import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import datetime
from sklearn import preprocessing
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pickle

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

((x_train, y_train), (x_valid, y_valid), _) = pickle.load(open('mnist.pkl', "rb"), encoding="latin-1")

# plt.imshow(x_train[0].reshape(28, 28), cmap="gray")
# plt.show()

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 10)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=16)

model = MnistModel()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(25):
    model.train()
    for x, y in train_dl:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_loss = []
        for x, y in valid_dl:
            x = x.to(device)
            y = y.to(device)
            loss = F.cross_entropy(model(x), y)
            valid_loss.append(loss.cpu().item())
        print("current epoch: {}, loss: {}".format(epoch, np.mean(valid_loss)))

x_valid = x_valid.to(device)
for i in range(x_valid.shape[0]):
    predict_y = torch.argmax(model(x_valid[i]))
    plt.imshow(x_valid[i].cpu().numpy().reshape(28, 28), cmap="gray")
    plt.title("label: {}, predicted: {}".format(y_valid[i].data.numpy(), predict_y.cpu().data.numpy()))
    plt.show()
