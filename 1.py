from statistics import mode
from tkinter.tix import ListNoteBook
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class LinearRegressModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_dim = 1
output_dim = 1
model = LinearRegressModel(input_dim, output_dim)
model.to(device)

epochs = 1000
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

x = np.array(range(11), dtype=np.float32)
y = 2 * x + 1
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


for epoch in range(epochs):
    inputs = torch.from_numpy(x).to(device)
    outputs = torch.from_numpy(y).to(device)
    optimizer.zero_grad()
    loss = criterion(model(inputs), outputs)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss.item()))

for parameter in model.parameters():
    print(parameter)

predicted = model(torch.from_numpy(x).to(device)).cpu().data.numpy()
print(predicted)

torch.save(model.state_dict(), "model.kpl")
print("model save success")
os.remove("model.kpl")
