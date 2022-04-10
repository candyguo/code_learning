import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import datetime
from sklearn import preprocessing

# torch.hub.list("pytorch/vision:v0.8.2")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

temper_feature = pd.read_csv('./temps.csv')

# convert to datetime format
# years = temper_feature['year']
# months = temper_feature['month']
# days = temper_feature['day']

# dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# draw temper figure
# plt.subplots(2, 2, figsize=(10, 10))
# ax1 = plt.subplot(2, 2, 1)
# ax1.plot(dates, temper_feature['actual'])
# ax1.set_xlabel("date"); ax1.set_ylabel("temper"); ax1.set_title("max temp")
# ax2 = plt.subplot(2, 2, 2)
# ax2.plot(dates, temper_feature["temp_1"])
# ax2.set_xlabel("date"); ax2.set_ylabel("temper"); ax2.set_title("previous max temp")
# ax3 = plt.subplot(2, 2, 3)
# ax3.plot(dates, temper_feature["temp_2"])
# ax3.set_xlabel("date"); ax3.set_ylabel("temper"); ax3.set_title("two day prior max temp")
# ax4 = plt.subplot(2, 2, 4)
# ax4.plot(dates, temper_feature["friend"])
# ax4.set_xlabel("date"); ax4.set_ylabel("temper"); ax4.set_title("friend estimate max temp")
# plt.tight_layout(pad = 2)
# plt.show()

temper_feature = pd.get_dummies(temper_feature)
# output label
labels = np.array(temper_feature['actual'], dtype = np.float)
# input feature
temper_feature = temper_feature.drop('actual', axis = 1)
# cache column name
temper_feature_list = list(temper_feature.columns)
temper_feature = np.array(temper_feature, dtype = np.float)

# 通过预处理对输入特征做归一化
temper_feature = preprocessing.StandardScaler().fit_transform(temper_feature)

input_size = temper_feature.shape[1]
hidden_size = 128
output_size = 1
# deine moduel
class TemperModel(nn.Module):
    def __init__(self):
        super(TemperModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = torch.nn.functional.sigmoid(self.linear1(x))
        out = self.linear2(out)
        return out

temper_model = TemperModel()
cost = nn.MSELoss()
optimizer = optim.Adam(temper_model.parameters(), lr = 0.001)

batch_size = 16
for epoch in range(1000):
    batch_loss = []
    for start in range(0, len(temper_feature), batch_size):
        end = start + batch_size if start + batch_size < len(temper_feature) else len(temper_feature)
        input = torch.tensor(temper_feature[start:end], dtype=torch.float)
        output = torch.tensor(labels[start:end],dtype=torch.float)
        loss = cost(temper_model(input), output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    batch_loss.append(loss.item())
    
    if epoch % 100 == 0:
        print("epoch: {}, loss: {}".format(epoch, np.mean(batch_loss)))

predicted = temper_model(torch.tensor(temper_feature, dtype=torch.float)).data.numpy()
plt.plot(labels, 'b-', label = 'actual')
plt.plot(predicted, 'ro', label = 'predicted')
plt.show()        
    



