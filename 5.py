from cProfile import label
import os
from sched import scheduler
from turtle import color
from cv2 import transform
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def im_convert(tensor):
    image = tensor.numpy().squeeze() # c,h,w
    image = image.transpose(1, 2, 0) # h,w,c
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


train_dir = './flower_data/train'
valid_dir = './flower_data/valid'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45), #正负45度随机旋转
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.025), #一定的概率转换成灰度图像
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
}

batch_size = 8
train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
valid_dataset = datasets.ImageFolder(valid_dir, data_transforms['valid'])
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True)
class_names = train_dataset.classes

idx_class_map = {}
for k, v in train_dataset.class_to_idx.items():
    idx_class_map[v] = int(k)


# 读取标签对应的实际名称
with open('./cat_to_name.json') as f:
    cat_to_name = json.load(f)

# # show train image and label
# for inputs, classes in train_dataloader:
#     fig = plt.figure(figsize=(20, 12))
#     for idx in range(batch_size):
#         ax = fig.add_subplot(2, 4, idx+1)
#         ax.set_title(cat_to_name[str(classes[idx].numpy())])
#         plt.imshow(im_convert(inputs[idx]))
#     plt.show()  

model_ft = models.resnet18(pretrained=True)

# we will finetune model, so freeze backbone model parameters update
for paramater in model_ft.parameters():
    paramater.requires_grad = False
# 替换最后一个fc以适应当前任务    
last_fc_features = model_ft.fc.in_features        
model_ft.fc = nn.Sequential(nn.Linear(last_fc_features, len(class_names)),
                            nn.LogSoftmax(dim=1))
model_ft = model_ft.to(device)

print("param to learn:")
params_to_update = []
for name, param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print(name)

# 设置优化器和损失
optimizer = optim.Adam(model_ft.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.NLLLoss()

def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer):
    since = time.time()
    best_acc = 0
    model = model.to(device)
    
    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_hisotry = []

    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(25):
        # train
        running_loss = 0.0
        running_correct = 0

        model.train()
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data)
        
        time_elapsed = time.time() - since
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_correct / len(train_dataset)
        print("time elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("epoch: {}, loss: {}, acc: {}".format(epoch, running_loss, epoch_acc))
        
        # valid
        running_correct = 0
        running_loss = 0
        model.eval()
        for inputs, labels in valid_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_correct / len(valid_dataset)
        # scheduler step
        scheduler.step()
   
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_state = copy.deepcopy(model.state_dict())
            state = {
                "state_dict":  model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "./checkpoint.pth")

    model.load_state_dict(best_model_state)
    return model

# trainning
model_ft = train_model(model_ft, train_dataloader, valid_dataloader, criterion, optimizer)
print("initial train finished")    

checkpoint = torch.load("./checkpoint.pth")
best_acc = checkpoint['best_acc']
print("initial freeze best acc: {}".format(best_acc))

# 在继续训练所有层
for param in model_ft.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model_ft.parameters(), lr=0.0001)
optimizer.load_state_dict(checkpoint['optimizer'])
model_ft = train_model(model_ft, train_dataloader, valid_dataloader, criterion, optimizer)

# load trained model then inference
checkpoint = torch.load("./checkpoint.pth")
best_acc = checkpoint['best_acc']
print("final finetune best acc: {}".format(best_acc))
model_ft.load_state_dict(checkpoint['state_dict'])

# predict
for inputs, labels in valid_dataloader:
    inputs = inputs.to(device)
    outputs = model_ft(inputs)
    _, preds = torch.max(outputs, dim=1)


    fig = plt.figure(figsize=(20, 20))    
    for idx in range(8):
        ax = fig.add_subplot(2, 4, idx + 1)
        ax.imshow(im_convert(inputs[idx].cpu()))
        gt_label = cat_to_name[str(idx_class_map[labels[idx].cpu().item()])]
        predict_label = cat_to_name[str(idx_class_map[preds[idx].cpu().item()])]
        ax.set_title("label: {}, predict: {}".format(gt_label, 
                                                     predict_label),
                    color = ("green" if gt_label == predict_label else "red"))
    plt.show()        





