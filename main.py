#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F 
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import copy
import tqdm
from PIL import Image
from tqdm import tqdm
from zmq import device
from model import AlexNet
from dataloader import *

model = AlexNet()
model.cuda()

train_dir = glob.glob("data/train/*/*.jpg")
test_dir = glob.glob("data/test/*/*.jpg")
val_dir = glob.glob("data/validation/*/*.jpg")

train_cat_files = [i for i in train_dir if 'cat' in i.split('/')[-1]]
train_dog_files = [i for i in train_dir if 'dog' in i.split('/')[-1]]
test_cat_files = [i for i in test_dir if 'cat' in i.split('/')[-1]]
test_dog_files = [i for i in test_dir if 'dog' in i.split('/')[-1]]
val_cat_files = [i for i in val_dir if 'cat' in i.split('/')[-1]]
val_dog_files = [i for i in val_dir if 'dog' in i.split('/')[-1]]

train_transform = transforms.Compose([
transforms.Resize(256),
transforms.ColorJitter(),
transforms.RandomCrop(224),
transforms.RandomHorizontalFlip(),
transforms.Resize(256),
transforms.ToTensor()
])

test_transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.ToTensor()
])

train_cats = CatDogDataset(train_cat_files, transform=train_transform)
train_dogs = CatDogDataset(train_dog_files, transform=train_transform)
test_cats = CatDogDataset(test_cat_files, transform=test_transform)
test_dogs = CatDogDataset(test_dog_files, transform=test_transform)
val_cats = CatDogDataset(val_cat_files, transform=test_transform)
val_dogs = CatDogDataset(val_dog_files, transform=test_transform)

train_catdogs = ConcatDataset([train_cats, train_dogs])
test_catdogs = ConcatDataset([test_cats, test_dogs])
val_catdogs = ConcatDataset([val_cats, val_dogs])

train_dataloader = DataLoader(train_catdogs, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_catdogs, batch_size=32, shuffle=False)
val_dataloader = DataLoader(val_catdogs, batch_size=32, shuffle=False)


### Train ###
epochs = 2

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label in train_dataloader:
        data = data.cuda()
        label = label.cuda()
        
        output = model(data)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_dataloader)
        epoch_loss += loss/len(train_dataloader)
        
    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))

    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in val_dataloader:
            data = data.cuda()
            label = label.cuda()
            
            val_output = model(data)
            val_loss = criterion(val_output,label)
        
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val_dataloader)
            epoch_val_loss += val_loss/ len(val_dataloader)
            
        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))
print(" Train over ")


test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
classes = ['cat', 'dog']

model.eval()

for data, target in test_dataloader:
    data, target = data.cuda(), target.cuda()
    output = model(data)

    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.detach().cpu().numpy())

    for i in range(8):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss/len(test_dataloader.dataset)
print('Test Loss: {:.6f}\n' .format(test_loss))

for i in range(2):
    if class_total[i] > 0 :
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100*class_correct[i]/class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))




# plt.plot(train_losses, label = 'loss')
# plt.plot(train_accuracies, label = 'accuracy')
# plt.legend()
# plt.title('train loss and accuracy')
# plt.show()

# plt.plot(val_losses, label = 'loss')
# plt.plot(val_accuracies, label = 'accuracy')
# plt.legend()
# plt.title('validation loss and accuracy')
# plt.show()
        

if __name__ == "__main__":
    train_x, train_y = next(iter(train_dataloader))
    test_x, test_y = next(iter(test_dataloader))

    train_x, train_y = next(iter(train_dataloader))
    test_x, test_y = next(iter(test_dataloader))


# %%