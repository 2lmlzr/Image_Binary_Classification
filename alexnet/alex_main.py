import numpy as np 
import glob
import torch
from torch.autograd import variable
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import tqdm
from PIL import Image
from tqdm import tqdm
from zmq import device
from Classification.alexnet.alex_model import AlexNet
from Classification.alexnet.alex_dataloader import *
# import wandb
# wandb.login()


model = AlexNet()
model.cuda()

train_dir = glob.glob("cats_and_dogs_small/train/*/*.jpg")
test_dir = glob.glob("cats_and_dogs_small/test/*/*.jpg")
val_dir = glob.glob("cats_and_dogs_small/validation/*/*.jpg")

train_cat_files = [i for i in train_dir if 'cat' in i.split('/')[-1]]
train_dog_files = [i for i in train_dir if 'dog' in i.split('/')[-1]]
test_cat_files = [i for i in test_dir if 'cat' in i.split('/')[-1]]
test_dog_files = [i for i in test_dir if 'dog' in i.split('/')[-1]]
val_cat_files = [i for i in val_dir if 'cat' in i.split('/')[-1]]
val_dog_files = [i for i in val_dir if 'dog' in i.split('/')[-1]]

train_transform = transforms.Compose([
transforms.RandomRotation(30), 
transforms.RandomResizedCrop(224), 
transforms.RandomHorizontalFlip(), 
transforms.ToTensor(), 
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
transforms.Resize(255), 
transforms.CenterCrop(224), 
transforms.ToTensor(), 
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_cats = CatDogDataset(train_cat_files, transform=train_transform)
train_dogs = CatDogDataset(train_dog_files, transform=train_transform)
test_cats = CatDogDataset(test_cat_files, transform=test_transform)
test_dogs = CatDogDataset(test_dog_files, transform=test_transform)
val_cats = CatDogDataset(val_cat_files, transform=test_transform)
val_dogs = CatDogDataset(val_dog_files, transform=test_transform)

train_catdogs = ConcatDataset([train_cats, train_dogs])
test_catdogs = ConcatDataset([test_cats, test_dogs])
val_catdogs = ConcatDataset([val_cats, val_dogs])

train_dataloader = DataLoader(train_catdogs, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_catdogs, batch_size=8, shuffle=False)
val_dataloader = DataLoader(val_catdogs, batch_size=8, shuffle=False)


########################### Train ###########################
epochs = 2
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    loss_history = []
    
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
        loss_history.append(epoch_loss)
        
    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))


########################### validation ########################### 
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


########################### Eval ########################### 
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


#wandb.init(project="ml", entity="mlzr2")
        

if __name__ == "__main__":
    train_x, train_y = next(iter(train_dataloader))
    test_x, test_y = next(iter(test_dataloader))

    train_x, train_y = next(iter(train_dataloader))
    test_x, test_y = next(iter(test_dataloader))

    print(train_y, test_y)
# %%