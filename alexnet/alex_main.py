import numpy as np 
import glob
import torch
from torch.autograd import variable
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torchvision import transforms
import tqdm
from PIL import Image
from tqdm import tqdm
from zmq import device
from alex_model import AlexNet
from alex_dataloader import *

import wandb
wandb.init()


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


epochs = 50

acc_best_models = []
loss_best_models = []


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
########################### Train ###########################
    train_losses = 0
    train_accuracies = 0


    for i, (image, target) in enumerate(tqdm(train_dataloader)):
        model.train()
        image, target = image.cuda(), target.cuda()
        train_output = model(image)

        train_loss = criterion(train_output, target)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_losses += train_loss.item()

        train_accuracy = ((train_output.argmax(dim=1) == target).float().mean()) # batch accuracy
        train_accuracies += train_accuracy.item()
        if i % 250 == 249:
            print(f'Step: {i}, Loss: {train_losses / i}, Accuracy: {train_accuracies / i}')


    print(f'Epoch: {epoch}, Loss: {train_losses / len(train_dataloader)}, Accuracy: {train_accuracies / len(train_dataloader)}')
        # if (i+1) % 50 == 0 :
        #     print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        #     wandb.log({'train loss':train_loss, 'average train accuracy':train_accuracy})

########################### Validation ########################### 
    val_losses = 0
    val_accuracies = 0
    for i, (images, target) in enumerate(tqdm(val_dataloader)):
        model.eval()
        images, target = images.cuda(), target.cuda()

        with torch.no_grad():
            val_output = model(images)
            val_loss = criterion(val_output, target)

            val_preds = val_output.argmax(dim=1)
            # val_preds = val_preds.detach().cpu().numpy()

            val_losses += val_loss.item()
            val_accuracy = (val_preds == target).float().mean() # batch accuracy
            val_accuracies += val_accuracy.item()

    print(f'Epoch: {epoch}, validation loss: {val_losses / len(val_dataloader)}, validation accuracy: {val_accuracies / len(val_dataloader)}')
    wandb.log({'average validation loss':val_losses/len(val_dataloader), 'average validation accuracy':val_accuracies/len(val_dataloader)})

    val_loss_th = np.inf
    val_acc_th = 0

    # best model save based on loss
    if val_loss_th > val_losses:
        valid_th = val_losses
        loss_best_models.append(model)

    # best model save based on accuracy
    if val_acc_th < val_accuracies:
        val_acc_max = val_accuracies
        acc_best_models.append(model)
    
########################### Test ########################### 
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

model.eval()

for data, target in test_dataloader:
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)    
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    for i in range(8): # 배치 사이즈로
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss/len(test_dataloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))
classes = ['cat', 'dog']
for i in range(2):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))



if __name__ == "__main__":

    train_x, train_y = next(iter(train_dataloader))
    test_x, test_y = next(iter(test_dataloader))

    train_x, train_y = next(iter(train_dataloader))
    test_x, test_y = next(iter(test_dataloader))

    print(train_x.size())
# %%