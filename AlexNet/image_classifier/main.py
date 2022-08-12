import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import copy
import tqdm
from PIL import Image
from tqdm import tqdm
from model import AlexNet
from dataloader import CatDogDataset


def train(epochs:int, dataloader, model, optimizer, criterion):
    loss_list = []
    acc_list = []
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        total_acc = 0

        for step, (samples, labels) in enumerate(tqdm(dataloader)):
            samples, labels = samples.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # scheduler.step()
            
            # if step % 50 == 0:
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(labels)
            acc = torch.mean(correct.float())
        print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, step, total_loss / len(dataloader), acc))
        # loss_list.append(total_loss / 50)
        # acc_list.append(acc.cpu())
        total_loss = 0
            
            # itr += 1


if __name__ == "__main__":
    epochs = 70
    model = AlexNet()
    model.cuda()

    train_dir = glob.glob("data/train/*/*.jpg")
    val_dir = glob.glob("data/validation/*/*.jpg")
    test_dir = glob.glob("data/test/*/*.jpg")

        
    cat_files = [tf for tf in train_dir if 'cat' in tf.split('/')[-1]]
    dog_files = [tf for tf in train_dir if 'dog' in tf.split('/')[-1]]

    tst_cat_files = [tf for tf in test_dir if 'cat' in tf.split('/')[-1]]
    tst_dog_files = [tf for tf in test_dir if 'dog' in tf.split('/')[-1]]

    data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])


    cats = CatDogDataset(cat_files, train_dir, transform=data_transform)
    dogs = CatDogDataset(dog_files, train_dir, transform=data_transform)

    tst_cats = CatDogDataset(tst_cat_files, test_dir, transform=test_transform)
    tst_dogs = CatDogDataset(tst_dog_files, test_dir, transform=test_transform)

    catdogs = ConcatDataset([cats, dogs])
    tst_catdogs = ConcatDataset([tst_cats, tst_dogs])

    tr_dl = DataLoader(catdogs, batch_size=16, shuffle=True)
    tst_dl = DataLoader(tst_catdogs, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)
    train(epochs, tr_dl, model, optimizer, criterion)