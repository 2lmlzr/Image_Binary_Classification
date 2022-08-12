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


class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode= mode
        self.transform = transform
        if self.mode == 'train':
            if 'dog' in self.file_list[0].split('/')[-1]:
                self.label = 1
            else:
                self.label = 0
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.dir[idx])

        if self.transform:
            img = self.transform(img)
        # if self.mode == 'train':
        img = img.numpy()
        return img.astype('float32'), self.label
        # else:
        #     img = img.numpy()
        #     return img.astype('float32'), self.file_list[idx]
        

if __name__ == "__main__":

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


    tr_x, tr_y = next(iter(tr_dl))
    tst_x, tst_y = next(iter(tst_dl))

    print(tr_y.size())