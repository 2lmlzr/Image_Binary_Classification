import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms

# Define Image Transform function -> need to modularization
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(299),
    transforms.ToTensor(),])
# val_transform = transforms.Compose([
#     transforms.Resize(299),
#     transforms.ToTensor(),])
test_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),])

class CustomDataset(Dataset):
    # Put cat and dog images together into CustomDataset
    def __init__(self, file_list):
        self.file_list = file_list
        s
        
        # Labeling by file name
        for i in file_list:
            if 'cat' in i.split('/')[-1]:
                self.label = 0
            elif 'dog' in i.split('/')[-1]:
                self.label = 1

        # Image transformation
        for i in file_list:
            img = Image.open(self.file_list[i])
            if transform_mode == 'train':
                img_trans = self.train_transform
            elif transform_mode == 'test' or 'val':
                img_trans = self.test_transform


        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):








if __name__ == "__main__":
    a=CustomDataset()