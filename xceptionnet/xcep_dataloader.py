#%%
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
    def __init__(self, file_list, transform=train_transform):
        self.file_list = file_list
        self.transform = transform
        
        # Labeling by file name
        self.label = []
        for i in self.file_list:
            if 'cat' in i.split('/')[-1]:
                self.label.append(0)
            elif 'dog' in i.split('/')[-1]:
                self.label.append(1)
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Image transformation
        img = Image.open(self.file_list[idx])
        img_trans = self.transform(img)
        return img_trans, self.label[idx]
        
if __name__ == "__main__":
    a = 1