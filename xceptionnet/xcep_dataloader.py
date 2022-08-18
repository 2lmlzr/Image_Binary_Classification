import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class CustomDataset(Dataset):
    # Put cat and dog images together into CustomDataset
    def __init__(self, file_list, transform):
        # Labeling
        self.file_list = file_list
        self.transform = transform
        for i in file_list:
            if 'cat' in i.split('/')[-1]:
                self.label = 0
            elif 'dog' in i.split('/')[-1]:
                self.label = 1
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):







if __name__ == "__main__":
    a=CustomDataset()