import numpy as np 
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image


class CatDogDataset(Dataset):

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(self.file_list[idx])
        img_transformed = self.transform(img)

        label = img_path.split('/')[-1][:3]
        if label == 'cat':
            label=0
        elif label == 'dog':
            label=1
        # import pdb; pdb.set_trace()

        return img_transformed, label


if __name__ == "__main__":
    a =2
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

samples, labels = iter(train_dataloader).next()
classes = {0:'cat', 1:'dog'}
fig = plt.figure(figsize=(10,24))
for i in range(16):
    a = fig.add_subplot(2,8,i+1)
    a.set_title(classes[labels[i].item()])
    a.axis('off')
    a.imshow(np.transpose(samples[i].numpy(), (1,2,0)))
plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
# %%
