
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import glob
import numpy as np
import matplotlib.pyplot as plt
from xcep_dataloader import CustomDataset
import wandb
from xcep_model import model


# All train/validation/test file names regardless of cat of dog -> need to modularization
all_train_jpg_list = glob.glob("/home/mlzr/Classification/xceptionnet/cats_and_dogs_small/train/*/*.jpg")
all_val_jpg_list = glob.glob("/home/mlzr/Classification/xceptionnet/cats_and_dogs_small/validation/*/*.jpg")
all_test_jpg_list = glob.glob("/home/mlzr/Classification/xceptionnet/cats_and_dogs_small/test/*/*.jpg")

# Split cat and dog by file names -> need to modularization
train_cat_jpg = [i for i in all_train_jpg_list if 'cat' in i.split('/')[-1]]
train_dog_jpg = [i for i in all_train_jpg_list if 'dog' in i.split('/')[-1]]
val_cat_jpg = [i for i in all_val_jpg_list if 'cat' in i.split('/')[-1]]
val_dog_jpg = [i for i in all_val_jpg_list if 'dog' in i.split('/')[-1]]
test_cat_jpg = [i for i in all_test_jpg_list if 'cat' in i.split('/')[-1]]
test_dog_jpg = [i for i in all_test_jpg_list if 'dog' in i.split('/')[-1]]

# Concat cat and dog images by train/validation/test
train_catdog_jpg = train_cat_jpg + train_dog_jpg
val_catdog_jpg = val_cat_jpg + val_dog_jpg
test_catdog_jpg = test_cat_jpg + test_dog_jpg

# Define Image Transform function -> need to modularization
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),])
# val_transform = transforms.Compose([
#     transforms.Resize(299),
#     transforms.ToTensor(),])
test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),])

train_catdog = CustomDataset(train_catdog_jpg, transform=train_transform)
val_catdog = CustomDataset(val_catdog_jpg, transform=test_transform)
test_catdog = CustomDataset(test_catdog_jpg, transform=test_transform)

train_dataloader = DataLoader(train_catdog, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_catdog, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_catdog, batch_size=8, shuffle=False)



criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 30

best_models = []

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    avg_accuracy = 0

########################### Train ###########################
    for i, (input, label) in enumerate(train_dataloader):
        model.train()
        input, label = input.cuda(), label.cuda()
        train_output = model(input)

        loss = criterion(train_output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 20 == 0 :
            print(f'Epoch: {epoch} - Loss: {loss:.6f}')

########################### validation ########################### 
    val_loss_list = []
    val_acc_list = []
    for i, (input, target) in enumerate(val_dataloader):
        model.eval()
        input, target = input.cuda(), target.cuda()

        with torch.no_grad():
            val_output = model(input)
            val_loss = criterion(val_output, target).detach().cpu().numpy()

            preds = torch.argmax(val_output, axis=1)
            preds = preds.detach().cpu().numpy()

            target = target.detach().cpu().numpy()
            batch_acc = (preds==target).mean()

            val_loss_list.append(val_loss)
            val_acc_list.append(batch_acc)

    val_loss_mean = np.mean(val_loss_list)
    val_acc_mean = np.mean(val_acc_list)

    print(f'Epoch: {epoch}, valid loss: {val_loss_mean:.6f}, valid acc: {val_acc_mean:.6f}')

    val_acc_th = 0.7
    val_loss_th = 0.5

    # best model save based on loss
    if val_loss_th > val_loss_mean:
        valid_th = val_loss_mean
        best_models.append(model)

    # best model save based on accuracy
    if val_acc_th < val_acc_mean:
        val_acc_max = val_acc_mean
        best_models.append(model)
    


if __name__ == "__main__":
    # print("Train list:", all_train_jpg_list[ :4], all_train_jpg_list[-4: ])
    # print("Validation list:", all_val_jpg_list[ :4], all_val_jpg_list[-4: ])
    # print("Test list:", all_test_jpg_list[ :4], all_test_jpg_list[-4: ])

    # print("Train dataset length:", len(all_train_jpg_list))
    # print("Validation dataset length:", len(all_val_jpg_list))
    # print("Test dataset length:", len(all_test_jpg_list))
    
    # print("Train cat file name list:", train_cat_jpg[ :4], train_cat_jpg[-4: ])
    # print("Train dog file name list:", train_dog_jpg[ :4], train_dog_jpg[-4: ])
    # print("Validation cat file name list:", train_cat_jpg[ :4], train_cat_jpg[-4: ])
    # print("Validation dog file name list:", val_dog_jpg[ :4], val_dog_jpg[-4: ])
    # print("Test cat file name list:", test_cat_jpg[ :4], test_cat_jpg[-4: ])
    # print("Test dog file name list:", test_dog_jpg[ :4], test_dog_jpg[-4: ])

    # print("Train dataloader length:", len(train_dataloader))
    # print("Validation dataloader length:", len(val_dataloader))
    # print("Test dataloader length:", len(test_dataloader))
    
    # wandb.init(project='Classification-xceptionnet', entity='mlzr2')

    # sample_intput, sample_label = next(iter(train_dataloader))
    # print("sample_input:", sample_intput)
    # print("sample_label:", sample_label)
    a=1
