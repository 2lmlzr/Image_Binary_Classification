
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
wandb.init()
from xcep_model import model
from tqdm import tqdm


# All train/validation/test file names regardless of cat of dog 
all_train_jpg_list = glob.glob("/home/mlzr/Classification/xceptionnet/cats_and_dogs_small/train/*/*.jpg")
all_val_jpg_list = glob.glob("/home/mlzr/Classification/xceptionnet/cats_and_dogs_small/validation/*/*.jpg")
all_test_jpg_list = glob.glob("/home/mlzr/Classification/xceptionnet/cats_and_dogs_small/test/*/*.jpg")

# Split cat and dog by file names
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

# Define Image Transform function
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
val_dataloader = DataLoader(val_catdog, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_catdog, batch_size=16, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100

acc_best_models = []
loss_best_models = []

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

    # sample_input, sample_label = next(iter(train_dataloader))
    # print("sample_input:", sample_input)
    # print("sample_input size:" , sample_input.size())
    # print("sample_label:", sample_label)
    a=1
