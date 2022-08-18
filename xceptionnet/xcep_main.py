
from torchvision import transforms
import glob
from xcep_dataloader import CustomDataset


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
    transforms.Resize(299),
    transforms.ToTensor(),])
val_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),])
test_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),])









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

    # print(len(train_cat_jpg), len(train_catdog_jpg))
    haha = ['hehe/he', 'hoho/ho']
    a = CustomDataset(haha)
