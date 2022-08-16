# Classification


## Problem Statement
Implementing AlexNet for classifying dog and cat images.
The total number of images available for training is 2,000 and final testing was done on 500 images.



## Dependencies (To be updated)
- Pytorch
- Python 3.8
- Numpy
## Data Augmentation
- RandomRotation
- RandomResizedCrop
- RandomHorizontalFlip

```python
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
```


## Conclusion
The Architecture and parameter used in this network are capable of producing accuracy of 74% on test dataset.
Need to achieving more accuracy with fine tuning and additional dataset.
These codes are not fully modularization yet, so I will seperate train.py and test.py from main.py module.
