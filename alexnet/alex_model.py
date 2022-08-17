import torch
import torch.nn as nn
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
      super(AlexNet, self).__init__()

      self.conv1 = nn.Conv2d(3, 32, 3, padding=1) 
      self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
      self.conv3 = nn.Conv2d(64, 128, 3, padding=1)     
      self.conv4 = nn.Conv2d(128, 256, 3, padding=1)    

      self.pool = nn.MaxPool2d(2, 2)
      
      self.fc1 = nn.Linear(256 * 14 * 14, 512)
      self.fc2 = nn.Linear(512, 2) 
      
      self.dropout = nn.Dropout(0.5)

    def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = self.pool(F.relu(self.conv3(x)))
      x = self.pool(F.relu(self.conv4(x)))

      x = x.view(-1, 256 * 14 * 14) 
      x = self.dropout(x)
      x = F.relu(self.fc1(x))
      x = self.dropout(x)
      
      x = F.log_softmax(self.fc2(x), dim=1)
      return x


model = AlexNet() 
print(model) 
model.cuda() 


if __name__ == "__main__":
    model = AlexNet()
    print(model)

    
