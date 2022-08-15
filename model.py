import torch
import torch.nn as nn
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F

# CNN 작성
class AlexNet(nn.Module):
    def __init__(self):
      super(AlexNet, self).__init__()

      # input image = 224 x 244 x 3

      # 224 x 224 x 3 --> 112 x 112 x 32 maxpool
      self.conv1 = nn.Conv2d(3, 32, 3, padding=1) 
      # 112 x 112x 32 --> 56 x 56 x 64 maxpool
      self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
      # 56 x 56 x 64 --> 28 x 28 x 128 maxpool
      self.conv3 = nn.Conv2d(64, 128, 3, padding=1)     
      # 28 x 28 x 128 --> 14 x 14 x 256 maxpool
      self.conv4 = nn.Conv2d(128, 256, 3, padding=1)    

      # maxpool 2 x 2
      self.pool = nn.MaxPool2d(2, 2)
      
      # 28 x 28 x 128 vector flat 512개
      self.fc1 = nn.Linear(256 * 14 * 14, 512)
      # 카테고리 2개 클래스
      self.fc2 = nn.Linear(512, 2) 
      
      # dropout 적용
      self.dropout = nn.Dropout(0.5) # 0.25 해보고 0.5로 해보기. 값 저장하고나서

    def forward(self, x):
      # conv1 레이어에 relu 후 maxpool. 112 x 112 x 32
      x = self.pool(F.relu(self.conv1(x)))
      # conv2 레이어에 relu 후 maxpool. 56 x 56 x 64
      x = self.pool(F.relu(self.conv2(x)))
      # conv3 레이어에 relu 후 maxpool. 28 x 28 x 128
      x = self.pool(F.relu(self.conv3(x)))
      # conv4 레이어에 relu 후 maxpool. 14 x 14 x 256
      x = self.pool(F.relu(self.conv4(x)))

      # 이미지 펴기
      x = x.view(-1, 256 * 14 * 14) 
      # dropout 적용
      x = self.dropout(x)
      # fc 레이어에 삽입 후 relu
      x = F.relu(self.fc1(x))
      # dropout 적용
      x = self.dropout(x)
      
      # 마지막 logsoftmax 적용
      x = F.log_softmax(self.fc2(x), dim=1)
      return x

model = AlexNet() # 모델 생성
print(model) # 출력
model.cuda() # cuda 사용


# class AlexNet(nn.Module) :
#     def __init__(self) -> None :
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
        
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256*6*6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 1),)
        
#     def forward(self, x :torch.Tensor) -> torch.Tensor :
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 2)
#         x = self.classifier(x)
#         return x


if __name__ == "__main__":
    model = AlexNet()
    print(model)
    
