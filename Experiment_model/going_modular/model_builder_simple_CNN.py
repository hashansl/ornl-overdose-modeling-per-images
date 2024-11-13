"""
Contains PyTorch model code to simple CNN
https://www.youtube.com/watch?v=iG8B7x_prLQ
"""
import torch
from torch import nn 
import torchinfo
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=30, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(64*53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        # print('first conv:', x.shape)
        x = self.pool(x)
        # print('first pool:', x.shape)

        x = F.relu(self.conv2(x))
        # print('second conv:', x.shape)

        # same pooling or conv layer can be used
        x = self.pool(x)
        # print('second pool:', x.shape)

        x =torch.flatten(x, 1)
        # print('flatten:', x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x



class AdvancedCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Adaptive pooling to make the fully connected layer size independent of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers with dropout for regularization
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Adaptive pooling and flattening
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# if __name__ == "__main__":
#     config_name = 50
# #     se_resnext = SEResNeXt(config_name)
#     image = torch.rand(8, 15, 224, 224)
# #     print(se_resnext(image).shape)

#     se_resnet = SimpleCNN(15,2)
#     print(se_resnet(image).shape)

#     # Print a summary using torchinfo 
#     torchinfo.summary(model=se_resnet, 
#                       input_data=image, 
#                       col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
#                       col_width=16)

    