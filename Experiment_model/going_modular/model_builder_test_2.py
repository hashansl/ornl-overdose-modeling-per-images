# """
# Contains PyTorch model code to instantiate SE-ResNet 
# """

import torch
from torch import nn
import torchinfo

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.c(x))

# Squeeze-and-Excitation Block
class SeBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, C // r, bias=False)
        self.fc2 = nn.Linear(C // r, C, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f = self.globpool(x)
        f = torch.flatten(f, 1)
        f = self.relu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))
        f = f[:, :, None, None]
        return x * f

# Experiment Model with SE Block
class ExperimentNet(nn.Module):
    def __init__(
        self, 
        in_channels: int = 15,  # Initial number of channels
        classes: int = 2,
        r: int = 16
    ):
        super().__init__()
        res_channels = int(in_channels // 4)

        self.seblock = SeBlock(in_channels, r=r)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Downscales to (N, C, 1, 1)
        self.fc = nn.Linear(in_channels, classes)
        # self.init_weight()

    # def forward(self, x):
    #     x = self.seblock(x)
    #     x = self.adaptive_pool(x)
    #     x = torch.flatten(x, 1)  # Resulting in shape [N, in_channels]
    #     x = self.fc(x)
    #     return x

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.seblock(x)
        print("After SeBlock shape:", x.shape)
        x = self.adaptive_pool(x)
        print("After Adaptive Pool shape:", x.shape)
        x = torch.flatten(x, 1)
        print("After Flatten shape:", x.shape)
        x = self.fc(x)

        return x

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

if __name__ == "__main__":
    image = torch.rand(1, 15, 224, 224)  # Sample input tensor
    Rsnet = ExperimentNet(in_channels=15, r=16)

    # Print a summary using torchinfo 
    torchinfo.summary(
        model=Rsnet,
        input_data=image,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=16
    )
