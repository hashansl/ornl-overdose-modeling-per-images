# """
# Contains PyTorch model code to instantiate SE-ResNet 
# """

import torch
from torch import nn
import torchinfo
import torch.nn.functional as F

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.c(x))


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        res_channels = in_channels // 4
        stride = 1

        self.projection = in_channels!=out_channels
        if self.projection:
            self.p = ConvBlock(in_channels, out_channels, 1, 2, 0)
            stride = 2
            res_channels = in_channels // 2

        if first:
            self.p = ConvBlock(in_channels, out_channels, 1, 1, 0)
            stride = 1
            res_channels = in_channels


        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0) 
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()


    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)

        if self.projection:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h
    

# Experiment Model with SE Block
class ExperimentNet(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, C//r, bias=False)
        self.fc2 = nn.Linear(C//r, C, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.resnetblock = ResNetBlock(C, C, first=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(C, 2)

    def forward(self, x):
        # print(f"Input shape: {x.shape}") 

        f = self.globpool(x)
        # print(f"After globpool shape: {f.shape}")
        
        f = torch.flatten(f,1)
        # print(f"After flatten shape: {f.shape}")

        f = self.relu(self.fc1(f))
        # print(f"After fc1 shape: {f.shape}")

        f = self.sigmoid(self.fc2(f))
        # print(f"After fc2 shape: {f.shape}")

        f = f[:,:,None,None]

        scale = x * f

        # print(f"After scale shape: {scale.shape}")

        # f = self.resnetblock(scale)

        f = self.avgpool(scale)
        # print(f"After avgpool shape: {x.shape}")

        f = torch.flatten(f, 1)
        # print(f"After flatten shape: {x.shape}")

        f = self.fc3(f)


        return f
        







    # def init_weight(self):
    #     for layer in self.modules():
    #         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #             nn.init.kaiming_normal_(layer.weight)

# if __name__ == "__main__":
#     image = torch.rand(32, 15, 224, 224)  # Sample input tensor
#     # Rsnet = ExperimentNet(in_channels=15, r=16)
#     Rsnet = ExperimentNet(in_channels=15)


#     # Print a summary using torchinfo 
#     torchinfo.summary(
#         model=Rsnet,
#         input_data=image,
#         col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
#         col_width=16
#     )


