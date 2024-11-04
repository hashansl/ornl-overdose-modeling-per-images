"""
Contains PyTorch model code to instantiate SE-ResNet 
"""
import torch
from torch import nn 
import torchinfo


# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        # Always call super().__init__() in any subclass of nn.Module to ensure proper initialization and compatibility with the PyTorch framework. 
        # It sets up everything nn.Module needs to manage the layer’s parameters and behavior within the model.
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        # By default, groups=1, which means it’s a regular convolution.
        self.bn = nn.BatchNorm2d(out_channels)
        # During training, for each mini-batch, it computes the mean and variance of each channel in the output.
        # It then normalizes the activations by subtracting the mean and dividing by the standard deviation, 
        # which scales the output to have a mean of 0 and a variance of 1.

    def forward(self, x):
    # The forward function defines the forward pass or the data flow through the layers.
        return self.bn(self.c(x))

        
# Block of ResNet
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, r, first=False):
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
        # self.seblock = SeBlock(out_channels, r=r)


    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)
        f = self.seblock(f)

        if self.projection:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h


# # SE-ResNet
# class SEResNet(nn.Module):
#     def __init__(
#         self, 
#         config_name : int, 
#         in_channels : int = 15, # intial number of channels is 3 
#         classes : int = 2,
#         r : int = 16
#         ):
#         super().__init__()

#         configurations = {
#             50 : [3, 4, 6, 3],
#             101 : [3, 4, 23, 3],
#             152 : [3, 8, 36, 3]
#         }    



if __name__ == "__main__":
    config_name = 50
#     se_resnext = SEResNeXt(config_name)
    image = torch.rand(8, 1, 224, 224)


    Rsnet = ResNetBlock(in_channels=256,out_channels=512,r=16,first=True)

    # Print a summary using torchinfo 
    torchinfo.summary(model=Rsnet, 
                      input_data=image, 
                      col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                      col_width=16)