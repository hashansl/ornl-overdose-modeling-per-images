# """
# Contains PyTorch model code to instantiate SE-ResNet 
# """

import torch
from torch import nn
import torchinfo
import torch.nn.functional as F


# Experiment Model with SE Block
class ExperimentNet(nn.Module):
    def __init__(self, in_channels: int = 15, classes: int = 2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, classes)

        # The fully connected (fc1) layer expects the input size to match the number of features coming out of the previous 
        # layer. After the convolutional layers, you end up with a tensor of shape [N, 64, 1, 1] 
        # (64 feature maps, each reduced to a single value). You flatten this tensor to [N, 64] for each sample in the batch.
        #  Therefore, the input size to fc1 must be 64—one value for each of the 64 feature maps produced by the last convolutional l

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        print(f"After conv1 shape: {x.shape}")
        x = self.pool(x)    

        print(f"After pool shape: {x.shape}")
        x = F.relu(self.conv2(x))

        print(f"After conv2 shape: {x.shape}")
        x = self.pool(x)

        print(f"After pool shape: {x.shape}")
        # flatten the tensor
        x = self.avgpool(x)

        print(f"After avgpool shape: {x.shape}")

        x = torch.flatten(x, 1)
    
        x = self.fc1(x)
        print(f"After fc1 shape: {x.shape}")

        print(f'-----End of forward pass-----')

        return x

    # def init_weight(self):
    #     for layer in self.modules():
    #         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #             nn.init.kaiming_normal_(layer.weight)

if __name__ == "__main__":
    image = torch.rand(64, 15, 224, 224)  # Sample input tensor
    # Rsnet = ExperimentNet(in_channels=15, r=16)
    Rsnet = ExperimentNet(in_channels=15)


    # Print a summary using torchinfo 
    torchinfo.summary(
        model=Rsnet,
        input_data=image,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=16
    )
