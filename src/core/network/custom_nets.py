"""
File: custom_nets.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: network architecture for fast image filters
"""
import torch
import torch.nn as nn
from torchsummary import summary

from core.network.basic_blocks import ConvBlock


class FIP(nn.Module):

    """Model architecture for fast image filter"""

    def __init__(self):
        """Initialization """
        super().__init__()

        nbLayers = 24

        self.conv1 = ConvBlock(3, nbLayers, 3, 1, 1)
        self.conv2 = ConvBlock(nbLayers, nbLayers, 3, 2, 2)
        self.conv3 = ConvBlock(nbLayers, nbLayers, 3, 4, 4)
        self.conv4 = ConvBlock(nbLayers, nbLayers, 3, 8, 8)
        self.conv5 = ConvBlock(nbLayers, nbLayers, 3, 16, 16)
        self.conv6 = ConvBlock(nbLayers, nbLayers, 3, 32, 32)
        self.conv7 = ConvBlock(nbLayers, nbLayers, 3, 64, 64)
        self.conv8 = ConvBlock(nbLayers, nbLayers, 3, 1, 1)
        self.conv9 = nn.Conv2d(nbLayers, 3, kernel_size=1, dilation=1)

        self.weights_init(self.conv9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        return x

    def weights_init(self, m):
        """conv2d Init
        """
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)


if __name__ == "__main__":
    net = FIP().cuda()
    summary(net, input_size=(3, 500, 500))
