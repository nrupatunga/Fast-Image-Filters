"""
File: basic_blocks.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Basic building block for model
"""
import numpy as np
import torch
import torch.nn as nn


class AdaptiveBatchNorm2d(nn.Module):

    """Adaptive batch normalization"""

    def __init__(self, num_feat, eps=1e-5, momentum=0.1, affine=True):
        """Adaptive batch normalization"""
        super().__init__()
        self.bn = nn.BatchNorm2d(num_feat, eps, momentum, affine)
        self.a = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)


class ConvBlock(nn.Module):

    """Convolution head"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 dilation: int,
                 norm_layer: nn.Module = AdaptiveBatchNorm2d):
        """
        @in_channels: number of input channels
        @out_channels: number of output channels
        @dilation: dilation factor @activation: 'relu'- relu,
        'lrelu': leaky relu
        @norm_layer: 'bn': batch norm, 'in': instance norm, 'gn': group
        norm, 'an': adaptive norm
        """
        super().__init__()
        convblk = []

        convblk.extend([
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation),
            nn.LeakyReLU(negative_slope=0.2),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity()])

        self.convblk = nn.Sequential(*convblk)
        self.init_weights(self.convblk)

    def identity_init(self, shape):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[2] // 2, shape[3] // 2
        for i in range(np.minimum(shape[0], shape[1])):
            array[i, i, cx, cy] = 1

        return array

    def init_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                weights = self.identity_init(m.weight.shape)
                with torch.no_grad():
                    m.weight.copy_(torch.from_numpy(weights).float())
                torch.nn.init.zeros_(m.bias)

    def forward(self, *inputs):
        return self.convblk(inputs[0])
