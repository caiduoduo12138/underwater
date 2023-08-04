import torch
import torch.nn.functional as F
import torch.nn as nn

import math

import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage



class Dilated (torch.nn.Module):

    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        dilations = [1, 3, 6, 1]
        paddings = [0, 3, 6, 0]
        self.da = torch.nn.ModuleList()
        for da_idx in range(len(kernel_sizes)):
            conv = torch.nn.Conv2d(
                in_channels,
                out_channels//4,
                kernel_size=kernel_sizes[da_idx],
                stride=1,
                dilation=dilations[da_idx],
                padding=paddings[da_idx],
                bias=True)
            self.da.append(conv)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.da_num = len(kernel_sizes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for da_idx in range(self.da_num):
            inp = avg_x if (da_idx == self.da_num - 1) else x
            out.append(F.relu_(self.da[da_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        # out = torch.cat(out, dim=1)
        x_1 = out[0]
        x_2 = F.max_pool2d(out[1], 5, stride=1, padding=2)
        x_3 = F.max_pool2d(out[2], 9, stride=1, padding=4)
        x_4 = F.max_pool2d(out[3], 13, stride=1, padding=6)
        out = torch.cat((x_1, x_2, x_3, x_4),dim=1) + x

        return out


class MaxFiltering(nn.Module):
    def __init__(self, in_channels=256, kernel_size=3, tau=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm = nn.GroupNorm(2, in_channels)
        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool3d(
            kernel_size=(tau + 1, kernel_size, kernel_size),
            padding=(tau // 2, kernel_size // 2, kernel_size // 2),
            stride=1
        )
        self.margin = tau // 2

    def draw(self, x):
        np.random.seed(0)
        sns.set_theme()
        for each in x:
            sns.heatmap(each[0,0,:,:].cpu().detach().numpy(), cbar=False, xticklabels=False, yticklabels=False)
            plt.show()


    def forward(self, inputs):
        features = []
        for l, x in enumerate(inputs):
            features.append(self.conv(x))

        outputs = []
        for l, x in enumerate(features):
            func = lambda f: F.interpolate(f, size=x.shape[2:], mode="bilinear", align_corners=True)
            feature_3d = []
            for k in range(max(0, l - self.margin), min(len(features), l + self.margin + 1)):
                feature_3d.append(func(features[k]) if k != l else features[k])
            feature_3d = torch.stack(feature_3d, dim=2)
            max_pool = self.max_pool(feature_3d)[:, :, min(l, self.margin)]
            output = max_pool + inputs[l]
            outputs.append(self.nonlinear(self.norm(output)))
        # self.draw(outputs)
        return outputs
