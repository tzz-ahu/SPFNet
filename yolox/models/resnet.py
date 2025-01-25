# ResNet50.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import Focus

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, groups=1):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        mid_channels = out_channels // 4
        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride, activation=False) \
            if in_channels != out_channels else nn.Identity()
        self.conv = nn.Sequential(*[
            Conv(in_channels, mid_channels, kernel_size=1, stride=1),
            Conv(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=groups),
            Conv(mid_channels, out_channels, kernel_size=1, stride=1, activation=False)
        ])

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        return F.relu(y, inplace=True)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.stem = Focus(3, 32, ksize=3, act="silu")
        
        self.dark2 = nn.Sequential(*self._make_stage(32, 64, down_sample=True, num_blocks=3))
        self.dark3 = nn.Sequential(*self._make_stage(64, 128, down_sample=True, num_blocks=4))
        self.dark4 = nn.Sequential(*self._make_stage(128, 256, down_sample=True, num_blocks=6))
        self.dark5 = nn.Sequential(*self._make_stage(256, 512, down_sample=True, num_blocks=3))
        
    @staticmethod
    def _make_stage(in_channels, out_channels, down_sample, num_blocks):
        layers = [Bottleneck(in_channels, out_channels, down_sample=down_sample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, out_channels, down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x,y):
        out = []
        x = self.stem(x,y)
        out.append(x)
        x = self.dark2(x)
        out.append(x)
        x = self.dark3(x)
        out.append(x)
        x = self.dark4(x)
        out.append(x)
        x = self.dark5(x)
        out.append(x)
        return out

