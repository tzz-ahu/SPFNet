# ???? se cbam ??? eca

import torch
import torch.nn as nn
import math
from .network_blocks import GELU,DilatedEncoder,BaseConv

# SE?????????
class SE(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # ????1x1????????????
        self.fc1 = BaseConv(in_planes*2, in_planes//8, ksize=1, stride=1, act='silu')
        self.gelu = GELU()
        self.fc2 = BaseConv(in_planes//8, in_planes, ksize=1, stride=1, act='silu')
        self.sigmoid = nn.Sigmoid()

        self.fc3 = BaseConv(in_planes*2, in_planes//8, ksize=1, stride=1, act='silu')
        self.fc4 = BaseConv(in_planes//8, in_planes, ksize=1, stride=1, act='silu')

    def forward(self, x,y):   #???? RGB , T
        cx1 = self.avg_pool(x)
        cx2 = self.max_pool(x)
        cy1 = self.avg_pool(y)
        cy2 = self.max_pool(y)
        cx = cx1 + cx2
        cy = cy1 + cy2
        csw = self.sigmoid(cx*cy)
        csw1 = torch.cat([cx,csw],1)
        csw2 = torch.cat([cy,csw],1)
        csw1 = self.fc2(self.gelu(self.fc1(csw1)))
        csw2 = self.fc4(self.gelu(self.fc3(csw2)))
        out = self.sigmoid(csw1+csw2)
        return torch.cat([out*x+x,out*y+y],1)

class SpatialAttention(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = BaseConv(8, 16, ksize=1, stride=1, act='silu')
        self.gelu = GELU()
        self.fc2 = BaseConv(16, 1, ksize=1, stride=1, act='silu')

        self.fc3 = nn.Conv2d(in_planes * 4, in_planes // ratio, 1, bias=True)
        self.fc4 = nn.Conv2d(in_planes // ratio, in_planes * 3, 1, bias=True)

        self.fc5 = nn.Conv2d(in_planes * 3, in_planes // ratio, 1, bias=True)
        self.fc6 = nn.Conv2d(in_planes // ratio, in_planes * 2, 1, bias=True)
        
        self.fc7 = nn.Conv2d(in_planes * 2, in_planes // ratio, 1, bias=True)
        self.fc8 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=True)

        self.encoder1 = DilatedEncoder(128, 128)
        self.encoder2 = DilatedEncoder(256, 256)
        self.encoder3 = DilatedEncoder(512, 512)
        self.encoder4 = DilatedEncoder(4, 1)

    def forward(self, x, y, z,m):
        sx1 = torch.mean(x, dim=1, keepdim=True)
        sy1 = torch.mean(y, dim=1, keepdim=True)
        sz1 = torch.mean(z, dim=1, keepdim=True)
        sm1 = torch.mean(m, dim=1, keepdim=True)
        sx2, _ = torch.max(x, dim=1, keepdim=True)
        sy2, _ = torch.max(y, dim=1, keepdim=True)
        sz2, _ = torch.max(z, dim=1, keepdim=True)
        sm2, _ = torch.max(m, dim=1, keepdim=True)
        
        p = torch.cat([sx1,sx2,sy1,sm1,sy2,sz1,sz2,sm2], 1)
        q = torch.cat([sx1,sy1,sz1,sm1], 1)
       
        p = self.fc2(self.gelu(self.fc1(p)))
        q = self.encoder4(q)
        w = self.sigmoid(p*q)   
        
        sz = torch.cat([x, y * w + y, z * w + z,m * w + m], 1)
        sz = self.fc4(self.gelu(self.fc3(sz))) + torch.cat([x, y,z], 1)
        sz = self.fc6(self.gelu(self.fc5(sz))) + torch.cat([x, y], 1)
        sz = self.fc8(self.gelu(self.fc7(sz))) + x
        return sz

# CBAM?????????
class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(channel,kernel_size=kernel_size)

    def forward(self, x,y,z,m):

        return self.spatialattention(x,y,z,m)
### ECA?????????
class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
